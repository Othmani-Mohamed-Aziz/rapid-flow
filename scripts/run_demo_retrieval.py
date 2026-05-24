"""
Démo **indexation Qdrant + retrieval** — sans extraction ni export eCRF.

Étapes pipeline couvertes : ingestion → parsing → chunking → index hybride
(dense + BM25 + reranker) → requêtes retrieval → ``list_chunks`` / ``health_check``.

Ne passe pas par ``PipelineOrchestrator`` : outil de debug de la couche RAG.

Script complémentaire pour la suite : ``run_demo_e2e.py`` (extraction + export).

Usage :
    python scripts/run_demo_retrieval.py
    python scripts/run_demo_retrieval.py path/to/other.pdf
    python scripts/run_demo_retrieval.py --no-sparse --no-reranker
    python scripts/run_demo_retrieval.py --keep-indexed --tenant=MY-STUDY

Inspection des étapes intermédiaires (JSON sur disque) :
    python scripts/run_demo_retrieval.py --dump-dir outputs/demo_ct
    python scripts/run_demo_retrieval.py --dump-dir outputs/demo_ct --dump-full-text

Fichiers écrits : ``01_ingestion.json``, ``02_parsed.json``, ``03_chunks.json``,
``04_index_upsert.json``, ``05_health.json``, ``06_retrieval.json``, ``07_list_chunks.json``.

Pré-requis :
  - Qdrant up sur ``ECRF_QDRANT_URL`` (défaut http://localhost:6333,
    ``.env`` dev = http://localhost:6533).
  - Modèles préchargés via ``python scripts/prefetch_vector_models.py``
    (sinon premier run télécharge ~1.5GB).

Alternative Docker (recommandée Windows / retrieval) :
    docker compose -f docker-compose.dev.yml build app
    docker compose -f docker-compose.dev.yml up -d qdrant
    docker compose -f docker-compose.dev.yml run --rm app python scripts/run_demo_retrieval.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

# Console Windows par défaut = cp1252 → on force UTF-8 pour pouvoir afficher
# tout caractère retourné par Docling (titres médicaux français : é, à, etc.).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, io.UnsupportedOperation):
        pass

from app.config.settings import Settings  # noqa: E402
from app.indexing.qdrant_vector_service import QdrantHybridVectorService  # noqa: E402
from app.ingestion.service import LocalFileIngestionService  # noqa: E402
from app.parsing.chunking import DefaultChunkingService  # noqa: E402
from app.parsing.smart_service import SmartParsingService  # noqa: E402
from app.utils.safe_paths import redact_file_paths_in_jsonable  # noqa: E402

DEFAULT_PDF = Path("data/ct_scan_report_liver.pdf")

# Requêtes ciblées sur le contenu typique d'un CR d'imagerie hépatique.
DEFAULT_QUERIES = [
    "lésion hépatique",
    "conclusion du compte rendu",
    "antécédents du patient",
    "technique d'examen et produit de contraste",
]


def _truncate(text: str, n: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _fastembed_safe() -> bool:
    """fastembed/ONNX crashe (access violation) sur Python 3.14 + Windows."""
    return not (sys.version_info >= (3, 14) and platform.system() == "Windows")


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"  [dump] {path.resolve()}")


def _maybe_truncate_full_text_in_parsed_dump(
    d: dict[str, Any], *, full_text: bool, limit: int
) -> None:
    ft = d.get("full_text")
    if full_text or not isinstance(ft, str) or len(ft) <= limit:
        return
    d["full_text"] = (
        ft[:limit]
        + f"\n\n... [tronqué à {limit} car.; relancer avec --dump-full-text pour le texte complet] ...\n"
    )


def _summarize_chunk(c: Any) -> dict[str, Any]:
    md = dict(c.metadata or {})
    return {
        "chunk_id": c.chunk_id,
        "document_id": c.document_id,
        "text_len": len(c.text or ""),
        "text_preview": _truncate(c.text or "", 400),
        "char_start": c.char_start,
        "char_end": c.char_end,
        "field_family": getattr(c.field_family, "value", None) if c.field_family else None,
        "metadata": md,
    }


def _check_vector_deps() -> str | None:
    """Vérifie que l'extra ``[vector]`` est installé dans l'interpréteur courant."""
    try:
        import qdrant_client  # noqa: F401

        return None
    except ImportError:
        return (
            "Dépendances vector/Qdrant absentes pour cet interpréteur Python.\n"
            f"  Python : {sys.executable}\n"
            "  1. Activer le venv : .\\.venv\\Scripts\\activate\n"
            '  2. Installer : pip install -e ".[dev,vector]"'
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "document",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Chemin PDF (défaut {DEFAULT_PDF}).",
    )
    parser.add_argument("--patient-id", default="DEMO-PDF")
    parser.add_argument("--study-id", default="DEMO-CT")
    parser.add_argument("--tenant", default=None, help="Override tenant_id (def = --study-id).")
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Désactive BM25 sparse (auto sur Py3.14+Win, cf. fastembed/ONNX).",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Désactive le rerank CrossEncoder (gagne ~5s + plusieurs Go RAM).",
    )
    parser.add_argument(
        "--keep-indexed",
        action="store_true",
        help="Ne supprime pas le document indexé à la fin (debug retrieval).",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Requêtes custom. Si absent, utilise un panel par défaut.",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="Répertoire : écrit un JSON par étape (ingestion, parse, chunks, index, health, retrieval, list_chunks).",
    )
    parser.add_argument(
        "--dump-full-text",
        action="store_true",
        help="Avec --dump-dir : n'applique pas la troncature de full_text dans 02_parsed.json.",
    )
    args = parser.parse_args(argv)

    if err := _check_vector_deps():
        print(f"❌ {err}", file=sys.stderr)
        return 2

    pdf_path = args.document
    if not pdf_path.is_file():
        print(f"❌ Fichier introuvable : {pdf_path}", file=sys.stderr)
        return 2

    use_sparse = not args.no_sparse and _fastembed_safe()
    if args.no_sparse:
        sparse_reason = "désactivé via --no-sparse"
    elif not _fastembed_safe():
        sparse_reason = "auto-désactivé (Py3.14+Win incompatible fastembed/ONNX)"
    else:
        sparse_reason = "activé (Qdrant/bm25 via fastembed)"

    use_reranker = not args.no_reranker
    tenant = args.tenant or args.study_id
    queries = args.queries or DEFAULT_QUERIES

    settings = Settings(
        vector_backend="qdrant",
        pdf_parser_backend="auto",  # Docling si dispo, sinon pypdf
        enable_sparse_bm25=use_sparse,
        enable_reranker=use_reranker,
    )

    dump_dir: Path | None = args.dump_dir

    # ─── 1. Ingestion + Parsing ─────────────────────────────────────────────
    _section(f"[1/6] Ingestion + parsing : {pdf_path.name}")
    raw = LocalFileIngestionService().ingest(
        str(pdf_path.resolve()),
        patient_id=args.patient_id,
        study_id=args.study_id,
    )
    if dump_dir:
        ing = {
            "document_id": raw.document_id,
            "patient_id": raw.patient_id,
            "study_id": raw.study_id,
            "source_path": raw.source_path,
            "mime_type": raw.mime_type,
            "content_bytes_len": len(raw.content_bytes or b""),
        }
        _dump_json(dump_dir / "01_ingestion.json", redact_file_paths_in_jsonable(ing))

    print(f"  document_id          : {raw.document_id}")
    print(f"  mime_type            : {raw.mime_type}")
    print(f"  content_bytes        : {len(raw.content_bytes or b'')}")

    parsed = SmartParsingService(settings).parse(raw)
    print(f"  pdf_parser           : {parsed.metadata.get('pdf_parser')}")
    print(f"  document_type_hint   : {parsed.document_type_hint.value}")
    for key in (
        "document_type_scores",
        "document_type_features",
        "document_type_rationale",
        "lab_document_score",
        "imaging_or_morpho_context",
        "docling_layout_stats",
        "docling_sectioning",
    ):
        if key in parsed.metadata:
            val = parsed.metadata[key]
            if isinstance(val, (dict, list)):
                print(
                    f"  {key:22s} : {json.dumps(val, ensure_ascii=False)[:500]}{'…' if len(json.dumps(val)) > 500 else ''}"
                )
            else:
                print(f"  {key:22s} : {val!r}")
    print(f"  full_text chars      : {len(parsed.full_text)}")
    print(f"  structured_sections  : {len(parsed.structured_sections)}")
    print(f"  structured_lab_lines : {len(parsed.structured_lab_lines)}")
    if parsed.structured_sections:
        print("  Sections détectées :")
        for i, sec in enumerate(parsed.structured_sections):
            heading = sec.heading or "(no heading)"
            ck = (sec.metadata or {}).get("content_kind")
            tag = f" metadata.content_kind={ck!r}" if ck else ""
            print(f"    [{i}] {heading!r}{tag}  ({len(sec.body)} chars)")

    if dump_dir:
        parsed_dump = parsed.model_dump(mode="json")
        _maybe_truncate_full_text_in_parsed_dump(
            parsed_dump, full_text=args.dump_full_text, limit=12_000
        )
        _dump_json(
            dump_dir / "02_parsed.json",
            redact_file_paths_in_jsonable(parsed_dump),
        )

    # ─── 2. Chunking ────────────────────────────────────────────────────────
    _section("[2/6] Chunking")
    chunks = DefaultChunkingService(settings).chunk(parsed)
    print(f"  chunking_strategy : {parsed.metadata.get('chunking_strategy')}")
    print(f"  chunks            : {len(chunks)}")
    for i, c in enumerate(chunks):
        heading = (c.metadata or {}).get("section_heading") or "(no heading)"
        kind = (c.metadata or {}).get("content_kind", "?")
        span = ""
        if c.char_start is not None and c.char_end is not None:
            span = f" span=[{c.char_start},{c.char_end})"
        print(f"    [{i}] heading={heading!r} kind={kind} chars={len(c.text)}{span}")
        print(f"        -> {_truncate(c.text)}")

    if dump_dir:
        _dump_json(dump_dir / "03_chunks.json", [_summarize_chunk(c) for c in chunks])

    # ─── 3. Indexing Qdrant hybride ─────────────────────────────────────────
    _section("[3/6] Indexing Qdrant (hybride)")
    svc = QdrantHybridVectorService(settings)
    print(f"  qdrant_url        : {settings.qdrant_url}")
    print(f"  tenant            : {tenant}")
    print(f"  collection        : ecrf_chunks__{tenant}  (Option C : 1 par tenant)")
    print(f"  embedding_model   : {svc.embeddings.model_id}")
    print(f"  embedding_dim     : {svc.embeddings.dim}")
    print(f"  embedding_version : {svc.embeddings.version_tag}")
    print(f"  sparse BM25       : {sparse_reason}")
    print(
        f"  reranker          : {'activé (' + settings.reranker_model + ')' if use_reranker else 'désactivé'}"
    )
    print(f"  upsert_batch_size : {settings.upsert_batch_size}")
    svc.upsert_chunks(
        chunks,
        tenant_id=tenant,
        patient_id=args.patient_id,
        study_id=args.study_id,
        document_type=parsed.document_type_hint.value,
    )
    print(f"  OK {len(chunks)} chunks upsertés.")

    if dump_dir:
        _dump_json(
            dump_dir / "04_index_upsert.json",
            {
                "tenant_id": tenant,
                "document_id": raw.document_id,
                "patient_id": args.patient_id,
                "study_id": args.study_id,
                "document_type": parsed.document_type_hint.value,
                "chunks_upserted": len(chunks),
                "qdrant_url": settings.qdrant_url,
                "embedding_model": svc.embeddings.model_id,
                "embedding_dim": svc.embeddings.dim,
                "embedding_version": svc.embeddings.version_tag,
                "enable_sparse_bm25": use_sparse,
                "enable_reranker": use_reranker,
                "upsert_batch_size": settings.upsert_batch_size,
            },
        )

    # ─── 4. Health check ────────────────────────────────────────────────────
    _section("[4/6] Health check (F2)")
    h = svc.health_check()
    print(json.dumps(h, indent=2, ensure_ascii=False))
    if dump_dir:
        _dump_json(dump_dir / "05_health.json", h)

    # ─── 5. Retrieval ───────────────────────────────────────────────────────
    _section(f"[5/6] Retrieval — {len(queries)} requête(s) (top_k={args.top_k})")
    retrieval_dump: list[dict[str, Any]] = []
    for q in queries:
        print(f"\n  >> Query : {q!r}")
        hits = svc.search(query_text=q, tenant_id=tenant, top_k=args.top_k)
        q_entry: dict[str, Any] = {"query": q, "hits": []}
        if not hits:
            print("    (aucun résultat)")
            retrieval_dump.append(q_entry)
            continue
        for rank, (chunk, final_score) in enumerate(hits, start=1):
            md: dict[str, Any] = chunk.metadata or {}
            retr = md.get("retrieval_score")  # C1
            retr_rank = md.get("retrieval_rank")
            rer = md.get("rerank_score")  # C1 (présent ssi rerank actif)
            heading = md.get("section_heading") or "(no heading)"
            print(f"    [{rank}] final={final_score:+.4f}  heading={heading!r}")
            if rer is not None:
                print(f"        retrieval_score (Qdrant) = {retr:+.4f}  (rang Qdrant #{retr_rank})")
                print(f"        rerank_score (BGE)       = {rer:+.4f}")
            else:
                print(f"        retrieval_score (Qdrant) = {retr:+.4f}  (rang Qdrant #{retr_rank})")
            print(f"        -> {_truncate(chunk.text)}")
            q_entry["hits"].append(
                {
                    "rank": rank,
                    "final_score": final_score,
                    "chunk_id": chunk.chunk_id,
                    "section_heading": heading,
                    "retrieval_score": retr,
                    "retrieval_rank": retr_rank,
                    "rerank_score": rer,
                    "text_preview": _truncate(chunk.text, 500),
                    "metadata": md,
                }
            )
        retrieval_dump.append(q_entry)

    if dump_dir:
        _dump_json(dump_dir / "06_retrieval.json", retrieval_dump)

    # ─── 6. list_chunks (admin / debug) ─────────────────────────────────────
    _section("[6/6] list_chunks (D1) — inspection paginée")
    page1, next_offset = svc.list_chunks(tenant_id=tenant, document_id=raw.document_id, limit=2)
    print(f"  page 1 ({len(page1)} chunks, next_offset={next_offset!r}):")
    for c in page1:
        print(f"    - {c.chunk_id[:24]}…  {_truncate(c.text, 80)}")
    if next_offset is not None:
        page2, next_offset2 = svc.list_chunks(
            tenant_id=tenant, document_id=raw.document_id, limit=10, offset=next_offset
        )
        print(f"  page 2 ({len(page2)} chunks, next_offset={next_offset2!r}):")
        for c in page2:
            print(f"    - {c.chunk_id[:24]}…  {_truncate(c.text, 80)}")

    if dump_dir:
        list_dump = {
            "page1": [_summarize_chunk(c) for c in page1],
            "page1_next_offset": next_offset,
            "page2": [_summarize_chunk(c) for c in page2] if next_offset is not None else [],
            "page2_next_offset": next_offset2 if next_offset is not None else None,
        }
        _dump_json(dump_dir / "07_list_chunks.json", list_dump)
        _section("Artefacts JSON (--dump-dir)")
        print(f"  Répertoire : {dump_dir.resolve()}")
        for name in (
            "01_ingestion.json",
            "02_parsed.json",
            "03_chunks.json",
            "04_index_upsert.json",
            "05_health.json",
            "06_retrieval.json",
            "07_list_chunks.json",
        ):
            p = dump_dir / name
            if p.is_file():
                print(f"    - {name}  ({p.stat().st_size} octets)")

    # ─── Cleanup (sauf --keep-indexed) ──────────────────────────────────────
    if args.keep_indexed:
        print(f"\n💾 Index conservé. Tenant : {tenant}. Document : {raw.document_id}.")
        print("   À supprimer manuellement :")
        print(
            f'     python -c "from app.config.settings import Settings; '
            f"from app.indexing.qdrant_vector_service import QdrantHybridVectorService; "
            f'print(QdrantHybridVectorService(Settings()).delete_document(tenant_id={tenant!r}, document_id={raw.document_id!r}))"'
        )
    else:
        _section("Cleanup")
        n = svc.delete_document(tenant_id=tenant, document_id=raw.document_id)
        print(f"  delete_document → {n} point(s) supprimé(s) (A6 : count exact).")

    return 0


if __name__ == "__main__":
    # Sur Py3.14+Win, on quitte explicitement pour éviter le hang ONNX au teardown.
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    if platform.system() == "Windows" and sys.version_info >= (3, 14):
        os._exit(rc)
    sys.exit(rc)
