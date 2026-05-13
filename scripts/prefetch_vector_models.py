"""
Pré-télécharge et met en cache les modèles utilisés par le vector store.

Modèles cibles (résolus depuis `Settings`) :
- Embeddings denses (HF) : `ECRF_EMBEDDING_MODEL` (par défaut `intfloat/multilingual-e5-base`)
- Reranker CrossEncoder (HF) : `ECRF_RERANKER_MODEL` (par défaut `BAAI/bge-reranker-v2-m3`)
- Sparse BM25 (fastembed) : `ECRF_SPARSE_MODEL` (par défaut `Qdrant/bm25`)

Usage :
    python scripts/prefetch_vector_models.py
    python scripts/prefetch_vector_models.py --skip-reranker
    python scripts/prefetch_vector_models.py --device cpu

Variables utiles :
    HF_HOME / HF_HUB_CACHE       Emplacement du cache Hugging Face
    HF_TOKEN                     Si modèle « gated » (non requis pour ceux par défaut)
    ECRF_EMBEDDING_MODEL=...     Surcharge le modèle d’embeddings
    ECRF_RERANKER_MODEL=...      Surcharge le modèle de reranker
    ECRF_SPARSE_MODEL=...        Surcharge le modèle BM25

Code de retour :
    0 = OK
    1 = au moins une étape a échoué (les autres sont quand même tentées)
    2 = extras vector non installés (sentence-transformers, fastembed)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv optionnel
    pass

from app.config.settings import Settings


def _get_persisted_token() -> str | None:
    """Récupère le token persisté (cache `~/.cache/huggingface/token`) si présent.

    Compatible avec `huggingface_hub` 0.x (`HfFolder`) et 1.x (`get_token`)."""
    try:
        from huggingface_hub import get_token  # 1.x

        return get_token()
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder  # 0.x

        return HfFolder.get_token()  # type: ignore[attr-defined]
    except Exception:
        return None


def _persisted_token_path() -> Path:
    try:
        from huggingface_hub.constants import HF_TOKEN_PATH

        return Path(HF_TOKEN_PATH)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "token"


def _validate_hf_token(token: str) -> bool | None:
    """True = valide, False = explicitement invalide (401), None = indéterminé (réseau)."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError

        HfApi().whoami(token=token)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        # 401 / repo not found auth-related → token clairement invalide
        if "401" in msg or "expired" in msg or "unauthorized" in msg:
            return False
        # Erreur réseau / SSL → on ne sait pas trancher
        try:
            if isinstance(exc, HfHubHTTPError):
                return False
        except Exception:
            pass
        return None


def _move_aside_token(reason: str) -> None:
    path = _persisted_token_path()
    if not path.exists():
        return
    backup = path.with_suffix(path.suffix + ".expired")
    try:
        if backup.exists():
            backup.unlink()
        path.replace(backup)
        print(
            f"⚠️  Token persisté mis de côté ({reason}) : {path} → {backup.name}. "
            "Pour réactiver, renommez-le ou faites `hf auth login`.",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"⚠️  Impossible de mettre de côté le token persisté {path} ({exc}). "
            "Si le téléchargement échoue, supprimez-le manuellement.",
            file=sys.stderr,
        )


def _check_hf_token() -> None:
    """
    Si un HF_TOKEN (env ou token persisté) est invalide/expiré, on bascule en
    accès anonyme. Les modèles publics (e5, bge-reranker, Qdrant/bm25) ne
    nécessitent aucun token.
    """
    env_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if env_token:
        verdict = _validate_hf_token(env_token)
        if verdict is False:
            print(
                "⚠️  HF_TOKEN (env) invalide/expiré. Bascule en accès anonyme.",
                file=sys.stderr,
            )
            for k in (
                "HF_TOKEN",
                "HUGGINGFACEHUB_API_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
            ):
                os.environ.pop(k, None)

    persisted = _get_persisted_token()
    if not persisted:
        return
    verdict = _validate_hf_token(persisted)
    if verdict is True:
        return
    # Invalide (False) ou indéterminé (None) : on met de côté pour ne pas
    # polluer les requêtes implicites. L’utilisateur pourra le restaurer.
    reason = "expiré/invalide" if verdict is False else "non vérifiable (réseau)"
    _move_aside_token(reason)


def _print_step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _hf_cache_dir() -> Path:
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if cache:
        return Path(cache)
    return Path.home() / ".cache" / "huggingface"


def _fmt_size(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    val = float(num_bytes)
    for u in units:
        if val < 1024:
            return f"{val:.1f} {u}"
        val /= 1024
    return f"{val:.1f} PiB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _resolve_device(arg: str | None, settings: Settings) -> str:
    candidate = (arg or settings.reranker_device or "auto").lower()
    if candidate != "auto":
        return candidate
    try:
        from torch.cuda import is_available

        return "cuda" if is_available() else "cpu"
    except Exception:
        return "cpu"


def _ensure_extras() -> tuple[bool, bool]:
    has_hf = importlib.util.find_spec("huggingface_hub") is not None
    has_fe = importlib.util.find_spec("fastembed") is not None
    return has_hf, has_fe


_TRANSIENT_WIN_ERRORS = (10038, 10053, 10054, 10060, 10061)


def _is_transient_network_error(exc: BaseException) -> bool:
    msg = str(exc)
    if any(f"WinError {code}" in msg for code in _TRANSIENT_WIN_ERRORS):
        return True
    name = type(exc).__name__
    return name in {
        "ReadError",
        "WriteError",
        "ConnectError",
        "RemoteProtocolError",
        "ConnectTimeout",
        "ReadTimeout",
    }


def _is_symlink_privilege_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "WinError 1314" in msg
        or "privilège nécessaire" in msg
        or "privilege not held" in msg.lower()
    )


# Variantes que nous n’utilisons pas côté runtime (notre stack = PyTorch + SBERT).
# On les exclut systématiquement pour réduire la surface du snapshot HF et
# éviter des symlinks superflus sur Windows.
_DEFAULT_IGNORE = [
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.onnx_data",
    "*.ot",
    "flax_model.msgpack",
    "tf_model.h5",
    "rust_model.ot",
    "model.safetensors.index.json",  # gardons un seul fichier d’index si présent
]


def _snapshot_download(
    repo_id: str,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    max_workers: int = 4,
    retries: int = 3,
) -> str:
    """Télécharge tous (ou un sous-ensemble) des fichiers d'un repo HF dans le cache.

    Pas d'instanciation torch / onnx ici : on se contente de matérialiser les
    fichiers sur disque. Le service les chargera plus tard au runtime.

    Retry au niveau script :
    - Sur Windows + Python 3.14 + httpx, la fermeture du socket TLS après un
      gros fichier peut lever `ReadError [WinError 10038]` (WSAENOTSOCK) alors
      que le fichier est intégralement écrit. `snapshot_download` étant
      idempotent (resume + ETag), un simple replay convertit ça en no-op.
    - `max_workers=1` aide pour les repos à nombreux petits fichiers (HF Hub
      ferme les connexions parallèles → `WinError 10054`).
    - Sur Windows sans privilège « Créer un lien symbolique » (Mode Développeur
      désactivé), HF échoue à matérialiser un fichier (`WinError 1314`). On
      affiche dans ce cas un message clair."""
    from huggingface_hub import snapshot_download

    effective_ignore = list(_DEFAULT_IGNORE) + list(ignore_patterns or [])
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                ignore_patterns=effective_ignore,
                token=False,  # force l'anonyme (modèles publics)
                max_workers=max_workers,
            )
        except Exception as exc:
            last_exc = exc
            if _is_symlink_privilege_error(exc):
                print(
                    "  ⚠️  Windows refuse les symlinks (WinError 1314).\n"
                    "     Activez le « Mode Développeur » : Paramètres > "
                    "Confidentialité et sécurité > Pour les développeurs > "
                    "« Mode développeur » = ON, puis relancez le script.\n"
                    "     Le téléchargement a déjà été partiellement écrit ; un "
                    "deuxième run aurait été un no-op pour les fichiers existants.",
                    file=sys.stderr,
                )
                raise
            if attempt >= retries or not _is_transient_network_error(exc):
                raise
            print(
                f"  Réseau transitoire ({type(exc).__name__}): {exc}. "
                f"Nouvelle tentative {attempt + 1}/{retries}…",
                file=sys.stderr,
            )
            time.sleep(min(2**attempt, 8))
    # Inatteignable, mais pour la complétude du type-checker.
    raise last_exc  # type: ignore[misc]


def prefetch_embeddings(settings: Settings, device: str) -> bool:
    _print_step(f"Embeddings denses : {settings.embedding_model} (dim={settings.embedding_dim})")
    try:
        t0 = time.perf_counter()
        # On exclut les variantes Flax/TF/ONNX/SafeTensors-multiples pour aller vite
        # tout en gardant config + tokenizer + poids PyTorch + SBERT-config.
        path = _snapshot_download(
            settings.embedding_model,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.md",
                "tokenizer*",
                "*.safetensors",
                "pytorch_model.bin",
                "1_Pooling/*",
                "sentence_bert_config.json",
                "modules.json",
                "config_sentence_transformers.json",
            ],
        )
        elapsed = time.perf_counter() - t0
        print(f"OK ({elapsed:.1f}s) — snapshot : {path}")
        return True
    except Exception as exc:
        print(f"ÉCHEC : {type(exc).__name__}: {exc}")
        return False


def prefetch_reranker(settings: Settings, device: str) -> bool:
    _print_step(f"Reranker CrossEncoder : {settings.reranker_model}")
    try:
        t0 = time.perf_counter()
        path = _snapshot_download(
            settings.reranker_model,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.md",
                "tokenizer*",
                "*.safetensors",
                "pytorch_model.bin",
            ],
        )
        elapsed = time.perf_counter() - t0
        print(f"OK ({elapsed:.1f}s) — snapshot : {path}")
        return True
    except Exception as exc:
        print(f"ÉCHEC : {type(exc).__name__}: {exc}")
        return False


def prefetch_sparse_bm25(settings: Settings) -> bool:
    _print_step(f"BM25 sparse (fastembed) : {settings.sparse_model}")
    try:
        t0 = time.perf_counter()
        # snapshot HF (anonyme, série) — évite d'initialiser ONNX runtime ici
        # (qui peut crasher au teardown sur Python 3.14 + Windows).
        path = _snapshot_download(settings.sparse_model, max_workers=1)
        elapsed = time.perf_counter() - t0
        print(f"OK ({elapsed:.1f}s) — snapshot : {path}")
        return True
    except Exception as exc:
        print(f"ÉCHEC : {type(exc).__name__}: {exc}")
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Force le device (sinon : Settings.reranker_device ou auto).",
    )
    args = parser.parse_args(argv)

    _check_hf_token()
    settings = Settings()
    device = _resolve_device(args.device, settings)
    cache_dir = _hf_cache_dir()
    print(f"Cache Hugging Face : {cache_dir}")
    print(f"Device : {device}")

    has_hf, has_fe = _ensure_extras()
    if not has_hf:
        print(
            '❌ `huggingface_hub` absent. Installer l\'extra : pip install -e ".[vector]"',
            file=sys.stderr,
        )
        return 2
    if not has_fe and not args.skip_sparse:
        print(
            "ℹ️  `fastembed` absent : étape BM25 ignorée (utilisez --skip-sparse "
            'ou installer l\'extra : pip install -e ".[vector]").',
            file=sys.stderr,
        )
        args.skip_sparse = True

    before = _dir_size(cache_dir)
    ok = True

    if not args.skip_embeddings:
        ok = prefetch_embeddings(settings, device) and ok
    if not args.skip_reranker:
        ok = prefetch_reranker(settings, device) and ok
    if not args.skip_sparse:
        ok = prefetch_sparse_bm25(settings) and ok

    after = _dir_size(cache_dir)
    added = max(0, after - before)
    print(f"\nCache HF : {_fmt_size(before)} -> {_fmt_size(after)} (+{_fmt_size(added)})")

    if not ok:
        print(
            "\n⚠️ Une ou plusieurs étapes ont échoué. Vérifier la connexion, "
            "HF_TOKEN (si modèle gated) et le nom du modèle.",
            file=sys.stderr,
        )
        return 1

    print("\n✅ Tous les modèles requis sont en cache.")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Évite un access-violation au teardown (ONNX/fastembed) sur Windows + Python 3.14.
    os._exit(rc)
