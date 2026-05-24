"""
Texte de passage pour **embedding dense**, **BM25 sparse** et **rerank**.

Le `DocumentChunk.text` reste le **corps seul** (aligné chunking / spans / affichage
ARC). Pour la recherche produit, le signal section (titre) est concaténé afin que
les modèles voient explicitement « CONCLUSION », « RÉSULTATS », etc.

Le payload Qdrant conserve `text` = corps seul ; les vecteurs denses et le sparse
BM25 sont calculés sur cette même chaîne `passage_for_embedding_and_rerank` à
l'upsert et au rerank, donc **alignés entre eux**. Pour un libellé proche du
passage indexé côté UI, joindre `metadata["section_heading"]` et `chunk.text`
(équivalent sémantique à cette fonction).
"""

from __future__ import annotations

from app.schemas.models import DocumentChunk


def passage_for_embedding_and_rerank(chunk: DocumentChunk) -> str:
    """
    Retourne `"{section_heading}\\n{body}"` lorsque le titre est disponible, sinon le corps.

    - `section_heading` provient de `chunk.metadata["section_heading"]`.
    - `body` = `chunk.text` (déjà le corps de section, sans titre dupliqué en amont).
    """
    md = chunk.metadata or {}
    raw_heading = md.get("section_heading")
    body = (chunk.text or "").strip()
    if raw_heading is not None and str(raw_heading).strip():
        heading = str(raw_heading).strip()
        return f"{heading}\n{body}" if body else heading
    return body
