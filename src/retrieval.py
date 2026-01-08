from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import logging
import re

import numpy as np
from sentence_transformers import CrossEncoder

from .utils import keep_text

logger = logging.getLogger("policy_retrieval")


@dataclass
class SearchResult:
    rank: int
    score: float
    chunk_id: str
    section_header: str
    text_preview: str
    index: int


def basic_search(query: str, embedder, doc_embeddings: np.ndarray, chunks: Sequence, top_k: int = 5) -> List[SearchResult]:
    q_emb = embedder.embed_query(query)
    scores = embedder.compute_similarity(q_emb, doc_embeddings)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results: List[SearchResult] = []
    for rank, i in enumerate(top_idx, start=1):
        header = chunks[i].metadata.get("section_header", "")
        results.append(
            SearchResult(
                rank=rank,
                score=float(scores[i]),
                chunk_id=chunks[i].chunk_id,
                section_header=header,
                text_preview=chunks[i].text[:300].replace("\n", " "),
                index=int(i),
            )
        )
    return results


def filtered_keyword_boost_search(
    query: str,
    embedder,
    doc_embeddings: np.ndarray,
    chunks: Sequence,
    top_k: int = 5,
    k_retrieve: int = 30,
) -> List[SearchResult]:
    scores = embedder.compute_similarity(embedder.embed_query(query), doc_embeddings)
    cand = np.argsort(scores)[::-1][:k_retrieve]
    ql = query.lower()

    def keyword_bonus(text: str) -> float:
        t = text.lower()
        bonus = 0.0
        if "overtime" in ql and "overtime" in t: bonus += 0.03
        if ("annual leave" in ql or "vacation" in ql) and ("vacation" in t or "annual leave" in t): bonus += 0.03
        if "sick" in ql and "sick" in t: bonus += 0.03
        if "holiday" in ql and "holiday" in t: bonus += 0.03
        return bonus

    scored: List[Tuple[float, int]] = []
    for i in cand:
        txt = chunks[i].text
        if not keep_text(txt):
            continue
        scored.append((float(scores[i]) + keyword_bonus(txt), int(i)))

    if not scored:
        scored = [(float(scores[i]), int(i)) for i in cand]

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    out: List[SearchResult] = []
    for r, (sc, i) in enumerate(top, start=1):
        out.append(
            SearchResult(
                rank=r,
                score=float(sc),
                chunk_id=chunks[i].chunk_id,
                section_header=chunks[i].metadata.get("section_header", ""),
                text_preview=chunks[i].text[:300].replace("\n", " "),
                index=i,
            )
        )
    return out


def rerank_search(
    query: str,
    embedder,
    doc_embeddings: np.ndarray,
    chunks: Sequence,
    reranker: CrossEncoder,
    top_k: int = 5,
    k_retrieve: int = 30,
    filter_text: bool = True,
) -> List[SearchResult]:
    scores = embedder.compute_similarity(embedder.embed_query(query), doc_embeddings)
    cand = np.argsort(scores)[::-1][:k_retrieve]

    cand = [int(i) for i in cand if (keep_text(chunks[int(i)].text) if filter_text else True)]
    if not cand:
        cand = [int(i) for i in np.argsort(scores)[::-1][:k_retrieve]]

    pairs = [(query, chunks[i].text) for i in cand]
    rr_scores = reranker.predict(pairs)

    order = np.argsort(rr_scores)[::-1][:top_k]
    top = [cand[int(j)] for j in order]

    out: List[SearchResult] = []
    for r, i in enumerate(top, start=1):
        out.append(
            SearchResult(
                rank=r,
                score=float(rr_scores[cand.index(i)]),
                chunk_id=chunks[i].chunk_id,
                section_header=chunks[i].metadata.get("section_header", ""),
                text_preview=chunks[i].text[:300].replace("\n", " "),
                index=i,
            )
        )
    return out


def load_default_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    logger.info(f"Loading reranker: {model_name}")
    return CrossEncoder(model_name)
