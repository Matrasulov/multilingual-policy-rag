from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("policy_embedder")


class MultilingualEmbedder:
    """
    Multilingual embedder for EN/KR documents (E5 style).
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Model loaded. Embedding dim: {self.embedding_dim}")

    def embed_documents(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        embs = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        logger.info(f"Embedded {len(texts)} documents -> {embs.shape}")
        return embs

    def embed_query(self, query: str) -> np.ndarray:
        prefixed = f"query: {query}"
        emb = self.model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        return emb

    def compute_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        return document_embeddings @ query_embedding


class EmbeddingCache:
    def __init__(self, cache_dir: str = "./data/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, embeddings: np.ndarray, cache_key: str) -> Path:
        path = self.cache_dir / f"{cache_key}.npy"
        np.save(path, embeddings)
        logger.info(f"Cached embeddings: {path}")
        return path

    def load(self, cache_key: str) -> Optional[np.ndarray]:
        path = self.cache_dir / f"{cache_key}.npy"
        if path.exists():
            embs = np.load(path)
            logger.info(f"Loaded cached embeddings: {path}")
            return embs
        return None
