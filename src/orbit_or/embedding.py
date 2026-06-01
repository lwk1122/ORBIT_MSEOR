import logging
import threading
from typing import List, Optional
import asyncio
import hashlib
import math
import re

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

_embedding_model = None
_initialization_attempted = False
_embedding_model_lock = threading.Lock()


def _hash_embedding(text: str) -> List[float]:
    """Deterministic offline fallback embedding.

    This is not a semantic model, but it keeps retrieval and tests functional
    when the sentence-transformers model is unavailable.
    """
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    vector = [0.0] * EMBEDDING_DIM
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized)
    if not tokens:
        tokens = [normalized]
    for token in tokens:
        for gram in {token, token[:4], token[-4:]}:
            if not gram:
                continue
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % EMBEDDING_DIM
            vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def get_embedding_model():
    """
    Get or create the SentenceTransformer model handle (thread-safe singleton).
    """
    global _embedding_model, _initialization_attempted

    if _initialization_attempted:
        return _embedding_model

    with _embedding_model_lock:
        if _initialization_attempted:
            return _embedding_model

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Determine device: MPS > CUDA > CPU
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            logger.info(f"[Embedding] Loading {DEFAULT_EMBEDDING_MODEL} on {device}...")
            _embedding_model = SentenceTransformer(
                DEFAULT_EMBEDDING_MODEL, device=device
            )
            logger.info(f"[Embedding] Loaded successfully on {device}")
            _initialization_attempted = True
        except ImportError as e:
            _initialization_attempted = True  # permanent — dependency missing
            logger.warning(f"[Embedding] sentence-transformers not installed: {e}")
        except Exception as e:
            # Transient failure — allow retry on next call
            logger.warning(
                f"[Embedding] Failed to load embedding model (will retry): {e}"
            )

    return _embedding_model


def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding for text.
    """
    if not text or not text.strip():
        logger.debug("[Embedding] Empty text provided")
        return None

    model = get_embedding_model()
    if model is None:
        return _hash_embedding(text)

    try:
        embedding = model.encode(text)
        if embedding is not None:
            return embedding.tolist()
        return None
    except Exception as e:
        logger.debug(f"[Embedding] Error: {e}")
        return None


def get_embeddings_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """
    Get embeddings for multiple texts.
    """
    if not texts:
        return []

    model = get_embedding_model()
    if model is None:
        return [_hash_embedding(text) for text in texts]

    try:
        embeddings = model.encode(texts)
        if embeddings is not None:
            return embeddings.tolist()
        return None
    except Exception as e:
        logger.debug(f"[Embedding] Batch error: {e}")
        return None


async def aget_embedding(text: str) -> Optional[List[float]]:
    """Async wrapper for get_embedding to prevent event loop blocking."""
    return await asyncio.to_thread(get_embedding, text)


async def aget_embeddings_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Async wrapper for get_embeddings_batch to prevent event loop blocking."""
    return await asyncio.to_thread(get_embeddings_batch, texts)
