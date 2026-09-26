import logging
import time
from typing import Optional

import faiss
import numpy as np
from cachetools import LRUCache
from cachetools.keys import hashkey

from core.config import Settings

logger = logging.getLogger(__name__)


class SimilarityEngine:
    def __init__(
        self,
        feature_matrix: np.ndarray,
        product_ids: list[str],
        settings: Settings,
    ) -> None:
        self._product_ids = product_ids
        self._id_to_idx: dict[str, int] = {pid: i for i, pid in enumerate(product_ids)}
        self._settings = settings
        self._cache: LRUCache = LRUCache(maxsize=settings.lru_cache_size)
        self._features = feature_matrix.astype(np.float32)
        self.vector_dim = feature_matrix.shape[1]
        self._build_index(self._features)

    def _build_index(self, features: np.ndarray) -> None:
        dim = features.shape[1]
        logger.info(
            "Building FAISS HNSW index: %d vectors, dim=%d, M=%d, efConstruction=%d, efSearch=%d",
            features.shape[0], dim,
            self._settings.faiss_hnsw_m,
            self._settings.faiss_hnsw_ef_construction,
            self._settings.faiss_hnsw_ef_search,
        )
        start = time.perf_counter()

        index = faiss.IndexHNSWFlat(dim, self._settings.faiss_hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self._settings.faiss_hnsw_ef_construction
        index.hnsw.efSearch = self._settings.faiss_hnsw_ef_search
        index.add(features)
        self._index = index

        logger.info("FAISS index built in %.2fs", time.perf_counter() - start)

    def find_similar(self, product_id: str, num_similar: int) -> Optional[list[str]]:
        if product_id not in self._id_to_idx:
            return None

        cache_key = hashkey(product_id, num_similar)
        if cache_key in self._cache:
            logger.debug("Cache hit: product_id=%s num_similar=%d", product_id, num_similar)
            return self._cache[cache_key]

        result = self._query(product_id, num_similar)
        self._cache[cache_key] = result
        logger.debug(
            "Cache miss: product_id=%s num_similar=%d -> %d results",
            product_id, num_similar, len(result),
        )
        return result

    def _query(self, product_id: str, num_similar: int) -> list[str]:
        idx = self._id_to_idx[product_id]
        query_vec = self._features[idx : idx + 1]

        scores, indices = self._index.search(query_vec, num_similar + 1)

        results: list[str] = []
        for i in indices[0]:
            if i < 0 or i == idx:
                continue
            results.append(self._product_ids[i])
            if len(results) == num_similar:
                break

        return results

    @property
    def product_count(self) -> int:
        return len(self._product_ids)

    def product_exists(self, product_id: str) -> bool:
        return product_id in self._id_to_idx
