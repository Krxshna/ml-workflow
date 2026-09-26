import logging
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, normalize

from core.config import Settings

logger = logging.getLogger(__name__)


_NUMERIC_COLS = [
    "sales_price",
    "rating",
    "weight",
    "no__of_reviews",
    "no__of_sellers",
    "discount_percentage",
    "parent_rank",
    "is_prime",
    "is_bestseller",
]


class FeatureEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        self._cat_tfidf = TfidfVectorizer(
            max_features=settings.cat_max_features,
            min_df=settings.cat_min_df,
            analyzer="word",
            sublinear_tf=True,
        )
        self._text_tfidf = TfidfVectorizer(
            max_features=settings.text_max_features,
            min_df=settings.text_min_df,
            ngram_range=(1, 2),
            analyzer="word",
            sublinear_tf=True,
            stop_words="english",
        )
        self._svd = TruncatedSVD(n_components=settings.text_svd_components, random_state=42)
        self.vector_dim: int = 0

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        logger.info("Fitting feature pipeline on %d products", len(df))
        start = time.perf_counter()

        numeric_cols = [c for c in _NUMERIC_COLS if c in df.columns and df[c].notna().any()]
        dropped_numeric = [c for c in _NUMERIC_COLS if c not in numeric_cols]
        if dropped_numeric:
            logger.warning(
                "Numeric columns unavailable (missing or no observed values), excluded from features: %s",
                dropped_numeric,
            )
        self._numeric_cols = numeric_cols
        raw_numeric = df[numeric_cols].values.astype(np.float64)
        numeric_features = self._scaler.fit_transform(
            self._imputer.fit_transform(raw_numeric)
        )

        cat_features = self._cat_tfidf.fit_transform(
            df["cat_blob"].fillna("").astype(str)
        ).toarray()

        text_sparse = self._text_tfidf.fit_transform(
            df["text_blob"].fillna("").astype(str)
        )
        text_features = self._svd.fit_transform(text_sparse)

        w_n = self._settings.weight_numeric
        w_c = self._settings.weight_categorical
        w_t = self._settings.weight_text

        combined = np.hstack([
            numeric_features * w_n,
            cat_features * w_c,
            text_features * w_t,
        ]).astype(np.float32)

        normalized = normalize(combined, norm="l2")
        self.vector_dim = normalized.shape[1]

        logger.info(
            "Feature pipeline fit in %.2fs: numeric=%d cat=%d text=%d -> combined=%d dims",
            time.perf_counter() - start,
            numeric_features.shape[1], cat_features.shape[1], text_features.shape[1],
            normalized.shape[1],
        )
        return normalized

    def transform(self, series_row: pd.Series) -> np.ndarray:
        logger.debug("Transforming single product row to feature vector")
        df_single = series_row.to_frame().T
        raw_numeric = df_single[self._numeric_cols].values.astype(np.float64)
        numeric_features = self._scaler.transform(self._imputer.transform(raw_numeric))

        cat_features = self._cat_tfidf.transform(
            df_single["cat_blob"].fillna("").astype(str)
        ).toarray()

        text_sparse = self._text_tfidf.transform(
            df_single["text_blob"].fillna("").astype(str)
        )
        text_features = self._svd.transform(text_sparse)

        w_n = self._settings.weight_numeric
        w_c = self._settings.weight_categorical
        w_t = self._settings.weight_text

        combined = np.hstack([
            numeric_features * w_n,
            cat_features * w_c,
            text_features * w_t,
        ]).astype(np.float32)

        return normalize(combined, norm="l2")
