from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_path: str = "data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"

    weight_numeric: float = 0.25
    weight_categorical: float = 0.40
    weight_text: float = 0.35

    faiss_hnsw_m: int = 32
    faiss_hnsw_ef_construction: int = 200
    faiss_hnsw_ef_search: int = 50

    lru_cache_size: int = 1024
    max_similar: int = 100

    text_svd_components: int = 100
    text_max_features: int = 8000
    text_min_df: int = 2

    cat_max_features: int = 500
    cat_min_df: int = 1

    log_level: str = "INFO"

    app_title: str = "Product Similarity Service"
    app_version: str = "1.0.0"
    app_description: str = "FAISS HNSW-powered product similarity search over Amazon Fashion dataset"


@lru_cache
def get_settings() -> Settings:
    return Settings()
