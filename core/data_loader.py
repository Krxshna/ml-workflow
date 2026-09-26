import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_WEIGHT_SENTINEL = 999_999_999


def _parse_sales_rank(rank_dict: object) -> Optional[int]:
    if not isinstance(rank_dict, dict) or not rank_dict:
        return None
    raw = next(iter(rank_dict.values()), None)
    if not isinstance(raw, str):
        return None
    cleaned = raw.replace("#", "").replace(",", "").strip()
    try:
        return int(cleaned)
    except ValueError:
        return None


def _extract_primary_category(cat_dict: object) -> str:
    if not isinstance(cat_dict, dict) or not cat_dict:
        return ""
    return " ".join(cat_dict.keys())


def _extract_child_category(cat_dict: object) -> str:
    if not isinstance(cat_dict, dict) or len(cat_dict) < 2:
        return ""
    return list(cat_dict.keys())[1]


def load_dataset(path: str) -> pd.DataFrame:
    logger.info("Loading dataset from %s", path)
    start = time.perf_counter()

    df = pd.read_json(path, lines=True, dtype=False)

    rows_before = len(df)
    df = df.drop_duplicates(subset=["uniq_id"])
    if len(df) != rows_before:
        logger.info("Dropped %d duplicate uniq_id rows", rows_before - len(df))
    df = df.set_index("uniq_id")

    df["sales_price"] = pd.to_numeric(df["sales_price"], errors="coerce")

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    weight_sentinel_mask = df["weight"] >= _WEIGHT_SENTINEL
    if weight_sentinel_mask.any():
        logger.info(
            "Weight sentinel cleanup: %d values >= %d replaced with NaN",
            weight_sentinel_mask.sum(), _WEIGHT_SENTINEL,
        )
    df.loc[weight_sentinel_mask, "weight"] = np.nan

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df["no__of_reviews"] = pd.to_numeric(df.get("no__of_reviews"), errors="coerce").fillna(0)
    df["no__of_sellers"] = pd.to_numeric(df.get("no__of_sellers"), errors="coerce").fillna(0)
    df["discount_percentage"] = pd.to_numeric(df.get("discount_percentage"), errors="coerce").fillna(0)

    df["is_prime"] = (df.get("amazon_prime__y_or_n", pd.Series("N", index=df.index)) == "Y").astype(np.int8)
    df["is_bestseller"] = (df.get("best_seller_tag__y_or_n", pd.Series("N", index=df.index)) == "Y").astype(np.int8)

    df["parent_rank"] = df["sales_rank_in_parent_category"].apply(_parse_sales_rank)
    df["category"] = df["parent___child_category__all"].apply(_extract_primary_category)
    df["child_category"] = df["parent___child_category__all"].apply(_extract_child_category)

    df["brand"] = df["brand"].fillna("unknown").astype(str).str.strip().str.lower()
    df["colour"] = df["colour"].fillna("").astype(str).str.strip().str.lower()

    df["product_name"] = df["product_name"].fillna("").astype(str)
    df["meta_keywords"] = df["meta_keywords"].fillna("").astype(str)

    df["text_blob"] = (
        df["product_name"] + " "
        + df["meta_keywords"] + " "
        + df["brand"] + " "
        + df["colour"]
    ).str.strip()

    df["cat_blob"] = df["child_category"].fillna("").astype(str).str.strip()

    drop_cols = [
        "crawl_timestamp", "asin", "product_url", "image_urls__small",
        "medium", "large", "browsenode", "delivery_type",
        "amazon_prime__y_or_n", "best_seller_tag__y_or_n",
        "sales_rank_in_parent_category", "sales_rank_in_child_category",
        "parent___child_category__all", "other_items_customers_buy",
        "product_details__k_v_pairs", "technical_details__k_v_pairs",
        "formats___editions", "name_of_author_for_books",
        "seller_id", "seller_name", "left_in_stock", "no__of_offers",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    logger.info("Loaded and cleaned %d products in %.2fs", len(df), time.perf_counter() - start)
    return df
