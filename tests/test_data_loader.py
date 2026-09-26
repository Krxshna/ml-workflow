import pandas as pd

from core.data_loader import (
    _extract_child_category,
    _extract_primary_category,
    _parse_sales_rank,
    load_dataset,
)


def test_drops_duplicate_uniq_id(dataset_path):
    df = load_dataset(dataset_path)
    assert len(df) == 4
    assert df.index.is_unique


def test_weight_sentinel_becomes_nan(dataset_path):
    df = load_dataset(dataset_path)
    assert pd.isna(df.loc["p2", "weight"])


def test_normal_weight_is_preserved(dataset_path):
    df = load_dataset(dataset_path)
    assert df.loc["p1", "weight"] == 500


def test_missing_brand_and_colour_are_filled(dataset_path):
    df = load_dataset(dataset_path)
    assert df.loc["p3", "brand"] == "unknown"
    assert df.loc["p3", "colour"] == ""


def test_cat_blob_uses_subcategory_not_brand_or_colour(dataset_path):
    df = load_dataset(dataset_path)
    assert df.loc["p1", "cat_blob"] == "WomensSarees"
    assert df.loc["p4", "cat_blob"] == "WomensSarees"
    assert df.loc["p2", "cat_blob"] == "MensKurtas"


def test_child_category_extracts_second_key():
    cat_dict = {"ClothingAccessories": "#100", "WomensSarees": "#20"}
    assert _extract_child_category(cat_dict) == "WomensSarees"


def test_child_category_is_empty_when_only_parent_present():
    assert _extract_child_category({"ClothingAccessories": "#300"}) == ""
    assert _extract_child_category({}) == ""
    assert _extract_child_category(None) == ""


def test_extract_primary_category_joins_all_keys():
    cat_dict = {"ClothingAccessories": "#100", "WomensSarees": "#20"}
    assert _extract_primary_category(cat_dict) == "ClothingAccessories WomensSarees"


def test_parse_sales_rank_strips_hash_and_commas():
    assert _parse_sales_rank({"WomensSarees": "#1,793"}) == 1793


def test_parse_sales_rank_handles_missing_or_malformed_input():
    assert _parse_sales_rank({}) is None
    assert _parse_sales_rank(None) is None
    assert _parse_sales_rank("not-a-dict") is None
