import json

import pytest

from core.config import Settings

RAW_RECORDS = [
    {
        "uniq_id": "p1",
        "product_name": "Cotton Saree",
        "brand": "Fabindia",
        "colour": "Red",
        "sales_price": "1200.00",
        "weight": "500",
        "rating": "4.2",
        "no__of_reviews": "10",
        "no__of_sellers": "2",
        "discount_percentage": "5",
        "amazon_prime__y_or_n": "Y",
        "best_seller_tag__y_or_n": "N",
        "sales_rank_in_parent_category": {"ClothingAccessories": "#100", "WomensSarees": "#20"},
        "parent___child_category__all": {"ClothingAccessories": "#100", "WomensSarees": "#20"},
        "meta_keywords": "saree cotton red",
    },
    {
        "uniq_id": "p1",
        "product_name": "Cotton Saree Duplicate",
        "brand": "Fabindia",
        "colour": "Red",
        "sales_price": "1200.00",
        "weight": "500",
        "rating": "4.2",
        "no__of_reviews": "10",
        "no__of_sellers": "2",
        "discount_percentage": "5",
        "amazon_prime__y_or_n": "Y",
        "best_seller_tag__y_or_n": "N",
        "sales_rank_in_parent_category": {"ClothingAccessories": "#100", "WomensSarees": "#20"},
        "parent___child_category__all": {"ClothingAccessories": "#100", "WomensSarees": "#20"},
        "meta_keywords": "saree cotton red",
    },
    {
        "uniq_id": "p2",
        "product_name": "Slim Fit Kurta",
        "brand": "unknown-brand",
        "colour": "Blue",
        "sales_price": "899.00",
        "weight": "999999999",
        "rating": "3.8",
        "no__of_reviews": "5",
        "no__of_sellers": "1",
        "discount_percentage": "0",
        "amazon_prime__y_or_n": "N",
        "best_seller_tag__y_or_n": "N",
        "sales_rank_in_parent_category": {"ClothingAccessories": "#200", "MensKurtas": "#50"},
        "parent___child_category__all": {"ClothingAccessories": "#200", "MensKurtas": "#50"},
        "meta_keywords": "kurta cotton",
    },
    {
        "uniq_id": "p3",
        "product_name": "Plain T-Shirt",
        "brand": None,
        "colour": None,
        "sales_price": "499.00",
        "weight": "300",
        "rating": None,
        "no__of_reviews": None,
        "no__of_sellers": None,
        "discount_percentage": None,
        "amazon_prime__y_or_n": "N",
        "best_seller_tag__y_or_n": "Y",
        "sales_rank_in_parent_category": {"ClothingAccessories": "#300"},
        "parent___child_category__all": {"ClothingAccessories": "#300"},
        "meta_keywords": "",
    },
    {
        "uniq_id": "p4",
        "product_name": "Printed Saree",
        "brand": "Fabindia",
        "colour": "Green",
        "sales_price": "1500.00",
        "weight": "600",
        "rating": "4.5",
        "no__of_reviews": "20",
        "no__of_sellers": "3",
        "discount_percentage": "10",
        "amazon_prime__y_or_n": "Y",
        "best_seller_tag__y_or_n": "Y",
        "sales_rank_in_parent_category": {"ClothingAccessories": "#90", "WomensSarees": "#15"},
        "parent___child_category__all": {"ClothingAccessories": "#90", "WomensSarees": "#15"},
        "meta_keywords": "saree printed green",
    },
]


@pytest.fixture
def dataset_path(tmp_path):
    path = tmp_path / "products.ldjson"
    with path.open("w") as f:
        for record in RAW_RECORDS:
            f.write(json.dumps(record) + "\n")
    return str(path)


@pytest.fixture
def settings(dataset_path):
    return Settings(
        data_path=dataset_path,
        text_max_features=200,
        text_min_df=1,
        text_svd_components=2,
        cat_max_features=50,
        cat_min_df=1,
    )
