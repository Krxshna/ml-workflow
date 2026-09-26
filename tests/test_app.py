import importlib

import pytest
from fastapi.testclient import TestClient

from core.config import get_settings


@pytest.fixture
def client(dataset_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", dataset_path)
    monkeypatch.setenv("TEXT_MAX_FEATURES", "200")
    monkeypatch.setenv("TEXT_MIN_DF", "1")
    monkeypatch.setenv("TEXT_SVD_COMPONENTS", "2")
    monkeypatch.setenv("CAT_MAX_FEATURES", "50")
    monkeypatch.setenv("CAT_MIN_DF", "1")
    get_settings.cache_clear()

    import app as app_module

    importlib.reload(app_module)

    with TestClient(app_module.app) as test_client:
        yield test_client

    get_settings.cache_clear()


def test_health_reports_ready_with_correct_product_count(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["products_loaded"] == 4


def test_healthz_liveness_does_not_depend_on_the_index(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_find_similar_products_unknown_id_is_404(client):
    response = client.get("/find_similar_products", params={"product_id": "nope", "num_similar": 2})
    assert response.status_code == 404


def test_find_similar_products_success(client):
    response = client.get("/find_similar_products", params={"product_id": "p1", "num_similar": 2})
    assert response.status_code == 200
    ids = response.json()
    assert "p1" not in ids
    assert len(ids) <= 2


def test_num_similar_out_of_range_is_422(client):
    response = client.get("/find_similar_products", params={"product_id": "p1", "num_similar": 0})
    assert response.status_code == 422


def test_similar_with_details_success(client):
    response = client.get("/similar_with_details", params={"product_id": "p1", "num_similar": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["product_id"] == "p1"
    assert body["count"] == len(body["similar_product_ids"])


def test_get_product_detail(client):
    response = client.get("/products/p1")
    assert response.status_code == 200
    assert response.json()["product_name"] == "Cotton Saree"


def test_get_product_not_found_is_404(client):
    response = client.get("/products/nope")
    assert response.status_code == 404
