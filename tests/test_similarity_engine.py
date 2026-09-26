from unittest.mock import patch

from core.data_loader import load_dataset
from core.feature_engine import FeatureEngine
from core.similarity_engine import SimilarityEngine


def _build_engine(dataset_path, settings):
    df = load_dataset(dataset_path)
    features = FeatureEngine(settings).fit_transform(df)
    return SimilarityEngine(features, df.index.tolist(), settings)


def test_find_similar_excludes_the_query_product_itself(dataset_path, settings):
    engine = _build_engine(dataset_path, settings)
    result = engine.find_similar("p1", 2)

    assert result is not None
    assert "p1" not in result
    assert len(result) <= 2


def test_find_similar_unknown_product_returns_none(dataset_path, settings):
    engine = _build_engine(dataset_path, settings)
    assert engine.find_similar("does-not-exist", 3) is None


def test_repeated_query_hits_the_cache(dataset_path, settings):
    engine = _build_engine(dataset_path, settings)

    with patch.object(engine, "_query", wraps=engine._query) as spy:
        first = engine.find_similar("p1", 2)
        second = engine.find_similar("p1", 2)

        assert first == second
        spy.assert_called_once()


def test_different_num_similar_is_a_different_cache_key(dataset_path, settings):
    engine = _build_engine(dataset_path, settings)

    with patch.object(engine, "_query", wraps=engine._query) as spy:
        engine.find_similar("p1", 1)
        engine.find_similar("p1", 2)

        assert spy.call_count == 2
