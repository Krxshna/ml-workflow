import numpy as np

from core.data_loader import load_dataset
from core.feature_engine import FeatureEngine


def test_fit_transform_output_is_unit_normalized_float32(dataset_path, settings):
    df = load_dataset(dataset_path)
    features = FeatureEngine(settings).fit_transform(df)

    assert features.shape[0] == len(df)
    assert features.dtype == np.float32
    norms = np.linalg.norm(features, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_all_nan_numeric_column_is_excluded_not_crashed(dataset_path, settings):
    df = load_dataset(dataset_path)
    df["rating"] = np.nan

    engine = FeatureEngine(settings)
    engine.fit_transform(df)

    assert "rating" not in engine._numeric_cols


def test_transform_matches_fit_transform_feature_columns(dataset_path, settings):
    df = load_dataset(dataset_path)
    engine = FeatureEngine(settings)
    engine.fit_transform(df)

    single_row_vector = engine.transform(df.loc["p1"])
    assert single_row_vector.shape == (1, engine.vector_dim)
