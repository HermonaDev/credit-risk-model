import sys
sys.path.insert(0, "src")

import pandas as pd
import pytest
from data_processing import load_data, create_customer_features


def test_load_valid_data(tmp_path):
    df_mock = pd.DataFrame({
        "customer_id": [1, 2],
        "amount": [100, 200]
    })

    file_path = tmp_path / "data.csv"
    df_mock.to_csv(file_path, index=False)

    df = load_data(file_path)
    assert len(df) == 2


def test_create_customer_features():
    df = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "amount": [100, 50, 200]
    })

    features = create_customer_features(df)
    assert len(features) == 2


def test_empty_dataframe_raises_error():
    with pytest.raises(ValueError):
        create_customer_features(pd.DataFrame())
