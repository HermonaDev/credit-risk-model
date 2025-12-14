import sys
sys.path.insert(0, "src")

import pandas as pd
import pytest
from data_processing import load_data, create_customer_features


def test_load_valid_data(tmp_path):
    df_mock = pd.DataFrame({
        "CustomerId": [1, 2],
        "Amount": [100, 200],
        "TransactionStartTime": pd.to_datetime([
            "2024-01-01",
            "2024-01-02"
        ])
    })

    file_path = tmp_path / "data.csv"
    df_mock.to_csv(file_path, index=False)

    df = load_data(file_path)
    assert len(df) == 2


def test_create_customer_features():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "Amount": [100, 50, 200],
        "TransactionStartTime": pd.to_datetime([
            "2024-01-01",
            "2024-01-05",
            "2024-01-03"
        ])
    })

    features = create_customer_features(df)
    assert len(features) == 2


def test_empty_dataframe_raises_error():
    with pytest.raises(ValueError):
        create_customer_features(pd.DataFrame())
