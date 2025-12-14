"""Data processing module for credit risk model."""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw transaction data with validation.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded and validated transaction data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing or date parsing fails.
    """
    import os

    if not os.path.exists(filepath):
        msg = f"Data file not found at: {filepath}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(filepath)

    # Validate required columns
    required_columns = ["CustomerId", "Amount", "TransactionStartTime"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert date column
    try:
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    except Exception as e:
        raise ValueError(f"Failed to parse 'TransactionStartTime': {e}")

    return df


def create_customer_features(
    df: pd.DataFrame, snapshot_date: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Aggregate transaction data to customer-level RFM features.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with 'CustomerId', 'Amount', 'TransactionStartTime'.
    snapshot_date : pd.Timestamp, optional
        Reference date for recency calculation. Defaults to max date in data.

    Returns
    -------
    pd.DataFrame
        DataFrame with CustomerId and RFM features.

    Raises
    ------
    ValueError
        If input DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    required = ["CustomerId", "Amount", "TransactionStartTime"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")

    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max()

    # Monetary features
    monetary = (
        df.groupby("CustomerId")["Amount"]
        .agg(
            total_amount="sum",
            avg_amount="mean",
            std_amount="std",
            transaction_count="count",
        )
        .reset_index()
    )

    # Frequency
    frequency = df.groupby("CustomerId").size().reset_index(name="frequency")

    # Recency
    last_transaction = (
        df.groupby("CustomerId")["TransactionStartTime"].max().reset_index()
    )
    time_diff = snapshot_date - last_transaction["TransactionStartTime"]
    last_transaction["recency"] = time_diff.dt.days

    # Merge features
    customer_features = monetary.merge(frequency, on="CustomerId")
    recency_df = last_transaction[["CustomerId", "recency"]]
    customer_features = customer_features.merge(recency_df, on="CustomerId")

    # Handle single-transaction customers (std is NaN)
    customer_features["std_amount"] = customer_features["std_amount"].fillna(0)

    return customer_features


def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with 'TransactionStartTime' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus time features.

    Raises
    ------
    ValueError
        If 'TransactionStartTime' column is missing or not datetime.
    """
    if "TransactionStartTime" not in df.columns:
        raise ValueError("DataFrame missing 'TransactionStartTime' column")

    if not pd.api.types.is_datetime64_any_dtype(df["TransactionStartTime"]):
        raise ValueError("'TransactionStartTime' must be datetime type")

    df = df.copy()

    # Extract time components
    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year
    df["transaction_dayofweek"] = df["TransactionStartTime"].dt.dayofweek
    is_weekend = df["transaction_dayofweek"].isin([5, 6])
    df["transaction_weekend"] = is_weekend.astype(int)
    df["transaction_weekend"] = df["transaction_weekend"].astype(int)

    return df

def get_feature_pipeline() -> ColumnTransformer:
    """
    Create sklearn pipeline for feature transformations.

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline with numerical and categorical transformers.

    Notes
    -----
    Assumes specific numerical and categorical feature lists.
    Update these lists based on final feature engineering.
    """
    # Define column groups
    numerical_features = [
        "recency",
        "frequency",
        "total_amount",
        "avg_amount",
        "std_amount",
        "transaction_count",
        "transaction_hour",
        "transaction_day",
    ]

    categorical_features = ["ProductCategory", "ChannelId", "ProviderId"]

    # Numerical pipeline
    numerical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Column transformer
    preprocessor = ColumnTransformer(
        [
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def prepare_training_data(raw_data_path: str) -> pd.DataFrame:
    """
    Full data processing pipeline from raw data to features.

    Parameters
    ----------
    raw_data_path : str
        Path to raw transaction data CSV.

    Returns
    -------
    pd.DataFrame
        Customer-level features ready for model training.

    Raises
    ------
    FileNotFoundError, ValueError
        Propagated from load_data and feature creation functions.
    """
    # Load and validate data
    df = load_data(raw_data_path)

    # Create customer-level RFM features
    customer_features = create_customer_features(df)

    # Create transaction-level time features (for completeness)
    # Note: These are not merged with customer features in this version
    _ = create_transaction_features(df)

    return customer_features


if __name__ == "__main__":
    # Test full pipeline
    try:
        features = prepare_training_data("data/raw/data.csv")
        print(f"✅ Prepared features for {len(features)} customers")
        print(f"Feature columns: {list(features.columns)}")

        # Test pipeline creation
        pipeline = get_feature_pipeline()
        print(f"✅ Pipeline created with {len(pipeline.transformers)} transformers")

    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
"" 
