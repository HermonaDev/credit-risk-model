"""Target variable engineering for credit risk model."""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_rfm_features(transaction_df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Create RFM (Recency, Frequency, Monetary) features from transaction data.
    
    Parameters
    ----------
    transaction_df : pd.DataFrame
        Must contain 'CustomerId', 'Amount', 'TransactionStartTime'.
    snapshot_date : pd.Timestamp, optional
        Reference date for recency calculation.
    
    Returns
    -------
    pd.DataFrame
        RFM features per customer.
    """
    # Validate input
    required = ['CustomerId', 'Amount', 'TransactionStartTime']
    missing = [col for col in required if col not in transaction_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if snapshot_date is None:
        snapshot_date = transaction_df['TransactionStartTime'].max()
    
    # Recency: days since last transaction
    last_transaction = (
        transaction_df.groupby('CustomerId')['TransactionStartTime']
        .max()
        .reset_index()
    )
    last_transaction['recency'] = (
        snapshot_date - last_transaction['TransactionStartTime']
    ).dt.days
    
    # Frequency: transaction count
    frequency = (
        transaction_df.groupby('CustomerId')
        .size()
        .reset_index(name='frequency')
    )
    
    # Monetary: total amount (absolute value for risk perspective)
    monetary = (
        transaction_df.groupby('CustomerId')['Amount']
        .sum()
        .reset_index()
        .rename(columns={'Amount': 'monetary_total'})
    )
    monetary['monetary_total'] = monetary['monetary_total'].abs()
    
    # Merge features
    rfm = pd.merge(last_transaction[['CustomerId', 'recency']], 
                   frequency, on='CustomerId')
    rfm = pd.merge(rfm, monetary, on='CustomerId')
    
    return rfm[['CustomerId', 'recency', 'frequency', 'monetary_total']]

def create_proxy_target(rfm_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Create proxy target variable using KMeans clustering on RFM features.
    
    Parameters
    ----------
    rfm_df : pd.DataFrame
        DataFrame with 'recency', 'frequency', 'monetary_total' columns.
    n_clusters : int
        Number of clusters for KMeans.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'CustomerId' and 'is_high_risk' (1 for highest risk cluster).
    """
    # Select RFM features
    features = ['recency', 'frequency', 'monetary_total']
    X = rfm_df[features].copy()
    
    # Create preprocessing and clustering pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10))
    ])
    
    # Fit and predict clusters
    rfm_df['cluster'] = pipeline.fit_predict(X)
    
    # Calculate cluster profiles to identify high-risk
    cluster_means = rfm_df.groupby('cluster')[features].mean()
    
    # Risk heuristic: high recency + low frequency + low monetary = higher risk
    cluster_means['risk_score'] = (
        cluster_means['recency'] / cluster_means['recency'].max() -
        cluster_means['frequency'] / cluster_means['frequency'].max() -
        cluster_means['monetary_total'] / cluster_means['monetary_total'].max()
    )
    
    high_risk_cluster = cluster_means['risk_score'].idxmax()
    
    # Create binary target
    rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
    
    print(f"Cluster {high_risk_cluster} identified as high-risk")
    print(f"High-risk customers: {rfm_df['is_high_risk'].sum()} ({rfm_df['is_high_risk'].mean()*100:.1f}%)")
    
    return rfm_df[['CustomerId', 'is_high_risk']]

def prepare_full_dataset(transaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: RFM features + proxy target.
    
    Returns
    -------
    pd.DataFrame
        Combined features and target for modeling.
    """
    rfm_features = create_rfm_features(transaction_df)
    target = create_proxy_target(rfm_features)
    
    # Merge features with target (drop duplicate column if exists)
    full_data = pd.merge(rfm_features, target, on='CustomerId', suffixes=('', '_drop'))
    
    # Remove any columns ending with '_drop'
    cols_to_drop = [col for col in full_data.columns if col.endswith('_drop')]
    if cols_to_drop:
        full_data = full_data.drop(columns=cols_to_drop)
    
    return full_data

if __name__ == "__main__":
    # Test the implementation
    from data_processing import load_data
    
    print("Testing target engineering...")
    df = load_data("data/raw/data.csv")
    full_data = prepare_full_dataset(df)
    
    print(f"\n✅ Prepared dataset with {len(full_data)} customers")
    print("Columns:", full_data.columns.tolist())
    print("\nTarget distribution:")
    print(full_data['is_high_risk'].value_counts())
    print("Percentage:")
    print(full_data['is_high_risk'].value_counts(normalize=True).round(3))