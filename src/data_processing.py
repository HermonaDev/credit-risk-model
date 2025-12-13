import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    """Load raw transaction data."""
    df = pd.read_csv(filepath)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def create_customer_features(df, snapshot_date=None):
    """
    Aggregate transaction data to customer-level RFM features.
    
    Parameters:
    df: DataFrame with transaction data
    snapshot_date: Reference date for recency calculation. If None, uses max date in data.
    
    Returns:
    DataFrame with CustomerId and RFM features
    """
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max()
    
    # Monetary: total amount spent (using absolute value of Amount)
    monetary = df.groupby('CustomerId')['Amount'].agg(
        total_amount='sum',
        avg_amount='mean',
        std_amount='std',
        count='count'
    ).reset_index()
    
    # Frequency: number of transactions
    frequency = df.groupby('CustomerId').size().reset_index(name='frequency')
    
    # Recency: days since last transaction
    last_transaction = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    last_transaction['recency'] = (snapshot_date - last_transaction['TransactionStartTime']).dt.days
    
    # Merge all features
    customer_features = monetary.merge(frequency, on='CustomerId')
    customer_features = customer_features.merge(last_transaction[['CustomerId', 'recency']], on='CustomerId')
    
    # Fill NaN for std_amount (customers with single transaction)
    customer_features['std_amount'] = customer_features['std_amount'].fillna(0)
    
    return customer_features

def create_transaction_features(df):
    """
    Extract time-based features from transactions.
    
    Returns:
    DataFrame with original columns plus time features
    """
    df = df.copy()
    
    # Extract time components
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek  # Monday=0
    df['transaction_weekend'] = df['transaction_dayofweek'].isin([5, 6]).astype(int)
    
    return df

def get_feature_pipeline():
    """
    Create sklearn pipeline for feature transformations.
    Assumes categorical columns are known.
    """
    # Define column groups (update based on final feature set)
    numerical_features = ['recency', 'frequency', 'total_amount', 
                          'avg_amount', 'std_amount', 'count',
                          'transaction_hour', 'transaction_day']
    
    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId']
    
    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor

def prepare_training_data(raw_data_path):
    """Full data processing pipeline from raw data to features and target."""
    # Load and create features
    df = load_data(raw_data_path)
    
    # Create customer-level RFM features
    customer_features = create_customer_features(df)
    
    # Create transaction-level time features
    transaction_features = create_transaction_features(df)
    
    # Merge transaction features with customer features
    # We'll implement target creation in Task 4
    # For now, return features dataframe
    return customer_features

if __name__ == "__main__":
    # Test full pipeline
    features = prepare_training_data('data/raw/data.csv')
    print(f"Prepared features for {len(features)} customers")
    print(f"Feature columns: {list(features.columns)}")
    
    # Test pipeline creation
    pipeline = get_feature_pipeline()
    print(f"\nPipeline created with {len(pipeline.transformers)} transformers")