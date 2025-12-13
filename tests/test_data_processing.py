import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_data, create_customer_features

def test_load_data():
    """Test that data loads correctly."""
    # Create minimal test data
    test_data = pd.DataFrame({
        'TransactionId': ['T1', 'T2'],
        'CustomerId': ['C1', 'C2'],
        'Amount': [100.0, -50.0],
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 11:00:00'],
        'ProductCategory': ['A', 'B'],
        'ChannelId': ['Web', 'App'],
        'ProviderId': ['P1', 'P2']
    })
    
    # Save to temporary file
    test_path = 'test_data.csv'
    test_data.to_csv(test_path, index=False)
    
    try:
        df = load_data(test_path)
        assert len(df) == 2
        assert 'TransactionStartTime' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime'])
        print("✅ test_load_data passed")
        return True
    except Exception as e:
        print(f"❌ test_load_data failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)

def test_create_customer_features():
    """Test RFM feature creation."""
    test_data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C2'],
        'Amount': [100, 200, -50, -30, -20],
        'TransactionStartTime': pd.date_range('2023-01-01', periods=5, freq='D')
    })
    
    features = create_customer_features(test_data)
    
    assert len(features) == 2  # Two unique customers
    assert 'recency' in features.columns
    assert 'frequency' in features.columns
    assert 'total_amount' in features.columns
    
    # Check C1 values
    c1 = features[features['CustomerId'] == 'C1'].iloc[0]
    assert c1['frequency'] == 2
    assert c1['total_amount'] == 300
    assert c1['recency'] == 3  # Days from last transaction (2023-01-02) to snapshot (2023-01-05)
    
    print("✅ test_create_customer_features passed")
    return True

if __name__ == "__main__":
    print("Running data processing tests...")
    test1 = test_load_data()
    test2 = test_create_customer_features()
    
    if test1 and test2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)