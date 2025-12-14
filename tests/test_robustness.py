import sys
sys.path.insert(0, 'src')
import pandas as pd
from data_processing import load_data, create_customer_features

# Test 1: Valid file
print("Test 1: Loading valid data...")
df = load_data('data/raw/data.csv')
print(f"✓ Loaded {len(df)} rows")

# Test 2: Create features
print("\nTest 2: Creating customer features...")
features = create_customer_features(df)
print(f"✓ Created features for {len(features)} customers")

# Test 3: Empty dataframe (should fail)
print("\nTest 3: Testing empty DataFrame handling...")
try:
    create_customer_features(pd.DataFrame())
except ValueError as e:
    print(f"✓ Correctly raised error: {e}")

print("\n✅ All robustness tests completed!")