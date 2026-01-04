import joblib
import numpy as np

# Load the top_features.joblib file
top_features = joblib.load("models/top_features.joblib")

print("=== Top Features ===")
print(f"\nType: {type(top_features)}")
print(f"\nContent:")
print(top_features)

# If it's a numpy array or list, show more details
if isinstance(top_features, (list, np.ndarray)):
    print(f"\nLength: {len(top_features)}")
    print(f"\nFirst few items: {top_features[:10] if len(top_features) > 10 else top_features}")
elif isinstance(top_features, dict):
    print(f"\nKeys: {list(top_features.keys())}")
    for key, value in top_features.items():
        print(f"\n{key}: {value}")
