import pickle
import os

# Path to light_sakt metrics
metrics_path = os.path.join("ckpts", "light_sakt", "Algebra2005", "metrics.pkl")

print(f"Checking: {metrics_path}")
print(f"File exists: {os.path.exists(metrics_path)}")

if os.path.exists(metrics_path):
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)
    
    print(f"\nType of metrics: {type(metrics)}")
    print(f"\nMetrics content:")
    print(metrics)
    
    if isinstance(metrics, dict):
        print(f"\nKeys in metrics: {metrics.keys()}")
        for key, value in metrics.items():
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")
            if isinstance(value, list):
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  First few values: {value[:5]}")
                    print(f"  Last few values: {value[-5:]}")
