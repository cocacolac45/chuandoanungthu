from sklearn.datasets import load_breast_cancer
import json

data = load_breast_cancer()

sample = data.data[0]

payload = {
    "features": {
        name: float(value)
        for name, value in zip(data.feature_names, sample)
    }
}

with open("sample_request.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print("Created sample_request.json")
