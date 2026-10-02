from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, recall_score, precision_score, roc_auc_score)
from pathlib import Path

import json
import joblib
import sklearn

RS = 110

# load du lieu
data = load_breast_cancer()
X = data.data
y = data.target

# chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = RS)

# tao pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel = "rbf",
        probability = True,
        class_weight = "balanced",
        random_state = RS
    ))
])

# khong gian sieu tham so
param_grid = {
    "svc__C": [0.01, 0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1, 1]
}

# cross-validation
cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state =RS
)

# gridsearchcv
search = GridSearchCV(
    estimator = pipeline,
    param_grid = param_grid,
    scoring = "recall_macro",
    cv = cv,
    n_jobs = -1,
    refit = True
)

# huan luyen mo hinh
search.fit(X_train, y_train)
model = search.best_estimator_

print("Best params:", search.best_params_)
print("Best CV score:", search.best_score_)

# du doan
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
classes = model.named_steps["svc"].classes_

print("Classes:", classes)

# lay xac xuat malignant
malignant_index = list(classes).index(0)
proba_malignant = proba[:, malignant_index]

# confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred, labels=[0, 1]))

# classification report
print("\nClassification Report:")
print(classification_report(y_test, pred, labels=[0, 1], target_names=["malignant", "benign"]))

# recall cua malignant
sensitivity = recall_score(y_test, pred, pos_label=0)
print("Sensitivity malignant:", sensitivity)

# precision cua malignant
precision_malignant = precision_score(y_test, pred, pos_label=0)
print("Precision malignant:", precision_malignant)

# ROC-AUC
roc_auc = roc_auc_score((y_test == 0).astype(int), proba_malignant)
print("ROC-AUC malignant:", roc_auc)

# luu model
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

model_path = ARTIFACT_DIR / "breast_cancer_svm.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)

# luu metadata
metadata = {
    "model_name": "breast-cancer-svm-rbf",
    "model_version": "1.0.0",
    "random_state": RS,
    "feature_names": list(data.feature_names),
    "class_mapping": {
        "0": "malignant",
        "1": "benign"
    },
    "best_params": search.best_params_,
    "cv_score": float(search.best_score_),
    "test_sensitivity_malignant": float(sensitivity),
    "test_precision_malignant": float(precision_malignant),
    "test_roc_auc_malignant": float(roc_auc),
    "sklearn_version": sklearn.__version__,
    "warning": (
        "Educational use only; "
        "not a medical diagnosis."
    )
}

metadata_path = ARTIFACT_DIR / "metadata.json"
with open(
    metadata_path, "w", encoding="utf-8"
) as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Metadata saved to:", metadata_path)
print("\nTraining completed successfully.")