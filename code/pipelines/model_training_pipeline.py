import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from code.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from code.models.evaluation_metrics import FraudDetectionMetrics

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv(
    "transactions.csv",
    parse_dates=["timestamp"]
)

# Basic cleaning
df = df.drop_duplicates()
df = df.dropna(subset=["user_id", "amount", "timestamp", "is_fraud"])
df = df.sort_values(["user_id", "timestamp"])

print(f"Total transactions: {len(df):,}")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================
feature_pipeline = FeatureEngineeringPipeline(time_col="timestamp")
df = feature_pipeline.fit_transform(df)

# Columns to exclude
exclude_cols = [
    "is_fraud",
    "user_id",
    "timestamp",
    "transaction_id"
]

feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df["is_fraud"]

print(f"Total features used: {len(feature_cols)}")

# =========================================================
# 3. TRAIN / TEST SPLIT (STRATIFIED)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Train fraud rate: {y_train.mean():.2%}")
print(f"Test fraud rate: {y_test.mean():.2%}")

# =========================================================
# 4. HANDLE CLASS IMBALANCE
# =========================================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# =========================================================
# 5. LIGHTGBM TRAINING
# =========================================================
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": scale_pos_weight,
    "verbosity": -1,
    "seed": 42
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(
    params=params,
    train_set=train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    early_stopping_rounds=50,
    verbose_eval=100
)

# =========================================================
# 6. EVALUATION
# =========================================================
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int)

evaluator = FraudDetectionMetrics(
    cost_fp=10.0,
    cost_fn=500.0
)

metrics = evaluator.calculate_metrics(
    y_true=y_test,
    y_pred=y_pred,
    y_prob=y_pred_proba
)

print("\n===== MODEL PERFORMANCE =====")
for k, v in metrics.items():
    print(f"{k}: {v}")

print(f"\nROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# =========================================================
# 7. SAVE MODEL + METADATA
# =========================================================
model.save_model("fraud_model.txt")
joblib.dump(feature_cols, "feature_columns.pkl")

metadata = {
    "model_type": "LightGBM",
    "features_used": len(feature_cols),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "fraud_rate_train": float(y_train.mean()),
    "fraud_rate_test": float(y_test.mean())
}

joblib.dump(metadata, "model_metadata.pkl")

print("\n✅ Model training complete")
print("Saved:")
print(" - fraud_model.txt")
print(" - feature_columns.pkl")
print(" - model_metadata.pkl")
