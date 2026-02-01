import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from code.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from code.models.lightgbm_fraud_detector import LightGBMFraudDetector
from code.models.evaluation_metrics import FraudDetectionMetrics

np.random.seed(42)

df = pd.DataFrame({
    'user_id': np.random.choice([f'user_{i}' for i in range(50)], 5000),
    'amount': np.random.lognormal(4, 1, 5000),
    'timestamp': pd.date_range('2024-01-01', periods=5000, freq='T'),
    'is_fraud': np.random.binomial(1, 0.02, 5000)
})

feature_pipeline = FeatureEngineeringPipeline()
df = feature_pipeline.fit_transform(df)

X = df.drop(columns=['is_fraud', 'timestamp', 'user_id'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

model = LightGBMFraudDetector().train(
    X_train, y_train,
    X_test, y_test
)

probs = model.predict(X_test)
preds = (probs >= 0.5).astype(int)

metrics = FraudDetectionMetrics().calculate_metrics(
    y_test, preds, probs
)

print(metrics)
