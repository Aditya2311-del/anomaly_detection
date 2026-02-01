import pandas as pd

class FraudPredictionPipeline:
    def __init__(self, feature_pipeline, model, threshold=0.5):
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.threshold = threshold

    def predict(self, transaction_dict):
        df = pd.DataFrame([transaction_dict])
        df = self.feature_pipeline.fit_transform(df)

        X = df.drop(columns=['is_fraud'], errors='ignore')
        prob = self.model.predict(X)[0]

        return {
            'fraud_probability': float(prob),
            'decision': 'BLOCK' if prob >= self.threshold else 'APPROVE'
        }
