import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class BehavioralDeviationDetector:
    def __init__(self, z_score_threshold=3.0):
        self.z = z_score_threshold

    def z_score_anomalies(self, df, features, group_by='user_id'):
        df = df.copy()

        for col in features:
            mean = df.groupby(group_by)[col].transform('mean')
            std = df.groupby(group_by)[col].transform('std').replace(0, 1e-6)
            zscore = (df[col] - mean) / std

            df[f'{col}_z'] = zscore
            df['is_anomaly_zscore'] = (zscore.abs() >= self.z).astype(int)

        return df

    def isolation_forest_anomalies(self, df, features, contamination=0.05):
        model = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        preds = model.fit_predict(df[features])
        df['is_anomaly_iforest'] = (preds == -1).astype(int)
        return df
