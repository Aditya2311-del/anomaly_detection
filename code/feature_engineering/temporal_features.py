import pandas as pd
import numpy as np

class TemporalFeatureEngineer:
    def __init__(self, time_column: str):
        self.time_column = time_column

    def extract_temporal_features(self, df, user_id_column='user_id'):
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])

        df['hour_of_day'] = df[self.time_column].dt.hour
        df['day_of_week'] = df[self.time_column].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour_of_day'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

        df['velocity_1h'] = (
            df.groupby(user_id_column)[self.time_column]
              .diff()
              .dt.total_seconds()
              .fillna(1e9)
              .lt(3600)
              .groupby(df[user_id_column])
              .cumsum()
        )

        return df

    def calculate_time_gaps(self, df, user_id_column='user_id'):
        df = df.copy()

        df['time_gap_minutes'] = (
            df.groupby(user_id_column)[self.time_column]
              .diff()
              .dt.total_seconds()
              .div(60)
              .fillna(1e9)
        )

        df['unusual_time_gap'] = (df['time_gap_minutes'] < 1).astype(int)
        return df
