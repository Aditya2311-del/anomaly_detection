from code.feature_engineering.temporal_features import TemporalFeatureEngineer
from code.feature_engineering.behavioral_deviation import BehavioralDeviationDetector
from code.feature_engineering.aggregation_features import AggregationFeatureEngineer

class FeatureEngineeringPipeline:
    def __init__(self, time_col='timestamp'):
        self.temporal = TemporalFeatureEngineer(time_col)
        self.behavioral = BehavioralDeviationDetector()
        self.aggregation = AggregationFeatureEngineer()

    def fit_transform(self, df):
        df = self.temporal.extract_temporal_features(df)
        df = self.temporal.calculate_time_gaps(df)
        df = self.aggregation.extract_user_aggregates(df)
        df = self.behavioral.z_score_anomalies(df, ['amount'])
        return df
