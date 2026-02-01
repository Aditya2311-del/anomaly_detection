import pandas as pd

class AggregationFeatureEngineer:
    def extract_user_aggregates(self, df, user_id_column='user_id'):
        agg = df.groupby(user_id_column)['amount'].agg([
            'mean',
            'std',
            'max',
            'count'
        ]).reset_index()

        agg.columns = [
            user_id_column,
            'user_mean_amount',
            'user_std_amount',
            'user_max_amount',
            'user_txn_count'
        ]

        return df.merge(agg, on=user_id_column, how='left')
