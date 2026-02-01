import lightgbm as lgb

class LightGBMFraudDetector:
    def train(self, X_train, y_train, X_val, y_val):
        train_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_val, y_val, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'is_unbalance': True,
            'verbosity': -1
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            early_stopping_rounds=50
        )

        return self.model
