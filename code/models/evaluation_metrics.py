from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

class FraudDetectionMetrics:
    def __init__(self, cost_fp=5.0, cost_fn=500.0):
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn

    def calculate_metrics(self, y_true, y_pred, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob),
            'business_cost': fp * self.cost_fp + fn * self.cost_fn,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
