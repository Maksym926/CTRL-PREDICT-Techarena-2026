from sklearn.metrics import roc_auc_score


def auc(y_true, y_pred) -> float:
    """Competition metric: probability a random positive is ranked above a random negative."""
    return roc_auc_score(y_true, y_pred)
