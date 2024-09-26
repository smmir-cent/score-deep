from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score, average_precision_score, mean_squared_error, r2_score, brier_score_loss

def get_cat_dims(X, cat_cols) -> list:
    """
    Takes a pd.DataFrame and a list of columns and returns a list of levels/cardinality per column in the same order.
    """
    return [(X[col].nunique()) for col in cat_cols]
