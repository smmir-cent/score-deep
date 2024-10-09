from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc, f1_score

def get_cat_dims(X, cat_cols) -> list:
    """
    Takes a pd.DataFrame and a list of columns and returns a list of levels/cardinality per column in the same order.
    """
    return [(X[col].nunique()) for col in cat_cols]



def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # Assuming binary classification for AUC metrics
    
    # Confusion Matrix and Classification Report
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # F1-Score
    f1 = f1_score(y_test, y_pred)
    print(f"\nF1-Score: {f1:.4f}")
    
    # AUC-ROC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall, precision)
    print(f"AUC-PR: {auc_pr:.4f}")