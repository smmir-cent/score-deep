from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE)
from imblearn.under_sampling import (RandomUnderSampler, NearMiss, EditedNearestNeighbours, TomekLinks)
from imblearn.combine import (SMOTEENN, SMOTETomek)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from helpers import evaluate_model, store_results, rank_models
import numpy

resampling_methods = {
    'ros': RandomOverSampler(random_state=2020),
    'rus': RandomUnderSampler(random_state=2020),
    'nearmiss': NearMiss(version=1, n_neighbors=3),
    'enn': EditedNearestNeighbours(),
    'tomek': TomekLinks(),
    'smote': SMOTE(random_state=2020),
    'smote_bs1': BorderlineSMOTE(kind='borderline-1', random_state=2020),
    'smote_bs2': BorderlineSMOTE(kind='borderline-2', random_state=2020),
    'smote_enn': SMOTEENN(random_state=2020),
    'smote_tomek': SMOTETomek(random_state=2020),
    'adasyn': ADASYN(random_state=2020)
}

classifiers = {
    'Random Forest': RandomForestClassifier(random_state=2020),
    'AdaBoost': AdaBoostClassifier(random_state=2020),
    'Gradient Boosting': GradientBoostingClassifier(random_state=2020)
}

def resample_data(X, y, method):
    resampler = resampling_methods.get(method)
    if resampler:
        return resampler.fit_resample(X, y)
    raise ValueError(f"Resampling method '{method}' not found.")

def train_classifier(X_train, y_train, clf_type):
    clf = classifiers.get(clf_type)
    if clf is not None:
        clf.fit(X_train, y_train)
        return clf
    raise ValueError(f"Classifier '{clf_type}' not found.")


def run_pipeline(X_train_trans, y_train, X_val_trans, y_val, X_test_trans, y_test, resample_method='ros', clf_type='rf'):
    X_res, y_res = resample_data(X_train_trans, y_train, resample_method)
    clf = train_classifier(X_res, y_res, clf_type)
    print("Validation Set Evaluation:")
    evaluate_model(clf, X_val_trans, y_val)
    print("Test Set Evaluation:")
    return evaluate_model(clf, X_test_trans, y_test)


def run_all(preprocessed_datasets):
    for ds_name in preprocessed_datasets:
        for resample_method in resampling_methods:
            for clf_type in classifiers:
                print(f"############### Dataset: {ds_name}, resampling method: {resample_method}, classifier: {clf_type} ###############")
                f1, roc_auc, auc_pr = run_pipeline(preprocessed_datasets[ds_name]["X_train_trans"], preprocessed_datasets[ds_name]["y_train"], preprocessed_datasets[ds_name]["X_val_trans"], preprocessed_datasets[ds_name]["y_val"], preprocessed_datasets[ds_name]["X_test_trans"], preprocessed_datasets[ds_name]["y_test"], resample_method, clf_type)
                store_results(ds_name, resample_method, clf_type, 'F1-Score', f1)
                store_results(ds_name, resample_method, clf_type, 'AUC-ROC', roc_auc)
                store_results(ds_name, resample_method, clf_type, 'AUC-PR', auc_pr)
            #     break
            # break
        rank_models()
        break

