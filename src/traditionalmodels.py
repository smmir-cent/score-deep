from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE)
from imblearn.under_sampling import (RandomUnderSampler, NearMiss, EditedNearestNeighbours, TomekLinks)
from imblearn.combine import (SMOTEENN, SMOTETomek)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from helpers import evaluate_model

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
    'rf': RandomForestClassifier(random_state=2020),
    'ada': AdaBoostClassifier(random_state=2020),
    'gb': GradientBoostingClassifier(random_state=2020)
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


def run_pipeline(X_train_trans, y_train, X_test_trans, y_test, resample_method='ros', clf_type='rf'):
    X_res, y_res = resample_data(X_train_trans, y_train, resample_method)
    clf = train_classifier(X_res, y_res, clf_type)
    evaluate_model(clf, X_test_trans, y_test)

def run_all(preprocessed_datasets):
    for ds_name in preprocessed_datasets:
        for resample_method in resampling_methods:
            for clf_type in classifiers:        
                run_pipeline(preprocessed_datasets[ds_name]["X_train_trans"], preprocessed_datasets[ds_name]["y_train"], preprocessed_datasets[ds_name]["X_test_trans"], preprocessed_datasets[ds_name]["y_test"], resample_method, clf_type)

