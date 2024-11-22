import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
import os
from helpers import evaluate_model, store_results
from dataloader import preprocess_data, get_datasets_nums_cat

original_dir = "configuration/datasets/"
generated_dir = "Experiments/"
output_dir = "Merged/"


# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=2020),
    'AdaBoost': AdaBoostClassifier(random_state=2020),
    'Gradient Boosting': GradientBoostingClassifier(random_state=2020)
}

def load_and_prepare_dataframe(num_data, cat_data, target_data, num_cols, cat_cols, target_col):
    """Load numerical, categorical, and target data into a Pandas DataFrame."""
    df = pd.DataFrame(num_data, columns=num_cols)
    for i, col in enumerate(cat_cols):
        df[col] = cat_data[:, i]
    df[target_col] = target_data
    return df

def merge_and_preprocess_training_set(
    original_dir, generated_dir, output_dir, num_cols, cat_cols, target_col, ds_name
):
    """
    Merge training sets from original and generated data directories, 
    adding only minority class synthetic samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original and validation data
    def load_data(suffix):
        X_num = np.load(f"{original_dir}/X_num_{suffix}.npy", allow_pickle=True)
        X_cat = np.load(f"{original_dir}/X_cat_{suffix}.npy", allow_pickle=True)
        y = np.load(f"{original_dir}/y_{suffix}.npy", allow_pickle=True)
        return load_and_prepare_dataframe(X_num, X_cat, y, num_cols, cat_cols, target_col)

    df_orig_train = load_data("train")
    df_orig_val = load_data("val")
    df_orig_test = load_data("test")

    # Load and prepare generated training data
    X_num_gen = np.load(f"{generated_dir}/X_num_train.npy", allow_pickle=True)
    if len(cat_cols) != 0:
        X_cat_gen = np.load(f"{generated_dir}/X_cat_train.npy", allow_pickle=True)
    else: 
        X_cat_gen = None
    y_gen = np.load(f"{generated_dir}/y_train.npy", allow_pickle=True)
    df_gen = load_and_prepare_dataframe(X_num_gen, X_cat_gen, y_gen, num_cols, cat_cols, target_col)

    # Filter minority class samples and combine
    df_gen_minority = df_gen[df_gen[target_col] == 1]
    df_combined = pd.concat([df_orig_train, df_orig_val, df_gen_minority], ignore_index=True)
    
    # Separate features and targets
    X_combined = df_combined[num_cols + cat_cols]
    y_combined = df_combined[target_col].to_numpy()
    X_val = df_orig_val[num_cols + cat_cols]
    y_val = df_orig_val[target_col].to_numpy()
    X_test = df_orig_test[num_cols + cat_cols]
    y_test = df_orig_test[target_col].to_numpy()

    # Preprocess data
    X_train_trans, X_val_trans, X_test_trans, prep = preprocess_data(
        X_combined, X_val, X_test, num_cols, cat_cols
    )

    # Save preprocessed data
    for data, name in zip(
        [X_train_trans, X_val_trans, X_test_trans, y_combined, y_val, y_test],
        ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
    ):
        np.save(f"{output_dir}/{name}.npy", data)

    # Log dataset summary
    log_message = (
        f"[VALIDATION] Data merged successfully:\n"
        f"- Original training data: {df_orig_train.shape}, Labels: {df_orig_train[target_col].value_counts().to_dict()}\n"
        f"- Generated data: {df_gen.shape}, Labels: {df_gen[target_col].value_counts().to_dict()}\n"
        f"- Combined data: {df_combined.shape}, Labels: {df_combined[target_col].value_counts().to_dict()}"
    )
    print(log_message)

    print(f"Training set merged, preprocessed, and saved in {output_dir}.")
    return X_train_trans, y_combined, X_val_trans, y_val, X_test_trans, y_test

def run_benchmark(ds_name, X_train, X_test, y_train, y_test):
    for clf_name, clf in classifiers.items():
        print(f"Training and evaluating {clf_name} on {ds_name} dataset...")
        clf.fit(X_train, y_train)
        f1, roc_auc, auc_pr = evaluate_model(clf, X_test, y_test)
        
        # Store results
        store_results(ds_name, "tabddpm" ,clf_name, 'F1-Score', f1)
        store_results(ds_name, "tabddpm" ,clf_name, 'AUC-ROC', roc_auc)
        store_results(ds_name, "tabddpm" ,clf_name, 'AUC-PR', auc_pr)

        print(f"Results for {clf_name}:\n F1-Score: {f1:.4f}, AUC-ROC: {roc_auc:.4f}, AUC-PR: {auc_pr:.4f}\n")

def run_all_diff(ds_names):    
    for ds_name in ds_names:
        infos = get_datasets_nums_cat(ds_name)        
        X_train_trans, y_train, X_val_trans, y_val, X_test_trans, y_test = merge_and_preprocess_training_set(original_dir + ds_name, generated_dir + ds_name, output_dir + ds_name, infos["num_cols"], infos["cat_cols"], infos["target_col"], ds_name)
        run_benchmark(ds_name, X_train_trans, X_test_trans, y_train, y_test)

