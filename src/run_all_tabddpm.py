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

def merge_and_preprocess_training_set(original_dir, generated_dir, output_dir, num_cols, cat_cols, target_col, ds_name):
    """
    Merge training sets from original and generated data directories, 
    adding only minority class synthetic samples.
    """
    X_num_test = np.load(f"{original_dir}/X_num_test.npy", allow_pickle=True)
    X_cat_test = np.load(f"{original_dir}/X_cat_test.npy", allow_pickle=True)
    y_test = np.load(f"{original_dir}/y_test.npy", allow_pickle=True)    
    df_orig_test = pd.DataFrame(X_num_test, columns=num_cols)
    for i, col in enumerate(cat_cols):
        df_orig_test[col] = X_cat_test[:, i]
    df_orig_test[target_col] = y_test

    X_num_val = np.load(f"{original_dir}/X_num_val.npy", allow_pickle=True)
    X_cat_val = np.load(f"{original_dir}/X_cat_val.npy", allow_pickle=True)
    y_val = np.load(f"{original_dir}/y_val.npy", allow_pickle=True)    
    df_orig_val = pd.DataFrame(X_num_val, columns=num_cols)
    for i, col in enumerate(cat_cols):
        df_orig_val[col] = X_cat_val[:, i]
    df_orig_val[target_col] = y_val


    os.makedirs(output_dir, exist_ok=True)
    
    suffix = 'train'
    print(f"Processing {suffix} set...")

    # Load original training data
    X_num_orig = np.load(f"{original_dir}/X_num_{suffix}.npy", allow_pickle=True)
    X_cat_orig = np.load(f"{original_dir}/X_cat_{suffix}.npy", allow_pickle=True)
    y_orig = np.load(f"{original_dir}/y_{suffix}.npy", allow_pickle=True)

    # Convert original data into DataFrame
    df_orig = pd.DataFrame(X_num_orig, columns=num_cols)
    for i, col in enumerate(cat_cols):
        df_orig[col] = X_cat_orig[:, i]
    df_orig[target_col] = y_orig

    # Load generated training data
    X_num_gen = np.load(f"{generated_dir}/X_num_{suffix}.npy", allow_pickle=True)
    if len(cat_cols) != 0:
        X_cat_gen = np.load(f"{generated_dir}/X_cat_{suffix}.npy", allow_pickle=True)
    y_gen = np.load(f"{generated_dir}/y_{suffix}.npy", allow_pickle=True)
    # Convert generated data into DataFrame
    df_gen = pd.DataFrame(X_num_gen, columns=num_cols)
    for i, col in enumerate(cat_cols):
        df_gen[col] = X_cat_gen[:, i]
    df_gen[target_col] = y_gen

    # Filter only minority class synthetic samples
    df_gen_minority = df_gen[df_gen[target_col] == 1]
    df_combined = pd.concat([df_orig, df_gen_minority], axis=0).reset_index(drop=True)
    # Separate features and target
    X_combined = df_combined[num_cols + cat_cols]
    y_combined = df_combined[target_col].to_numpy()
    X_test = df_orig_test[num_cols + cat_cols]
    y_test = df_orig_test[target_col].to_numpy()
    X_val = df_orig_val[num_cols + cat_cols]
    y_val = df_orig_val[target_col].to_numpy()    
    # Split into train, val, test (here using all as train for demonstration)
    X_train = X_combined
    y_train = y_combined

    # Preprocess combined dataset
    X_train_trans, X_val_trans, X_test_trans, prep = preprocess_data(X_train, X_val, X_test, num_cols, cat_cols)

    # Save preprocessed data
    np.save(f"{output_dir}/X_train.npy", X_train_trans)
    np.save(f"{output_dir}/X_val.npy", X_val_trans)
    np.save(f"{output_dir}/X_test.npy", X_test_trans)
    np.save(f"{output_dir}/y_train.npy", y_train)
    np.save(f"{output_dir}/y_val.npy", y_val)
    np.save(f"{output_dir}/y_test.npy", y_test)

    # Logging
    log_message = (
        f"[VALIDATION] Data merged successfully:\n"
        f"- Original data: {df_orig.shape}, Labels: {df_orig[target_col].value_counts().to_dict()}\n"
        f"- Generated data: {df_gen.shape}, Labels: {df_gen[target_col].value_counts().to_dict()}\n"
        f"- Combined data: {df_combined.shape}, Labels: {df_combined[target_col].value_counts().to_dict()}"
    )
    print(log_message)

    print(f"Training set merged, preprocessed, and saved in {output_dir}.")
    return X_train_trans, y_train, X_val_trans, y_val, X_test_trans, y_test

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
        run_benchmark(ds_name, X_train_trans, X_val_trans, y_train, y_val)

