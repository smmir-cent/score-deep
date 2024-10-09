import logging
from sklearn.model_selection import train_test_split
logging.getLogger().setLevel(logging.DEBUG)

from dataloader import load_data, preprocess_data
from traditionalmodels import run_all

if __name__ == "__main__":
    datasets_path = "data/raw/"
    dataset_names = ['german', 'taiwan', 'pakdd', 'homeeq', 'gmsc']    
    preprocessed_datasets = {}
    for ds_name in dataset_names:
        print("########## " + ds_name + " ##########")
        preprocessed_datasets[ds_name] = {}
        df, cat_cols, num_cols, target_col = load_data(ds_name, datasets_path)
        X = df.loc[:, num_cols + cat_cols]
        y = df.loc[:, target_col]
        test_size=0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=2020)
        X_train_trans, X_test_trans = preprocess_data(X_train, X_test, num_cols, cat_cols)
        preprocessed_datasets[ds_name]["X_train_trans"] = X_train_trans
        preprocessed_datasets[ds_name]["y_train"] = y_train
        preprocessed_datasets[ds_name]["X_test_trans"] = X_test_trans
        preprocessed_datasets[ds_name]["y_test"] = y_test
        
        
    run_all(preprocessed_datasets)
        
        
        