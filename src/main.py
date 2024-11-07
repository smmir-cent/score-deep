import logging
from sklearn.model_selection import train_test_split
logging.getLogger().setLevel(logging.DEBUG)

from dataloader import load_data, preprocess_data
from traditionalmodels import run_all
from helpers import get_cat_dims, rank_models
from cwgan import run_all_cwgan

if __name__ == "__main__":
    datasets_path = "data/raw/"
    # dataset_names = ['german', 'taiwan', 'pakdd', 'homeeq', 'gmsc']    
    dataset_names = ['german']
    preprocessed_datasets = {}
    for ds_name in dataset_names:
        print("########## " + ds_name + " ##########")
        preprocessed_datasets[ds_name] = {}
        df, cat_cols, num_cols, target_col = load_data(ds_name, datasets_path)
        X = df.loc[:, num_cols + cat_cols]
        y = df.loc[:, target_col]
        test_size=0.1
        # Splitting data into training (70%), validation (20%), and test (10%)
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=2020)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2222, random_state=2020)  # 0.2222 to get 20% of the original data        
        cat_dims = get_cat_dims(X_train, cat_cols)
        X_train_trans, X_val_trans, X_test_trans, prep = preprocess_data(X_train, X_val, X_test, num_cols, cat_cols)
        preprocessed_datasets[ds_name]["X_train_trans"] = X_train_trans
        preprocessed_datasets[ds_name]["y_train"] = y_train
        preprocessed_datasets[ds_name]["X_val_trans"] = X_val_trans
        preprocessed_datasets[ds_name]["y_val"] = y_val        
        preprocessed_datasets[ds_name]["X_test_trans"] = X_test_trans
        preprocessed_datasets[ds_name]["y_test"] = y_test
        preprocessed_datasets[ds_name]["num_cols"] = num_cols
        preprocessed_datasets[ds_name]["cat_cols"] = cat_cols
        preprocessed_datasets[ds_name]["cat_dims"] = cat_dims
        preprocessed_datasets[ds_name]["prep"] = prep
        
    run_all(preprocessed_datasets)
    # run_all_cwgan(preprocessed_datasets)
    rank_models()
    
        