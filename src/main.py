import logging
from dataloader import load_data
logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    dataset_names = ['german', 'taiwan', 'pakdd', 'homeeq', 'gmsc']    
    for ds_name in dataset_names:
        df, cat_cols, num_cols, target_col = load_data(ds_name)
        print("########## " + ds_name + " ##########")
        print(df.head())
        print("########## ##########")
        