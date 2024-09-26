import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from helpers import get_cat_dims
logging.getLogger().setLevel(logging.DEBUG)

from dataloader import load_data


if __name__ == "__main__":
    datasets_path = "data/raw/"
    dataset_names = ['german', 'taiwan', 'pakdd', 'homeeq', 'gmsc']    
    for ds_name in dataset_names:
        df, cat_cols, num_cols, target_col = load_data(ds_name, datasets_path)
        print("########## " + ds_name + " ##########")
        print(df.head())
        X = df.loc[:, num_cols + cat_cols]
        y = df.loc[:, target_col]

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2020)
        cat_dims = get_cat_dims(X_train, cat_cols)

        # preprocess data
        num_prep = make_pipeline(SimpleImputer(strategy='mean'),
                                MinMaxScaler())
        cat_prep = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                OneHotEncoder(handle_unknown='ignore', sparse=False))
        prep = ColumnTransformer([
            ('num', num_prep, num_cols),
            ('cat', cat_prep, cat_cols)],
            remainder='drop')
        X_train_trans = prep.fit_transform(X_train) 
        print(X_train_trans)       
        print("########## ##########")