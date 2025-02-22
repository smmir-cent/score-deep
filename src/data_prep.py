import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

def save_data(data_dir, X_num, X_cat, y, split_name):
    """Save numerical, categorical, and target arrays to .npy files."""
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, f'X_num_{split_name}.npy'), X_num)
    np.save(os.path.join(data_dir, f'X_cat_{split_name}.npy'), X_cat)
    np.save(os.path.join(data_dir, f'y_{split_name}.npy'), y)


def pr_data(X_train, X_val, X_test, num_cols, cat_cols):
    """Preprocess numerical and categorical data and split them separately."""
    num_prep = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        # MinMaxScaler()
    )
    cat_prep = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        # OneHotEncoder(handle_unknown='ignore', sparse=False)
    )
    prep = ColumnTransformer(
        [('num', num_prep, num_cols), ('cat', cat_prep, cat_cols)],
        remainder='drop'
    )
    X_train_trans = prep.fit_transform(X_train)
    X_val_trans = prep.transform(X_val)
    X_test_trans = prep.transform(X_test)
    
    num_cols_transformed = [i for i, col in enumerate(num_cols)]
    cat_cols_transformed = [i + len(num_cols) for i, col in enumerate(cat_cols)]

    X_num_train = X_train_trans[:, num_cols_transformed]
    X_cat_train = X_train_trans[:, cat_cols_transformed]
    X_num_val = X_val_trans[:, num_cols_transformed]
    X_cat_val = X_val_trans[:, cat_cols_transformed]
    X_num_test = X_test_trans[:, num_cols_transformed]
    X_cat_test = X_test_trans[:, cat_cols_transformed]

    return (
        X_num_train.astype(np.int64), 
        X_cat_train, 
        X_num_val.astype(np.int64), 
        X_cat_val, 
        X_num_test.astype(np.int64), 
        X_cat_test, 
        prep
    )
    # return X_num_train, X_cat_train, X_num_val, X_cat_val, X_num_test, X_cat_test, prep

def preprocess_data(X, y, num_cols, cat_cols, data_dir):
    """Preprocess and save dataset splits as .npy files."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    X_num_train, X_cat_train, X_num_val, X_cat_val, X_num_test, X_cat_test, prep = pr_data(
        X_train, X_val, X_test, num_cols, cat_cols
    )

    # Encode target variable if needed
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train).reshape(-1,)
    y_val = label_encoder.transform(y_val).reshape(-1,)
    y_test = label_encoder.transform(y_test).reshape(-1,)

    print("train set: ", X_num_train.shape, X_cat_train.shape, y_train.shape)
    print("val set: ", X_num_val.shape, X_cat_val.shape, y_val.shape)
    print("test set: ", X_num_test.shape, X_cat_test.shape, y_test.shape)

    # Save all splits
    save_data(data_dir, X_num_train, X_cat_train, y_train, 'train')
    save_data(data_dir, X_num_val, X_cat_val, y_val, 'val')
    save_data(data_dir, X_num_test, X_cat_test, y_test, 'test')

def load_german_data():
    """Load and preprocess the German dataset."""
    col_names = ['Status_checking_account', 'Duration_months', 'Credit_history', 'Purpose',
                 'Credit_amount', 'Savings_account_bonds', 'Present_employment_since',
                 'Instalment_rate_percent_of_income', 'Personal_status_sex', 'Other_debtors_guarantors',
                 'Present_residence_since', 'Property', 'Age_years', 'Other_instalment_plans',
                 'Housing', 'Number_of_existing_credits', 'Job', 'Dependants', 'Telephone',
                 'Foreign_worker', 'Status_loan']

    cat_cols = ['Status_checking_account', 'Credit_history', 'Purpose', 'Savings_account_bonds',
                'Present_employment_since', 'Personal_status_sex', 'Other_debtors_guarantors',
                'Property', 'Other_instalment_plans', 'Housing', 'Job', 'Telephone', 'Foreign_worker']
    num_cols = ['Duration_months', 'Credit_amount', 'Instalment_rate_percent_of_income',
                'Present_residence_since', 'Age_years', 'Number_of_existing_credits', 'Dependants']
    target_col = 'Status_loan'

    # Load dataset
    df = pd.read_csv('data/raw/uci_german/german.data', sep=' ', header=None, index_col=False, names=col_names,
                     dtype={col: 'category' for col in cat_cols})
            
    df[target_col] = df[target_col] - 1  # Adjust target labels if necessary

    X = df.loc[:, num_cols + cat_cols]
    y = df[target_col]

    # Preprocess and save
    preprocess_data(X, y, num_cols, cat_cols, 'configuration/datasets/uci_german')

def load_pakdd_data():
    path = 'data/raw/pakdd/PAKDD2010_Modeling_Data.txt'
    
    columns = ["ID_CLIENT", "CLERK_TYPE", "PAYMENT_DAY", "APPLICATION_SUBMISSION_TYPE", "QUANT_ADDITIONAL_CARDS",
               "POSTAL_ADDRESS_TYPE", "SEX", "MARITAL_STATUS", "QUANT_DEPENDANTS", "EDUCATION_LEVEL", "STATE_OF_BIRTH",
               "CITY_OF_BIRTH", "NACIONALITY", "RESIDENCIAL_STATE", "RESIDENCIAL_CITY", "RESIDENCIAL_BOROUGH",
               "FLAG_RESIDENCIAL_PHONE", "RESIDENCIAL_PHONE_AREA_CODE", "RESIDENCE_TYPE", "MONTHS_IN_RESIDENCE",
               "FLAG_MOBILE_PHONE", "FLAG_EMAIL", "PERSONAL_MONTHLY_INCOME", "OTHER_INCOMES", "FLAG_VISA",
               "FLAG_MASTERCARD", "FLAG_DINERS", "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS", "QUANT_BANKING_ACCOUNTS",
               "QUANT_SPECIAL_BANKING_ACCOUNTS", "PERSONAL_ASSETS_VALUE", "QUANT_CARS", "COMPANY", "PROFESSIONAL_STATE",
               "PROFESSIONAL_CITY", "PROFESSIONAL_BOROUGH", "FLAG_PROFESSIONAL_PHONE", "PROFESSIONAL_PHONE_AREA_CODE",
               "MONTHS_IN_THE_JOB", "PROFESSION_CODE", "OCCUPATION_TYPE", "MATE_PROFESSION_CODE",
               "MATE_EDUCATION_LEVEL", "FLAG_HOME_ADDRESS_DOCUMENT", "FLAG_RG", "FLAG_CPF", "FLAG_INCOME_PROOF",
               "PRODUCT", "FLAG_ACSP_RECORD", "AGE", "RESIDENCIAL_ZIP_3", "PROFESSIONAL_ZIP_3", "TARGET_BAD"]

    cat_cols = ['PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE', 'POSTAL_ADDRESS_TYPE', 'SEX', 'MARITAL_STATUS',
                'STATE_OF_BIRTH', 'NACIONALITY', 'RESIDENCIAL_STATE', 'FLAG_RESIDENCIAL_PHONE',
                'RESIDENCE_TYPE', 'FLAG_EMAIL', 'FLAG_VISA', 'FLAG_MASTERCARD',
                'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS', 'QUANT_BANKING_ACCOUNTS',
                'QUANT_SPECIAL_BANKING_ACCOUNTS', 'COMPANY', 'PROFESSIONAL_STATE', 'FLAG_PROFESSIONAL_PHONE',
                'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_EDUCATION_LEVEL', 'PRODUCT']

    num_cols = ['PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES', 'PERSONAL_ASSETS_VALUE', 'AGE', 'MONTHS_IN_RESIDENCE',
                'QUANT_DEPENDANTS', 'QUANT_CARS', 'MONTHS_IN_THE_JOB']

    target_col = 'TARGET_BAD'

    drop_cols = ['CITY_OF_BIRTH', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH', 'PROFESSIONAL_CITY',
                 'PROFESSIONAL_BOROUGH', 'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3', 'FLAG_HOME_ADDRESS_DOCUMENT',
                 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF', 'FLAG_ACSP_RECORD', 'CLERK_TYPE', 'QUANT_ADDITIONAL_CARDS',
                 'EDUCATION_LEVEL', 'FLAG_MOBILE_PHONE', 'PROFESSIONAL_PHONE_AREA_CODE', 'MATE_PROFESSION_CODE', 'RESIDENCIAL_PHONE_AREA_CODE']

    df = pd.read_csv(path, sep='\t',
                     index_col='ID_CLIENT', encoding='unicode_escape',
                     header=None, names=columns, low_memory=False, 
                     dtype={col: 'category' for col in cat_cols}).drop(drop_cols, axis=1)
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            # Map each categorical value to a unique string like "cat_1", "cat_2", ...
            unique_values = df[col].cat.categories
            mapping = {val: f"cat_{i+1}" for i, val in enumerate(unique_values)}
            df[col] = df[col].map(mapping).astype('category')
        else:
            print(f"Warning: Column {col} not found in the dataset.")
    
    X = df.loc[:, num_cols + cat_cols]
    y = df[target_col]

    # Preprocess and save
    preprocess_data(X, y, num_cols, cat_cols, 'configuration/datasets/pakdd')
    
def load_taiwan():
    path = 'data/raw/uci_taiwan/default of credit card clients.xls'

    # Define categorical and numerical columns
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    num_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", 
                "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", 
                "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    target_col = 'default payment next month'

    # Read the dataset
    df = pd.read_excel(path, index_col=0)
    
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            # Map each categorical value to a unique string like "cat_1", "cat_2", ...
            unique_values = df[col].cat.categories
            mapping = {val: f"cat_{i+1}" for i, val in enumerate(unique_values)}
            df[col] = df[col].map(mapping).astype('category')
        else:
            print(f"Warning: Column {col} not found in the dataset.")
    # Select features and target
    X = df.loc[:, num_cols + cat_cols]
    y = df[target_col]

    # Preprocess and save
    preprocess_data(X, y, num_cols, cat_cols, 'configuration/datasets/uci_taiwan')


def load_hmeq_data():
    path = 'data/raw/hmeq/hmeq.csv'

    cat_cols = ['REASON', 'JOB']
    num_cols = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG',
                'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
    target_col = 'BAD'

    # Load dataset with specified column types
    df = pd.read_csv(path, sep=',', index_col=False,
                     dtype={col: 'category' for col in cat_cols})

    # Separate features and target
    X = df.loc[:, num_cols + cat_cols]
    y = df[target_col]

    # Preprocess and save
    preprocess_data(X, y, num_cols, cat_cols, 'configuration/datasets/hmeq')

def load_gmsc_data():
    path = 'data/raw/gmsc/cs-training.csv'
    cat_cols = []

    target_col = 'SeriousDlqin2yrs'

    df = pd.read_csv(path, sep=',', index_col=0,
                     dtype={col: 'category' for col in cat_cols})  

    num_cols = [c for c in df.columns if c != target_col]
    X = df.loc[:, num_cols + cat_cols]
    y = df[target_col]

    # Preprocess and save
    preprocess_data(X, y, num_cols, cat_cols, 'configuration/datasets/gmsc')

def main():
    print("german_data dataset preparation")
    load_german_data()
    print("pakdd_data dataset preparation")
    load_pakdd_data()
    print("hmeq_data dataset preparation")
    load_hmeq_data()
    print("taiwan dataset preparation")
    load_taiwan()
    print("gmsc_data dataset preparation")
    load_gmsc_data()
    

if __name__ == "__main__":
    main()