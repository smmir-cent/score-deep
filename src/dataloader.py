import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

datasets_path = ""
datasets_info = {
    "uci_german": {
        "cat_cols": [
            "Status_checking_account",
            "Credit_history",
            "Purpose",
            "Savings_account_bonds",
            "Present_employment_since",
            "Personal_status_sex",
            "Other_debtors_guarantors",
            "Property",
            "Other_instalment_plans",
            "Housing",
            "Job",
            "Telephone",
            "Foreign_worker"
        ],
        "num_cols": [
            "Duration_months",
            "Credit_amount",
            "Instalment_rate_percent_of_income",
            "Present_residence_since",
            "Age_years",
            "Number_of_existing_credits",
            "Dependants"
        ],
        "target_col": "Status_loan"
    },
    "uci_taiwan": {
        "cat_cols": [
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6"
        ],
        "num_cols": [
            "LIMIT_BAL",
            "AGE",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6"
        ],
        "target_col": "default payment next month"
    },
    "pakdd": {
        "cat_cols": [
            "PAYMENT_DAY",
            "APPLICATION_SUBMISSION_TYPE",
            "POSTAL_ADDRESS_TYPE",
            "SEX",
            "MARITAL_STATUS",
            "STATE_OF_BIRTH",
            "NACIONALITY",
            "RESIDENCIAL_STATE",
            "FLAG_RESIDENCIAL_PHONE",
            "RESIDENCE_TYPE",
            "FLAG_EMAIL",
            "FLAG_VISA",
            "FLAG_MASTERCARD",
            "FLAG_DINERS",
            "FLAG_AMERICAN_EXPRESS",
            "FLAG_OTHER_CARDS",
            "QUANT_BANKING_ACCOUNTS",
            "QUANT_SPECIAL_BANKING_ACCOUNTS",
            "COMPANY",
            "PROFESSIONAL_STATE",
            "FLAG_PROFESSIONAL_PHONE",
            "PROFESSION_CODE",
            "OCCUPATION_TYPE",
            "MATE_EDUCATION_LEVEL",
            "PRODUCT"
        ],
        "num_cols": [
            "PERSONAL_MONTHLY_INCOME",
            "OTHER_INCOMES",
            "PERSONAL_ASSETS_VALUE",
            "AGE",
            "MONTHS_IN_RESIDENCE",
            "QUANT_DEPENDANTS",
            "QUANT_CARS",
            "MONTHS_IN_THE_JOB"
        ],
        "target_col": "TARGET_BAD"
    },
    "hmeq": {
        "cat_cols": [
            "REASON",
            "JOB"
        ],
        "num_cols": [
            "LOAN",
            "MORTDUE",
            "VALUE",
            "YOJ",
            "DEROG",
            "DELINQ",
            "CLAGE",
            "NINQ",
            "CLNO",
            "DEBTINC"
        ],
        "target_col": "BAD"
    },
    "gmsc": {
        "cat_cols": [],
        "num_cols": [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents"
        ],
        "target_col": "SeriousDlqin2yrs"
    }
}

def get_datasets(names_only: bool = False):
    DATASET_DICT = {
        'uci_german': load_german,
        'uci_taiwan': load_taiwan,
        'pakdd': load_pakdd,
        'hmeq': load_homeeq,
        'gmsc': load_gmsc,
    }

    if names_only:
        return list(DATASET_DICT.keys())
    else:
        return DATASET_DICT

def get_datasets_nums_cat(ds_name):
    global datasets_info
    return datasets_info[ds_name]
    
def load_data(dataset: str, ds_path: str):
    global datasets_path
    datasets_path = ds_path
    logging.debug(f'Dataloader: Loading {dataset}')

    dataset_dict = get_datasets()

    logging.debug(f'Dataloader: Loaded available datasets.')

    if dataset in dataset_dict.keys():
        func = dataset_dict[dataset]
        df, cat_cols, num_cols, target_col = func()
    else:
        logging.error(f'Dataloader: Dataset {dataset} not found.')
        raise ValueError(f'Dataloader: Dataset "{dataset}" not found.')

    df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes.astype('category'))

    logging.info(f'Dataloader: Loaded dataset: {dataset}. Returning data.')

    # datasets_info[dataset] = {}
    # datasets_info[dataset]["cat_cols"] = cat_cols
    # datasets_info[dataset]["num_cols"] = num_cols
    # datasets_info[dataset]["target_col"] = target_col
    # if dataset == "gmsc":
    #     import json
    #     print(json.dumps(datasets_info, indent=4))
    return df, cat_cols, num_cols, target_col


# #### statlog
# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

def load_german():
    path = datasets_path + 'uci_german/german.data'

    col_names = ['Status_checking_account', 'Duration_months', 'Credit_history', 'Purpose',
                 'Credit_amount', 'Savings_account_bonds', 'Present_employment_since',
                 'Instalment_rate_percent_of_income', 'Personal_status_sex', 'Other_debtors_guarantors',
                 'Present_residence_since', 'Property', 'Age_years', 'Other_instalment_plans',
                 'Housing', 'Number_of_existing_credits', 'Job', 'Dependants', 'Telephone',
                 'Foreign_worker', 'Status_loan']

    cat_cols = ['Status_checking_account', 'Credit_history', 'Purpose',
                'Savings_account_bonds', 'Present_employment_since', 'Personal_status_sex',
                'Other_debtors_guarantors', 'Property', 'Other_instalment_plans', 'Housing',
                'Job', 'Telephone', 'Foreign_worker']
    num_cols = ['Duration_months', 'Credit_amount', 'Instalment_rate_percent_of_income',
                'Present_residence_since', 'Age_years', 'Number_of_existing_credits', 'Dependants']

    target_col = 'Status_loan'

    df = pd.read_csv(path, sep=' ', header=None, index_col=False,
                     names=col_names,
                     dtype={col: 'category' for col in cat_cols})

    df[target_col] = df[target_col] - 1

    return df, cat_cols, num_cols, target_col


# ####  PAKDD2010
# https://github.com/JLZml/Credit-Scoring-Data-Sets/blob/master/2.%20PAKDD%202009%20Data%20Mining%20Competition/PAKDD%202010.zip
# http://sede.neurotech.com.br:443/PAKDD2009/

def load_pakdd():
    path = datasets_path + 'pakdd/PAKDD2010_Modeling_Data.txt'

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
                'RESIDENCIAL_PHONE_AREA_CODE', 'RESIDENCE_TYPE', 'FLAG_EMAIL', 'FLAG_VISA', 'FLAG_MASTERCARD',
                'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS', 'QUANT_BANKING_ACCOUNTS',
                'QUANT_SPECIAL_BANKING_ACCOUNTS', 'COMPANY', 'PROFESSIONAL_STATE', 'FLAG_PROFESSIONAL_PHONE',
                'PROFESSIONAL_PHONE_AREA_CODE', 'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE',
                'MATE_EDUCATION_LEVEL', 'PRODUCT']

    num_cols = ['PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES', 'PERSONAL_ASSETS_VALUE', 'AGE', 'MONTHS_IN_RESIDENCE',
                'QUANT_DEPENDANTS', 'QUANT_CARS', 'MONTHS_IN_THE_JOB']

    target_col = 'TARGET_BAD'

    drop_cols = ['CITY_OF_BIRTH', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH', 'PROFESSIONAL_CITY',
                 'PROFESSIONAL_BOROUGH', 'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3', 'FLAG_HOME_ADDRESS_DOCUMENT',
                 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF', 'FLAG_ACSP_RECORD', 'CLERK_TYPE', 'QUANT_ADDITIONAL_CARDS',
                 'EDUCATION_LEVEL', 'FLAG_MOBILE_PHONE']

    df = pd.read_csv(path, sep='\t',
                     index_col='ID_CLIENT', encoding='unicode_escape',
                     header=None, names=columns, low_memory=False, 
                     dtype={col: 'category' for col in cat_cols}).drop(drop_cols, axis=1)

    return df, cat_cols, num_cols, target_col


# #### Taiwan
# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

def load_taiwan():
    path = datasets_path + 'uci_taiwan/default of credit card clients.xls'

    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    target_col = 'default payment next month'

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
    print(df) 

    num_cols = [c for c in df.columns if c not in cat_cols and c != target_col]

    return df, cat_cols, num_cols, target_col


# #### homeeq
# http://www.creditriskanalytics.net/datasets-private2.html

def load_homeeq():
    path = datasets_path + 'hmeq/hmeq.csv'

    cat_cols = ['REASON', 'JOB']
    num_cols = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG',
                'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']

    target_col = 'BAD'

    df = pd.read_csv(path, sep=',', index_col=False,
                     dtype={col: 'category' for col in cat_cols})

    return df, cat_cols, num_cols, target_col


# #### Give me credit
# https://www.kaggle.com/c/GiveMeSomeCredit/data?select=cs-training.csv

def load_gmsc():
    path = datasets_path + 'gmsc/cs-training.csv'
    cat_cols = []

    target_col = 'SeriousDlqin2yrs'

    df = pd.read_csv(path, sep=',', index_col=0,
                     dtype={col: 'category' for col in cat_cols})

    num_cols = [c for c in df.columns if c != target_col]

    return df, cat_cols, num_cols, target_col

def preprocess_data(X_train, X_val, X_test, num_cols, cat_cols):
    num_prep = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler())
    cat_prep = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse=False))
    prep = ColumnTransformer([('num', num_prep, num_cols), ('cat', cat_prep, cat_cols)], remainder='drop')
    X_train_trans = prep.fit_transform(X_train)
    X_test_trans = prep.transform(X_test)
    X_val_trans = prep.transform(X_val)
    return X_train_trans, X_val_trans, X_test_trans, prep
