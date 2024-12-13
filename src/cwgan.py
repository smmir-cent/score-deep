from sklearn.ensemble import RandomForestClassifier
from models import WGANGP
from helpers import evaluate_model, store_results
import numpy as np

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

classifiers = {
    'Random Forest': RandomForestClassifier(random_state=2020),
    'AdaBoost': AdaBoostClassifier(random_state=2020),
    'Gradient Boosting': GradientBoostingClassifier(random_state=2020)
}

def run_wgan_pipeline(preprocessed_dataset, num_cols, cat_cols, cat_dims, prep):
    gan = WGANGP(write_to_disk=False, # whether to create an output folder. Plotting will be surpressed if flase
                compute_metrics_every=1250, print_every=2500, plot_every=10000,
                num_cols = num_cols, cat_dims=cat_dims,
                # pass the one hot encoder to the GAN to enable count plots of categorical variables
                transformer=prep.named_transformers_['cat']['onehotencoder'],
                # pass column names to enable
                cat_cols=cat_cols,
                use_aux_classifier_loss=True,
                d_updates_per_g=3, gp_weight=15)

    condition_num_on_cat = bool(cat_cols)
    X_combined = np.concatenate(
        [preprocessed_dataset["X_train_trans"], preprocessed_dataset["X_val_trans"]],
        axis=0
    )

    y_combined = np.concatenate(
        [preprocessed_dataset["y_train"].values, preprocessed_dataset["y_val"].values],
        axis=0
    )
    gan.fit(X_combined, y=y_combined, 
            condition=True,
            epochs=300,  
            batch_size=64,
            netG_kwargs = {'hidden_layer_sizes': (128,64), 
                            'n_cross_layers': 1,
                            'cat_activation': 'gumbel_softmax',
                            'num_activation': 'none',
                            'condition_num_on_cat': condition_num_on_cat, 
                            'noise_dim': 30, 
                            'normal_noise': False,
                            'activation':  'leaky_relu',
                            'reduce_cat_dim': True,
                            'use_num_hidden_layer': True,
                            'layer_norm':False,},
            netD_kwargs = {'hidden_layer_sizes': (128,64,32),
                            'n_cross_layers': 2,
                            'embedding_dims': 'auto',
                            'activation':  'leaky_relu',
                            'sigmoid_activation': False,
                            'noisy_num_cols': True,
                            'layer_norm':True,}
        )

    X_res, y_res = gan.resample(X_combined, y=y_combined)
    return X_res, y_res

def run_all_cwgan(preprocessed_datasets):
    print("###########################################")
    print("###########################################")
    print("################## CWGAN ##################")
    print("###########################################")
    print("###########################################")
    for ds_name in preprocessed_datasets:
        print("############# " + ds_name + " #############")        
        X_res, y_res = run_wgan_pipeline(preprocessed_datasets[ds_name], preprocessed_datasets[ds_name]["num_cols"], preprocessed_datasets[ds_name]["cat_cols"], preprocessed_datasets[ds_name]["cat_dims"], preprocessed_datasets[ds_name]["prep"])
        for clf_name, clf in classifiers.items():
            print(f"Training and evaluating {clf_name} on {ds_name} dataset...")
            clf.fit(X_res, y_res)
            brier, roc_auc, auc_pr = evaluate_model(clf, preprocessed_datasets[ds_name]["X_test_trans"], preprocessed_datasets[ds_name]["y_test"])
            store_results(ds_name, "cwgan", clf_name, 'Brier-Score', brier)
            store_results(ds_name, "cwgan", clf_name, 'AUC-ROC', roc_auc)
            store_results(ds_name, "cwgan", clf_name, 'AUC-PR', auc_pr)

