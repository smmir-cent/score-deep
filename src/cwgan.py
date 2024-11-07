from sklearn.ensemble import RandomForestClassifier
from models import WGANGP
from helpers import evaluate_model
import torch
import time

def run_wgan_pipeline(preprocessed_dataset, num_cols, cat_cols, cat_dims, prep):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")    
    # num_cols = num_cols.to(device)
    # cat_cols = cat_cols.to(device)
    # prep = prep.to(device)
    gan = WGANGP(write_to_disk=False, # whether to create an output folder. Plotting will be surpressed if flase
                compute_metrics_every=1250, print_every=2500, plot_every=10000000000,
                num_cols = num_cols, cat_dims=cat_dims,
                # pass the one hot encoder to the GAN to enable count plots of categorical variables
                transformer=prep.named_transformers_['cat']['onehotencoder'],
                # pass column names to enable
                cat_cols=cat_cols,
                use_aux_classifier_loss=True,
                d_updates_per_g=3, gp_weight=15)
    # gan = gan.to(device)

    gan.fit(preprocessed_dataset["X_train_trans"], y=preprocessed_dataset["y_train"].values, 
            condition=True,
            epochs=10,  
            batch_size=64,
            netG_kwargs = {'hidden_layer_sizes': (128,64), 
                            'n_cross_layers': 1,
                            'cat_activation': 'gumbel_softmax',
                            'num_activation': 'none',
                            'condition_num_on_cat': True, 
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

    X_res, y_res = gan.resample(preprocessed_dataset["X_train_trans"], y=preprocessed_dataset["y_train"])
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                                random_state=2020, n_jobs=2)

    clf.fit(X_res, y_res)
    evaluate_model(clf, preprocessed_dataset["X_test_trans"], preprocessed_dataset["y_test"])

    return

def run_all_cwgan(preprocessed_datasets):
    for ds_name in preprocessed_datasets:
        start_time = time.time()
        print("############# " + ds_name + " #############")
        run_wgan_pipeline(preprocessed_datasets[ds_name], preprocessed_datasets[ds_name]["num_cols"], preprocessed_datasets[ds_name]["cat_cols"], preprocessed_datasets[ds_name]["cat_dims"], preprocessed_datasets[ds_name]["prep"])
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Print the elapsed time
        print(f"Time taken for {ds_name} dataset: {elapsed_time:.4f} seconds")

