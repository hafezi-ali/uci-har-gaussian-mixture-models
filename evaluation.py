
print('modified code version_')
# %%

import copy
# Import necessary libs
import os
import pickle
import numpy as np



os.chdir('/content/drive/MyDrive/Thp')
# Set the PYTHONPATH environment variable to include the directory containing your modules
os.environ['PYTHONPATH'] = '/content/drive/MyDrive/Thp'

from Funcs.Functions import *
from Funcs.Model_functions import *
import Funcs
from Lib.lib import *
# %%

uci_data_handle_new = pd.read_csv('Data/UCI_HAR_Dataset/data_uci_handled.csv', index_col=0)
uci_data_handle_new['Activity'] = uci_data_handle_new['Activity'] + 1
database = {'uci': uci_data_handle_new}
# %%

X = database['uci'].drop(['Activity', 'ActivityName', 'subject'], axis=1)
y = database['uci']['ActivityName']
# %%

# df_single_resampling_res_grid = pd.read_pickle('Results/df_single_resampling_res_grid_main_full.pkl')
# single_preprocessing_res_grid = pd.read_pickle('Results/single_preprocessing_res_grid_main.pkl')

df_single_resampling_res_grid = pd.read_pickle('Results/df_single_resampling_res_grid_last2.pkl')
single_preprocessing_res_grid = pd.read_pickle('Results/single_preprocessing_res_grid_last2.pkl')
# %%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
models = get_models()
models.pop('gn')
models['sgd'] = models['sgd'].set_params(loss='log_loss')
# %%
phi_list = np.arange(0.1, 1, 0.1)
n_c_list = np.arange(2, 4)
print('n_c_list', n_c_list)
n_repeats = 4
n_splits = 5
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

with tqdm(total=len(n_c_list) * len(phi_list) * n_splits * n_repeats,
          desc="Outer Loop", position=0, leave=True, unit='epoch', colour='green') as outer_pbar:
    df_single_best_res_nested = pd.DataFrame(columns=['phi', 'n_c', 'best_res_nested'])
    for n_c in n_c_list:
        for phi in phi_list:
            start_main_time = time.time()
            outer_pbar.update(n_splits * n_repeats)
            scores = []
            print(
                '------------------------------------- n_c: {} & phi: {} -------------------------------------'.format(
                    n_c, phi))
            final_outs = {}
            final_res = {}
            data_split_y = dict()
            # data_split = dict()
            # single_preprocessing_reses = dict()
            # single_resampling_reses = dict()
            # main_model_architectures_dict_nested = dict()
            # signed_base_classifires_dict_nested = dict()
            # Ordered_centers_indexes = dict()

            with tqdm(total=n_splits * n_repeats,
                      desc="Inner Loop (n_c={}, phi={})".format(n_c, phi),
                      position=1, leave=False, unit='fold', colour='blue') as inner_pbar:
                for k, (train_ix, test_ix) in enumerate(cv.split(X)):
                    start_k_time = time.time()

                    print(
                        '-------------------------------------k value: {} -------------------------------------'.format(
                            k))

                    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
                    # data_per_ix = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
                    # data_split[k] = data_per_ix
                    data_split_y[k] = {'y_train': y_train, 'y_test': y_test}
                    

                    # ~~~~~~~~~
                    n_components = single_preprocessing_res_grid['X_train'].shape[1]
                    random_state_pca = None
                    # random_state_moe_dist_split = None

                    single_preprocessing_res = Funcs.Functions.single_preprocessing(X_train=X_train, X_test=X_test,
                                                                                    y_train=y_train,
                                                                                    n_components=n_components,
                                                                                    random_state_pca=random_state_pca,
                                                                                    )
                    # single_preprocessing_reses[k] = single_preprocessing_res

                    # ******************************** Single models *******************************

                    start_time = time.time()

                    df_cm_single_models = {}
                    cr_single_models = {}
                    for single_model in models:
                        model = copy.deepcopy(models[single_model])
                        model.fit(single_preprocessing_res['X_train'], y_train)
                        y_pred_test_single_model = model.predict(single_preprocessing_res['X_test'])
                        df_cm_single_model, cr_single_model = model_performance_report(
                          y_real=y_test, y_pred=y_pred_test_single_model, plot=False,
                           title=f'model_single_{single_model}')

                        # df_cm_single_model, cr_single_model = \
                        #     model_performance_report(y_real=y_test,
                        #                              y_pred=y_pred_test_single_model,
                        #                              plot=False,
                        #                              title='model_{}'.format('single_' + single_model))

                        df_cm_single_models[f'single_{single_model}'] = df_cm_single_model
                        cr_single_models[f'single_{single_model}'] = cr_single_model

                    end_time = time.time()
                    print(f"Single models calculation: {end_time - start_time:.2f} seconds")
                    # ******************************** Ensemble models *******************************

                    # clustering on data
                    # random_state_clustering = None
                    # random_state_cluster_samples_shuffling = 42
                    # distance_metrics = 'Euclidean'
                    # gmm = GaussianMixture()
                    # gmm_params = {
                    #     'n_components': n_c,
                    #     'random_state': random_state_clustering,
                    # }
                    # clusterig_method = gmm.set_params(**gmm_params)

                    start_time = time.time()

                    gmm_params = {'n_components': n_c, 'random_state': None}
                    clusterig_method = GaussianMixture(**gmm_params)
                    # clusterig_method = KMeans(n_clusters=n_c, random_state=random_state_clustering)

                    # ~~~~~~~~~~
                    # resample data for each cluster
                    single_resampling_res = resampling(clusterig_method=clusterig_method,
                                                       X_train_dist=single_preprocessing_res['X_train'],
                                                       y_train_dist=y_train, phi=phi,
                                                       random_state_cluster_samples_shuffling=42,
                                                       distance_metrics='Euclidean')
                    
                    end_time = time.time()
                    print(f"single_resampling_res calculation: {end_time - start_time:.2f} seconds")
                    # single_resampling_reses[k] = single_resampling_res

                    # __________________________________________ Training  __________________________________________

                    start_time = time.time()
                    df_out = df_single_resampling_res_grid[
                        (df_single_resampling_res_grid.phi == phi) & (df_single_resampling_res_grid.n_c == n_c)]

                    centers_1 = np.array(single_resampling_res['cluster_centers'])
                    centers_2 = np.array(df_out['single_resampling_res'].iloc[0]['cluster_centers'])

                    # Calculate the differences between each pair of centers (using broadcasting)
                    differences = centers_1[:, np.newaxis, :] - centers_2[np.newaxis, :, :]

                    # Calculate the Euclidean distances
                    distances = np.linalg.norm(differences, axis=2)

                    # Find the index of the closest center in centers_2 for each center in centers_1
                    ordered_centers_indexes = np.argmin(distances, axis=1)

                    # signed_base_classifires[k] = [df_out['best_models_clusters'].iloc[0][i] for i in ordered_centers_indexes]
                    # Ordered_centers_indexes[k] = ordered_centers_indexes

                    # new
                    signed_base_classifires_dict = dict()
                    for single_model in models:
                        signed_base_classifires = [df_out['best_models_clusters'].iloc[0][single_model][i] for i in
                                                   ordered_centers_indexes]
                        signed_base_classifires_dict[single_model] = signed_base_classifires

                    # signed_base_classifires_dict_nested[k] = signed_base_classifires_dict

                    for l in range(n_c):
                        x_re_cluster = single_resampling_res['rm_cluster_samples'][l]
                        y_re_clusters = single_resampling_res['rm_cluster_samples_labels'][l]
                        for single_model_name in models:
                            # Create a copy of the original model for each cluster
                            single_model = copy.deepcopy(signed_base_classifires_dict[single_model_name][l])
                            signed_base_classifires_dict[single_model_name][l] = single_model.fit(x_re_cluster,
                                                                                                  y_re_clusters)
                    end_time = time.time()
                    print(f"signed_base_classifires calculation: {end_time - start_time:.2f} seconds")
                    # single_resampling_reses[k] = single_resampling_res
                    # ~~~~~~~~~~

                    input_size = single_preprocessing_res['X_train'].shape[1]
                    use_ensemble = {'diste': True, 'avge': True, 'acce': False, 'maxe': True, 'densitye': True}
                    num_models = len(use_ensemble)
                    classes = sorted(y_test.unique())

                    # ~~~~~~~~~~
                    meta_learner_use = False

                    start_time_out = time.time()
                    main_model_architecture_dict = dict()
                    for single_model in models:

                        main_model_architecture = MainModel(
                            Best_models_clusters=signed_base_classifires_dict[single_model],
                            normalization=False,
                            scale_re_clusters=None,
                            candidate_models_cluster=df_out['sorted_candidate_models_cluster_grid'],
                            cluster_centers=single_resampling_res['cluster_centers'],
                            min_distances=single_resampling_res['min_distances'],
                            classes=classes, use_ensemble=use_ensemble,
                            cluster_model=single_resampling_res['cluster_model'],
                            meta_learner=meta_learner_use)
                        main_model_architecture_dict[single_model] = main_model_architecture
                    
                    end_time_out = time.time()
                    print(f"main_model_architecture calculation: {end_time_out - start_time_out:.2f} seconds")

                    # main_model_architectures_dict_nested[k] = main_model_architecture_dict

                    # Encode your labels
                    le = LabelEncoder()
                    encoded_y_train = le.fit_transform(y)
                    # ~~~~~~~~~~

                    
                    start_time_out = time.time()
                    final_out_dict = dict()
                    output_arrays_dict = dict()
                    decoded_final_xtest_model_array_normalized_list_dict = dict()
                    df_cm_models_dict = dict()
                    cr_models_dict = dict()
                    for single_model in models:
                        # print(
                        #     '------------------------------------- single_model: {} -------------------------------------'.format(
                        #         single_model))

                        start_time = time.time()
                        final_out = main_model_architecture_dict[single_model](
                            torch.Tensor(single_preprocessing_res['X_test']), le.fit_transform(y_test))
                        final_out_dict[single_model] = final_out
                        end_time = time.time()
                        print(f"final_out calculation for {single_model} model: {end_time - start_time:.2f} seconds")

                        output_arrays = dict()
                        for sub_model in use_ensemble.keys():
                            if use_ensemble[sub_model]:
                                output_arrays[sub_model] = final_out['ensemble_outputs'][sub_model].detach().numpy()

                        output_arrays_dict[single_model] = output_arrays
                        # ~~~~~~~~~~~

                        decoded_final_xtest_model_array_normalized_list = dict()
                        df_cm_models = dict()
                        cr_models = dict()
                        for sub_model in use_ensemble.keys():
                            if use_ensemble[sub_model]:
                                final_xtest_model_array_normalized = \
                                    pd.DataFrame([np.argmax(x) for x in output_arrays[sub_model]])
                                # Decode the labels for the test data
                                decoded_final_xtest_model_array_normalized = \
                                    le.inverse_transform(final_xtest_model_array_normalized.values.ravel())
                                decoded_final_xtest_model_array_normalized_list[
                                    sub_model] = decoded_final_xtest_model_array_normalized

                                df_cm_model, cr_model = \
                                    model_performance_report(y_real=y_test,
                                                             y_pred=decoded_final_xtest_model_array_normalized,
                                                             plot=False,
                                                             title='model_{}'.format(sub_model))

                                df_cm_models[sub_model] = df_cm_model
                                cr_models[sub_model] = cr_model

                        decoded_final_xtest_model_array_normalized_list_dict[
                            single_model] = decoded_final_xtest_model_array_normalized_list
                        df_cm_models_dict[single_model] = df_cm_models
                        cr_models_dict[single_model] = cr_models

                    # oracle_accs.append(final_out_dict['oracle_acc'])
                    # final_out['ensemble_outputs'][sub_model]
                    final_outs[k] = decoded_final_xtest_model_array_normalized_list_dict
                    final_res[k] = {'cm': df_cm_models_dict, 'cr': cr_models_dict, 'cm_single': df_cm_single_models,
                                    'cr_single': cr_single_models, 'final_out_dict': final_out_dict,
                                    # 'dissagreement_between_individual_classifiers':final_out_dict['dissagreement_between_individual_classifiers'],
                                    }
                    end_time_out = time.time()
                    print(f"results calculation: {end_time_out - start_time_out:.2f} seconds")
                    # ************************************************************************************************
                    # ~~~~~~~~~~~~~~~~~

                    # Update the inner progress bar
                    inner_pbar.update(1)
                    k += 1
                    end_k_time = time.time()
                    print(f"fold calculation: {end_k_time - start_k_time:.2f} seconds") 
            best_res = {'final_outs': final_outs, 'final_res': final_res,
                        # 'data_split': data_split,
                        # 'single_preprocessing_reses': single_eprocessing_reses,
                        # 'single_resampling_reses': single_resampling_reses,
                        # 'main_model_architectures': main_model_architectures_dict_nested,
                        # 'train_main_model_reses': train_main_model_reses,
                        # 'Ordered_centers_indexes': Ordered_centers_indexes,
                        # 'signed_base_classifires': signed_base_classifires_dict
                        }

            new_row = pd.DataFrame({'phi': [phi], 'n_c': [n_c], 'best_res_nested': [best_res]})
            df_single_best_res_nested = pd.concat([df_single_best_res_nested, new_row], ignore_index=True)
            end_main_time = time.time()
            print(f"\033[32mmain calculation: {end_main_time - start_main_time:.2f} seconds\033[0m")
 

# Print the resulting dataframe
print('------------------------- succesfully df_single_best_res_nested -------------------------')
# %%

with open('Results/finall_res2.pkl', 'wb') as f:
    pickle.dump(df_single_best_res_nested, f)

print('------------------------- succesfully saved df_single_best_res_nested -------------------------')
# %%
