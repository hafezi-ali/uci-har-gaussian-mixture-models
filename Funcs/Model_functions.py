import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import datetime
from functools import reduce
from scipy.spatial.distance import mahalanobis

# ML imports
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, 
    precision_score, recall_score
)
from sklearn.exceptions import ConvergenceWarning

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

# Ignore all sklearn warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="torch")

# Function to get current timestamp
def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Remove circular import - these functions will need to be defined here or refactored

def confusion_mn(y_real: object, y_pred: object, classes: object, plot: object, title: object) -> object:
    cm = confusion_matrix(y_real, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    if plot:
        # Create heatmap using seaborn
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="d", ax=ax)
        # Add title and axis labels
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.xticks(rotation=30)
        plt.yticks(rotation=60)

        # Show plot
        plt.show()
    return df_cm

# use:
def get_models():
    models = dict()
    models['gn'] = GaussianNB()
    models['sgd'] = SGDClassifier(loss='log', n_jobs=-1)
    models['svm'] = SVC(max_iter=1000, probability=True, C=1, kernel='linear')
    models['lr'] = LogisticRegression(solver='lbfgs', max_iter=1000)
    models['knn'] = KNeighborsClassifier(n_jobs=-1)
    models['dt'] = DecisionTreeClassifier()
    return models


def model_performance_report(y_real, y_pred, plot, title):
    # confusion matrix
    classes = sorted(y_real.unique())
    cm_title = 'Confusion matrix for {}'.format(title)
    df_cm = confusion_mn(y_real, y_pred, classes, plot, cm_title)

    # classification report
    cr = classification_report(y_real, y_pred, digits=4, output_dict=True)
    print("classification report for {}:\n".format(title), cr)
    return df_cm, cr


# ~~
class Ensemble:
    def __init__(self, Best_models_clusters, normalization, scale_re_clusters, candidate_models_cluster,
                 cluster_centers,
                 min_distances, classes, use_ensemble, cluster_model):
        self.diste = use_ensemble['diste']
        self.avge = use_ensemble['avge']
        self.maxe = use_ensemble['maxe']
        self.acce = use_ensemble['acce']
        self.densitye = use_ensemble['densitye']
        self.classes = classes
        self.cluster_model = cluster_model
        self.Best_models_clusters = Best_models_clusters
        self.cluster_centers = cluster_centers
        self.min_distances = min_distances
        self.candidate_models_cluster = candidate_models_cluster
        self.scale_re_clusters = scale_re_clusters
        self.normalization = normalization

    def predict_sample(self, X, Y):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # base classifiers predictions
        if self.normalization:
            X_pred_for_clusters = [self.Best_models_clusters[i].predict_proba(self.scale_re_clusters[i].transform(X))
                                   for i in range(len(self.cluster_centers))]
        else:
            X_pred_for_clusters = [model.predict_proba(X) for model in self.Best_models_clusters]

            if any(np.isnan(arr).any() for arr in X_pred_for_clusters):
                X_pred_for_clusters_with_nan, X_pred_for_clusters = X_pred_for_clusters.copy(), [np.nan_to_num(x, nan=0) for x in X_pred_for_clusters.copy()]
            else:
                X_pred_for_clusters_with_nan = None

        # $
        c = 0
        for l in self.classes:
            for i in range(len(self.cluster_centers)):
                class_i = self.Best_models_clusters[i].classes_
                if l not in class_i:
                    X_pred_for_clusters[i] = np.insert(X_pred_for_clusters[i], c, values=0, axis=1)
            c += 1

        models_outputs = dict()
        # ~~

        dissagreement_between_classifiers = np.mean(
            [np.corrcoef(X_pred_for_clusters[0][i], X_pred_for_clusters[1][i])[0, 1] for i in
             range(X_pred_for_clusters[0].shape[0])]
        )

        # ~~
        # Oracle
        # true_predicted_indexes for each base classifier
        true_predicted_sample_indexes_classifiers = \
            [np.where(Y == np.argmax(X_pred_for_clusters[i], axis=1))[0] for i in range(len(X_pred_for_clusters))]

        oracle_acc = reduce(np.union1d, tuple(true_predicted_sample_indexes_classifiers[i] for i
                                              in range(len(X_pred_for_clusters)))).shape[0] / Y.shape[0]
        # ~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # base classifies outputs combination
        # ---------------------------------------------------------        

        start_time = time.time()
        if self.avge:
            # Average of base classifier outputs
            final_X_array_avge = np.mean(X_pred_for_clusters, axis=0)  # Vectorized average calculation
            models_outputs['avge'] = final_X_array_avge
        else:
          models_outputs['avge'] = None
        
        end_time = time.time()
        print(f"avge calculation: {end_time - start_time:.2f} seconds")        

        start_time = time.time()
        if self.maxe:

            # Convert X_pred_for_clusters to a NumPy array if it's not already
            X_pred_for_clusters = np.array(X_pred_for_clusters)

            # Find the indices of the maximum values along the last axis (features) for each cluster and sample
            max_indices_per_cluster = np.argmax(X_pred_for_clusters, axis=2)

            # Find the maximum values along the last axis (features) for each cluster and sample
            max_values_per_cluster = np.max(X_pred_for_clusters, axis=2)

            # Find the indices of the clusters that have the maximum values for each sample and feature
            max_cluster_indices = np.argmax(max_values_per_cluster, axis=0)

            # Use advanced indexing to get the final indices of the maximum values
            final_X_array_maxe = max_indices_per_cluster[max_cluster_indices, np.arange(max_indices_per_cluster.shape[1])]

            # Encode the predicted labels using pd.get_dummies()
            encoded_labels = pd.get_dummies(final_X_array_maxe, columns=range(len(self.classes)))

            # Add zero columns for absent classes
            absent_classes = list(set(range(len(self.classes))) - set(final_X_array_maxe))
            for class_label in absent_classes:
                encoded_labels[class_label] = 0
            # Sort columns in ascending order
            encoded_labels = encoded_labels.reindex(sorted(encoded_labels.columns), axis=1)
            final_X_array_maxe = encoded_labels

        else:
            final_X_array_maxe = None

        models_outputs['maxe'] = final_X_array_maxe

        end_time = time.time()
        print(f"maxe calculation: {end_time - start_time:.2f} seconds")  

        #diste
        start_time = time.time()
        if self.diste:
            start_time = time.time()
            # Compute distances between each test sample and each cluster center
            distances = np.linalg.norm(self.cluster_centers[:, np.newaxis] - X, axis=2)

            # Filter distances based on the threshold (2 * minimum distances)
            thresholds = 2 * np.array([min_dist[-1] for min_dist in self.min_distances])
            distances_filtered = np.where(distances <= thresholds[:, np.newaxis], distances, np.inf)

            # Ensure at least one non-infinite distance per sample
            min_distances = np.min(distances, axis=0)
            min_indices = np.argmin(distances, axis=0)
            mask_all_inf = np.all(np.isinf(distances_filtered), axis=0)
            
            if np.any(mask_all_inf):
                distances_filtered[:, mask_all_inf] = distances[:, mask_all_inf]
                valid_min_indices = min_indices[mask_all_inf]
                valid_min_distances = min_distances[mask_all_inf]
                
                for idx, valid_min_index in enumerate(valid_min_indices):
                    distances_filtered[valid_min_index, mask_all_inf[idx]] = valid_min_distances[idx]

            # Compute the inverse distances and normalize them to get weights
            inv_distances = 1 / (distances_filtered + 1e-20)
            weights_Xtest_to_clusters = inv_distances / inv_distances.sum(axis=0, keepdims=True)

            # Repeat weights for each class and compute the final weighted output
            weights_Xtest_to_clusters_expanded = np.repeat(weights_Xtest_to_clusters[:, :, np.newaxis], len(self.classes), axis=2)
            final_X_diste = np.sum(weights_Xtest_to_clusters_expanded * np.array(X_pred_for_clusters), axis=0)

            final_X_array_diste = final_X_diste
            final_X_tensor_diste = torch.tensor(final_X_array_diste)

        else:
            final_X_array_diste = None

        models_outputs['diste'] = final_X_array_diste

        end_time = time.time()
        print(f"diste calculation: {end_time - start_time:.2f} seconds")  

        start_time = time.time()
        if self.densitye:
            # Predict cluster assignment probabilities for the test samples
            cluster_assignments_proba = self.cluster_model.predict_proba(X)
            
            # Ensure cluster_assignments_proba is of shape (num_samples, num_clusters)
            num_samples, num_clusters = cluster_assignments_proba.shape
            num_classes = len(self.classes)
            
            # Expand cluster probabilities to match the number of classes
            cluster_assignments_proba_expanded = np.repeat(cluster_assignments_proba[:, :, np.newaxis], num_classes, axis=2)
            
            # Ensure X_pred_for_clusters is correctly shaped (num_clusters, num_samples, num_classes)
            X_pred_for_clusters = np.array(X_pred_for_clusters)
            if X_pred_for_clusters.shape != (num_clusters, num_samples, num_classes):
                raise ValueError(f"X_pred_for_clusters should be of shape ({num_clusters}, {num_samples}, {num_classes})")

            # Compute the final weighted output by summing over the cluster assignments and predictions
            final_X_densitye = np.sum(cluster_assignments_proba_expanded.transpose(1, 0, 2) * X_pred_for_clusters, axis=0)
            
            # Convert the result to a torch tensor
            final_X_array_densitye = final_X_densitye
            final_X_tensor_densitye = torch.tensor(final_X_array_densitye)
        else:
            final_X_array_densitye = None

        models_outputs['densitye'] = final_X_array_densitye

        end_time = time.time()
        print(f"densitye calculation: {end_time - start_time:.2f} seconds")  


        if self.acce:
            # acc weights
            # val_model based model
            val1 = self.candidate_models_cluster[0]['svm']['best_score_']
            val2 = self.candidate_models_cluster[1]['svm']['best_score_']
            # val3 = candidate_models_cluster[2]['svm']['best_score_']
            sum_vals = val1 + val2
            weights_val_models = [val1 / sum_vals, val2 / sum_vals]

            weights_val_models_Xtest_filtered = list()
            for i in range(X.shape[0]):
                distances_xtest = [np.linalg.norm(np.abs(self.cluster_centers[x] - X[i]))
                                   for x in range(len(self.cluster_centers))]
                val_models_xtest_filtered = [
                    weights_val_models[j] if distances_xtest[j] <= 2 * self.min_distances[j][-1]
                    else 0 for j in range(len(self.cluster_centers))]

                if np.all(np.array(val_models_xtest_filtered) == 0):
                    min_value = min(distances_xtest)
                    min_index = distances_xtest.index(min_value)
                    val_models_xtest_filtered[min_index] = weights_val_models[min_index]

                weights_val_models_Xtest_filtered.append(val_models_xtest_filtered)

            weights_val_models_Xtest_filtered_normalized = np.array(
                [x / (np.sum(x)) for x in weights_val_models_Xtest_filtered])

            # val acc based model
            final_X_val_acc = list()
            for i in range(len(X)):
                final_X_val_acc.append(
                    (X_pred_for_clusters[0][i, :] * weights_val_models_Xtest_filtered_normalized[i][0] +
                     X_pred_for_clusters[1][i, :] * weights_val_models_Xtest_filtered_normalized[i][1]))

            final_X_array_val_acc = np.array(final_X_val_acc)
            finall_X_tensr_acce = torch.tensor(final_X_array_val_acc)
        else:
            final_X_array_val_acc = None
            weights_val_models_Xtest_filtered_normalized = None
        models_outputs['acce'] = final_X_array_val_acc
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Results = {'models_outputs': models_outputs,
                   'oracle_acc': oracle_acc,
                   'distances_xtest_filtered': distances_filtered,
                   'weights_Xtest_dist': weights_Xtest_to_clusters,
                   'weights_Xtest_val': weights_val_models_Xtest_filtered_normalized,
                   'dissagreement_between_individual_classifiers': dissagreement_between_classifiers,
                   'X_pred_for_clusters_with_nan': X_pred_for_clusters_with_nan,
                   'X_pred_for_clusters': X_pred_for_clusters
                   }
        return Results


class MainModel(nn.Module):
    def __init__(self, Best_models_clusters, normalization, scale_re_clusters, candidate_models_cluster,
                 cluster_centers,
                 min_distances, classes, use_ensemble, cluster_model, meta_learner):
        super().__init__()
        self.Best_models_clusters = Best_models_clusters
        self.classes = classes
        self.candidate_models_cluster = candidate_models_cluster
        self.cluster_centers = cluster_centers
        self.min_distances = min_distances
        self.use_ensemble = use_ensemble
        self.cluster_model = cluster_model
        self.meta_learner = meta_learner
        self.scale_re_clusters = scale_re_clusters
        self.normalization = normalization
        # distance based ensemble learning
        self.EModel = Ensemble(Best_models_clusters=Best_models_clusters,
                               normalization=normalization,
                               scale_re_clusters=scale_re_clusters,
                               candidate_models_cluster=candidate_models_cluster, cluster_centers=cluster_centers,
                               min_distances=min_distances, classes=classes, use_ensemble=use_ensemble,
                               cluster_model=cluster_model)

    def forward(self, x, y=None):
        # Pass the input through the ensemble model
        # new
        if y is None:
            # For inference mode, create a dummy y of the right shape
            y = np.zeros(x.shape[0])
            
        self.EModel.Best_models_clusters = [m.set_params(loss='log_loss') if isinstance(m, SGDClassifier) else m for m in self.EModel.Best_models_clusters]
        
        # cpu
        ensemble_results = self.EModel.predict_sample(x.cpu().numpy(), y)

        ensemble_outputs = dict()
        if 'main' in self.use_ensemble.keys():
            use_ensemble_list = list(self.use_ensemble.keys())[0:-1]
        else:
            use_ensemble_list = list(self.use_ensemble.keys())
        for emodel in use_ensemble_list:
            if self.use_ensemble[emodel]:
                emodel_output = ensemble_results['models_outputs'][emodel]

                if emodel == 'maxe':
                    emodel_output.replace(to_replace=0, value=False, inplace=True)
                emodel_outputs_tensor = torch.tensor(np.array(emodel_output)).float()
                ensemble_outputs[emodel] = emodel_outputs_tensor

        if self.meta_learner:
            main_gate_outputs = self.gates[0](x.unsqueeze(1))
            # Combine the ensemble outputs
            # use:
            main_outputs = torch.matmul(
                main_gate_outputs.unsqueeze(1),
                torch.transpose(torch.stack(list(ensemble_outputs.values()), dim=2), 1, 2)).squeeze(1)

            ensemble_outputs['main'] = main_outputs
            Results = {'main': main_outputs, 'main_gate': main_gate_outputs, 'ensemble_outputs': ensemble_outputs,
                       'dissagreement_between_individual_classifiers':
                           ensemble_results['dissagreement_between_individual_classifiers']}
        else:
            Results = {'ensemble_outputs': ensemble_outputs,
                       'dissagreement_between_individual_classifiers':
                           ensemble_results['dissagreement_between_individual_classifiers'],
                       'oracle_acc': ensemble_results['oracle_acc'],
                       'ensemble_results': ensemble_results}
        return Results

# ~~

def train(model, optimizer, criterion, train_loader, use_ensemble):
    model.train()
    running_loss = 0.0
    forward_train_outputs_per_batch = list()
    models_scores_train_results_per_batch = list()
    for i, (inputs, labels) in enumerate(train_loader, 0):
        if inputs.shape[0] < 2:
            break
        optimizer.zero_grad()
        forward_train_outputs = model(inputs, labels)
        models_scores_train_results = dict()
        # sub_models = ['main', 'diste', 'maxe', 'avge']
        avg_methods = ['macro', 'weighted']
        use_ensemble['main'] = True
        for sub_model in use_ensemble.keys():
            if use_ensemble[sub_model]:
                for method in avg_methods:
                    # Calculate loss F1 score, precision, and recall for main model
                    labels_np = labels.detach().numpy()
                    y_pred = forward_train_outputs['ensemble_outputs'][sub_model]
                    y_pred_np = y_pred.detach().numpy()
                    y_pred_np = pd.DataFrame([np.argmax(x) for x in y_pred_np])

                    f1 = f1_score(labels_np, y_pred_np, average=method)
                    precision = precision_score(labels_np, y_pred_np, average=method, zero_division=0)
                    recall = recall_score(labels_np, y_pred_np, average=method)

                    metrics_results = {'f1_score': f1, 'precision': precision, 'recall': recall}
                    models_scores_train_results[sub_model + '_' + method] = metrics_results

        final_train_main_model_loss_scaler = criterion(forward_train_outputs['ensemble_outputs']['main'], labels)
        final_train_main_model_loss_scaler.backward()
        optimizer.step()
        running_loss += final_train_main_model_loss_scaler.item()

        forward_train_outputs_per_batch.append(forward_train_outputs)
        models_scores_train_results_per_batch.append(models_scores_train_results)

    return running_loss / len(train_loader), models_scores_train_results_per_batch, forward_train_outputs_per_batch


# ~~

# Define the validation loop
def validate(model, criterion, val_loader, use_ensemble):
    model.eval()
    running_loss = 0.0
    forward_eval_outputs_per_batch = list()
    models_scores_eval_results_per_batch = list()
    with torch.no_grad():
        for inputs, labels in val_loader:
            forward_eval_outputs = model(inputs, labels)
            models_scores_eval_results = dict()
            avg_methods = ['macro', 'weighted']
            use_ensemble['main'] = True
            for sub_model in use_ensemble.keys():
                if use_ensemble[sub_model]:
                    for method in avg_methods:
                        # Calculate loss F1 score, precision, and recall for main model
                        labels_np = labels.detach().numpy()
                        y_pred = forward_eval_outputs['ensemble_outputs'][sub_model]
                        y_pred_np = y_pred.detach().numpy()
                        y_pred_np = pd.DataFrame([np.argmax(x) for x in y_pred_np])

                        f1 = f1_score(labels_np, y_pred_np, average=method)
                        precision = precision_score(labels_np, y_pred_np, average=method, zero_division=0)
                        recall = recall_score(labels_np, y_pred_np, average=method)

                        metrics_results = {'f1_score': f1, 'precision': precision, 'recall': recall}
                        models_scores_eval_results[sub_model + '_' + method] = metrics_results

            final_eval_main_model_loss_scaler = criterion(forward_eval_outputs['ensemble_outputs']['main'], labels)

            running_loss += final_eval_main_model_loss_scaler.item()
            forward_eval_outputs_per_batch.append(forward_eval_outputs)
            models_scores_eval_results_per_batch.append(models_scores_eval_results)

    return running_loss / len(val_loader), models_scores_eval_results_per_batch, forward_eval_outputs_per_batch


def train_main_model(model, optimizer, criterion_main_model, n_epoch, preprocessing_res, use_ensemble):
    print(f"[{get_timestamp()}] Starting model training for {n_epoch} epochs")
    
    # Train the model
    weights_list = list()
    models_scores_results_per_epoch = list()
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience = 5
    n_epochs_no_improvement = 0
    
    for epoch in tqdm(range(n_epoch), position=0, leave=True, unit='epoch', colour='green', desc="Training epochs"):
        epoch_start = time.time()
        print(f"\n[{get_timestamp()}] Epoch {epoch+1}/{n_epoch}")
        
        train_start = time.time()
        train_loss, models_scores_train_results_per_batch, forward_train_outputs_per_batch = \
            train(model, optimizer, criterion_main_model, preprocessing_res['train_loader'], use_ensemble=use_ensemble)
        train_end = time.time()
        
        print(f"[{get_timestamp()}] Training completed in {train_end - train_start:.2f} seconds")

        models_train_scores_results_per_batch = \
            {'models_scores': models_scores_train_results_per_batch,
             'forward_train_outputs': forward_train_outputs_per_batch,
             'loss': train_loss}

        val_start = time.time()
        val_loss, models_scores_eval_results_per_batch, forward_eval_outputs_per_batch = \
            validate(model, criterion_main_model, preprocessing_res['valid_loader'], use_ensemble=use_ensemble)
        val_end = time.time()
        
        print(f"[{get_timestamp()}] Validation completed in {val_end - val_start:.2f} seconds")
        
        models_eval_scores_results_per_batch = {
            'models_scores': models_scores_eval_results_per_batch,
            'forward_eval_outputs': forward_eval_outputs_per_batch,
            'loss': val_loss}

        models_scores_results_per_batch = {'train': models_train_scores_results_per_batch,
                                           'eval': models_eval_scores_results_per_batch}

        models_scores_results_per_epoch.append(models_scores_results_per_batch)

        weights_list.append(copy.deepcopy(model.state_dict()))

        epoch_end = time.time()
        print(f"[{get_timestamp()}] Epoch {epoch+1} completed in {epoch_end - epoch_start:.2f} seconds")
        print(f"Training loss = {train_loss:.4f}, Validation loss = {val_loss:.4f}")

        # Check if validation loss improved
        if np.round(val_loss, 3) < np.round(best_val_loss, 3):
            best_val_loss = np.round(val_loss, 3)
            n_epochs_no_improvement = 0
            print(f"[{get_timestamp()}] Validation loss improved to {best_val_loss:.4f}")
        else:
            n_epochs_no_improvement += 1
            print(f"[{get_timestamp()}] No improvement for {n_epochs_no_improvement} epochs")

        # Stop training if validation loss does not improve for 'patience' epochs
        if n_epochs_no_improvement == patience:
            print(f"[{get_timestamp()}] Validation loss did not improve for {patience} epochs. Stopping early.")
            break
    
    print(f"[{get_timestamp()}] Training completed")
    print(f"Best validation loss: {best_val_loss:.4f}")

    weights = copy.deepcopy(model.state_dict())
    Results = {'model': model, 'weights_list': weights_list,
               'models_scores_results_per_epoch': models_scores_results_per_epoch}
    return Results

# ______________________________________________
# grid search for base_models
def f1_tuned_model(cs, csy, model, model_name, param_grid, normalization, n_jobs=-1):
    if normalization:
        pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True), model)
        param_grid = {'{}__{}'.format(model_name, param_key): param_grid[param_key] for param_key in param_grid.keys()}
        grid = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1_macro', cv=4, return_train_score=True, n_jobs=n_jobs)
    else:
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_macro', cv=4, return_train_score=True, n_jobs=n_jobs)
    grid.fit(cs, csy)
    best_index = grid.best_index_
    train_best_grid = grid.cv_results_['mean_train_score'][best_index]
    val_best_grid = grid.cv_results_['mean_test_score'][best_index]
    return grid, train_best_grid, val_best_grid


def grid_train_base_models(models, cluster_samples, cluster_samples_lables, model_param_grids, normalization, n_jobs=-1):
    best_param_models_samples = list()
    print(f"[{get_timestamp()}] Starting grid search for {len(cluster_samples)} clusters")
    
    for n_c in tqdm(range(len(cluster_samples)), position=0, leave=True, unit='cluster', colour='green', 
                   desc="Grid searching clusters"):
        start_time = time.time()
        print(f"\n[{get_timestamp()}] Processing cluster {n_c+1}/{len(cluster_samples)}")
        
        best_param_models = dict()
        for model, param_grid in model_param_grids.items():
            model_start = time.time()
            print(f"  [{get_timestamp()}] Grid search for {model} model")
            
            grid_model_cluster, train_best_grid_model_cluster, val_best_grid_model_cluster = f1_tuned_model(
                cs=cluster_samples[n_c],
                csy=cluster_samples_lables[n_c],
                model=models[model],
                model_name=model,
                param_grid=param_grid,
                normalization=normalization,
                n_jobs=n_jobs
            )
            best_param_models[model] = {'grid_' + model + '_cluster': grid_model_cluster,
                                        'train_best_grid_' + model + '_cluster': train_best_grid_model_cluster,
                                        'val_best_grid_' + model + '_cluster': val_best_grid_model_cluster}
            
            model_end = time.time()
            print(f"  [{get_timestamp()}] Completed {model} in {model_end - model_start:.2f} seconds")
            print(f"  Best score: {val_best_grid_model_cluster:.4f}")

        best_param_models_samples.append(best_param_models)
        
        end_time = time.time()
        print(f"[{get_timestamp()}] Completed cluster {n_c+1} in {end_time - start_time:.2f} seconds")
    
    print(f"[{get_timestamp()}] Grid search completed for all clusters")
    return best_param_models_samples


# ______________________________________________


# Define a function to sort a list of dictionaries by best_score_
def sort_by_score(lst):
    # Create a new list to store the sorted dictionaries
    sorted_lst = []
    # Loop through each dictionary in the original list
    for d in lst:
        # Convert the dictionary into a list of tuples, where each tuple contains the classifier name and the
        # best_score_
        items = list(d.items())
        # Sort the list of tuples by the best_score_ in descending order
        items.sort(key=lambda x: x[1]['best_score_'], reverse=True)
        # Convert the sorted list of tuples back into a dictionary and append it to the new list
        sorted_d = dict(items)
        sorted_lst.append(sorted_d)
    # Return the new list
    return sorted_lst


param_grid_svm = {
    'class_weight': [None, 'balanced'],
    'C': [0.1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 10],
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'decision_function_shape': ['ovo', 'ovr'],
    'tol': [1e-3],
}

# Gaussian Naive Bayes (GNB)
param_grid_gn = {
    'priors': [None, [0.1, 0.9], [0.3, 0.7]],
    'var_smoothing': [1e-08, 1e-09, 1e-10],
}

param_grid_sgd = {
    'class_weight': ['balanced'],
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'n_jobs': [-1],
}

param_grid_lr = {
    'class_weight': ['balanced'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs'],
    'n_jobs': [-1],
}

# K-Nearest Neighbors (KNN)
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13],
    'weights': ['uniform', 'distance', 'custom'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'n_jobs': [-1],
}

param_grid_dt = {
    'class_weight': ['balanced'],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

base_models_param_grids = {'param_grid_dt': param_grid_dt, 'param_grid_lr': param_grid_lr,
                           'param_grid_knn': param_grid_knn,
                           'param_grid_svm': param_grid_svm, 'param_grid_sgd': param_grid_sgd,
                           'param_grid_gn': param_grid_gn}
