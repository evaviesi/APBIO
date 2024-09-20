#!/usr/bin/env python
# -*- coding: utf-8 -*-

# define imports 
import os
import random
import pickle
import numpy as np
import pandas as pd

# import scikit-learn modules 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# import plotting libraries 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# set seed 
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
 

def eval_cv_scores(datasets_list, models_list, seed=SEED):

    """This function computes and plots the average scores 
    from nested cross validation results. 

    :param datasets_list: a list of dataset names 
    :param models_list: a list of model names
    :param seed: seed for reproducibility 
    :return: a dataframe of average cv scores 
    """

    # seed the global RNG 
    random.seed(seed)
    # seed the global NumPy RNG
    np.random.seed(seed)

    # get current directory path
    curr_dir = os.path.dirname(__file__)

    cv_scores = {}
    avg_cv_scores = pd.DataFrame()

    # iterate through datasets and models 
    for dataset in datasets_list: 
        for model in models_list:
            
            # read cv scores
            filename = curr_dir + '/models/' + model + '.' + dataset + '.pkl'
            if os.path.exists(filename):
                with open(filename,'rb') as f:
                    cv_scores[model] = pickle.load(f)
                f.close()

            # create a dataframe of the computed metrics
            scores = pd.DataFrame(cv_scores[model]).iloc[:,2:].mean().round(4)
            df = pd.DataFrame({'Score' : scores}).assign(Model=model.upper()) \
                                                 .assign(Dataset=dataset)
            avg_cv_scores = pd.concat([avg_cv_scores, df])

    # set index as column
    avg_cv_scores.reset_index(inplace=True)
    avg_cv_scores = avg_cv_scores.rename(columns = {'index' : 'Metric'})

    # figure
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    plt.subplots_adjust(hspace=1)
    plt.subplots_adjust(wspace=0.2)

    # plot results for each model
    for model, ax in zip(models_list, axes.ravel()):
        sns.pointplot(ax=ax, 
                      data=avg_cv_scores[avg_cv_scores.Model==model.upper()], 
                      x='Metric', y='Score', hue='Dataset', 
                      errorbar='se', palette=sns.set_palette("husl", 5))
        # lines
        plt.setp(ax.lines, alpha=.7) 
        # legend
        plt.setp(ax.get_legend().get_title(), fontsize=18)
        sns.move_legend(ax, "lower left", fontsize=18)
        
        # title
        ax.set_title(model.upper(), fontsize=18)

        # x-axis
        ax.set_xlabel('Metric', fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=18) 

        # y-axis
        ax.set_ylabel('Average CV score', fontsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        ax.set_ylim(min(avg_cv_scores.Score), max(avg_cv_scores.Score))

        # grid
        ax.grid(visible=True)

    return avg_cv_scores
 

def sel_model(X, y, dataset_name, models_list, cv_scores, 
              output_path, seed=SEED):

    """This function selects and re-fits the best estimator based on 
    cross validation results. 

    :param X: a numpy array of (n_samples, n_features)
    :param y: a dataframe of class labels having (inchikey, uniprot) as index  
    :param dataset_name: a string indicating the dataset name
    :param models_list: a list of model names
    :param cv_scores: a dataframe of cross validation scores 
    :param output_path: path to save model 
    :param seed: seed for reproducibility 
    :return: a scikit-learn estimator
    """

    # seed the global RNG 
    random.seed(seed)
    # seed the global NumPy RNG
    np.random.seed(seed)

    # dataset shape 
    print("X: ", X.shape, "\ny: ", y.shape)

    # save average cv score for each model 
    avg_scores = {} 
    for model in models_list:
        avg_scores[model] = np.round(
            cv_scores[(cv_scores.Model==model.upper()) & \
                      (cv_scores.Dataset==dataset_name)].Score.mean(), 4)
    print("Average scores: ", avg_scores)

    # select model with maximum average score         
    selected_model = max(avg_scores, key=avg_scores.get)
    print("Model: " + selected_model.upper(), "\nMax score: " + 
          str(avg_scores[selected_model]))

    # define grid of parameters to be tuned 
    C_range = [0.01, 0.1, 1.0, 10]                                 # LR
    n_neighbors_range = [5, 15, 25, 35]                            # KNN                 
    n_estimators_range = [100, 200, 300, 400]                      # RF 
    layer_sizes_range = [(64,), (128,), (256,), (512,)]            # MLP 

    # define type of cross-validation 
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)

    # initialize models 
    if selected_model == 'lr':
        estimator = LogisticRegression(max_iter=1000, random_state=seed)   
        param_grid = dict(C=C_range)

    if selected_model == 'knn':
        estimator = KNeighborsClassifier(metric='cosine')
        param_grid = dict(n_neighbors=n_neighbors_range)

    if selected_model == 'rf':
        estimator = RandomForestClassifier(random_state=seed)
        param_grid = dict(n_estimators=n_estimators_range)

    if selected_model == 'mlp':
        estimator = MLPClassifier(max_iter=1000, random_state=seed)
        param_grid = dict(hidden_layer_sizes=layer_sizes_range)

    # fit the estimator via grid search cross-validation 
    grid_cv = GridSearchCV(estimator, param_grid=param_grid, 
                           scoring='balanced_accuracy',
                           cv = rskf, refit=True).fit(X, y)
    
    # save model 
    estimator = grid_cv.best_estimator_
    pickle.dump(estimator, open(output_path + '/' + selected_model + '.' + 
                                dataset_name + '.pkl', 'wb')) 

    return estimator


def eval_sampling(dataset_name, model, output_path, seed=SEED):

    """This function evaluates ocsvm and random negative sampling strategies
    by calculating the area under the precision-recall curve.

    :param dataset_name: a string indicating the dataset name
    :param model: a scikit-learn estimator
    :param output_path: path to save figure
    :param seed: seed for reproducibility 
    :return: a dictionary of recall, precision, and average precision values
    """
    
    # seed the global RNG 
    random.seed(seed)
    # seed the global NumPy RNG
    np.random.seed(seed)

    # get current directory path
    curr_dir = os.path.dirname(__file__)
    # define datasets path
    datasets_path = "../../cti_datasets/AP_CTIs"

    ratio = [10, 20, 100]
    type = ["sampled", "random"]
    pr_res = {}
        
    for r in ratio:
        # load dataset
        res = {}
        for t in type:
            if t == "sampled":
                with open(curr_dir + "/" + datasets_path + "/sampled/" + 
                          dataset_name + "_ratio" + str(r) + 
                          "_sampled_CT_ds.pkl", 'rb') as f:
                    X = pickle.load(f)
                    y = pickle.load(f)
                f.close()
            else:   
                with open(curr_dir + "/" + datasets_path + "/random/" + 
                          dataset_name + "_rnd_ratio" + str(r) + 
                          "_sampled_CT_ds.pkl", 'rb') as f:
                    X = pickle.load(f)
                    y = pickle.load(f)
                f.close()

            # perform a train-test split to evaluate model performances
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                stratify=y, 
                                                                test_size=0.3, 
                                                                random_state=seed)
            model.fit(X_train, y_train)

            # predict probabilities of each class 
            y_score = model.predict_proba(X_test)[:,1]          

            # pr calculation
            precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)
            # ap calculation
            ap = metrics.average_precision_score(y_test, y_score) 

            # store results
            res[t] = {   
                "recall": recall,
                "precision": precision,    
                "average_precision_score": ap
            }     

        pr_res[r] = res

    # figure    
    ratio = [10, 20, 100]
    type = ["sampled", "random"]
    fig = plt.figure(figsize=(8,6))
    for i, r in enumerate(ratio):
        ratios = [0.1, 0.2, 0.01]
        for t in type:
            ap = pr_res[r][t]["average_precision_score"]
            if t == "sampled": 
                label = "R={:}, AP1={:.3f}".format(ratios[i], ap)
                colors = ["#860222", "#d00c3a", "#f87e9b"]
            else: 
                label = "R={:}, AP2={:.3f}".format(ratios[i], ap) 
                colors = ["#4c4c4c", "#816f73", "#bbb7b8"]

            plt.plot(pr_res[r][t]["recall"], pr_res[r][t]["precision"], 
                        color=colors[i], label=label, alpha = .7) 
                                                        
            # x ticks
            plt.xticks(np.arange(0.0, 1.1, step=0.1), fontsize=20)
            plt.xlabel("Recall", fontsize=20)

            # y ticks 
            plt.yticks(np.arange(0.0, 1.1, step=0.1), fontsize=20)
            plt.ylabel("Precision", fontsize=20)

            # grid
            plt.grid(zorder=0, color='lavender')

            plt.title(dataset_name, fontsize=20)
            plt.legend(prop={'size' : 18}, loc='lower left')
            
    # save
    plt.savefig(output_path + "/PRC_" + dataset_name + "_sampling.png", 
                bbox_inches='tight', dpi=600)

    return pr_res


def validate_model(dataset_name, model, seed=SEED):
    
    """This function validates the model on pre-built external datasets.

    :param dataset_name: a string indicating the dataset name
    :param model: a scikit-learn estimator
    :return: a dataframe of recall scores for each external dataset 
    """
    
    # seed the global RNG 
    random.seed(seed)
    # seed the global NumPy RNG
    np.random.seed(seed)

    # get current directory path
    curr_dir = os.path.dirname(__file__)
    # define external datasets path
    ext_datasets = ["AP_CTIs", "B2_B4_CTIs", "CTD_CTIs/CYP", "CTD_CTIs/CA"]
    datasets_path = ["../../cti_datasets/" + d for d in ext_datasets]

    recall_scores = []  
    for path in datasets_path:
        # read dataset 
        pred_ds = {}
        with open(curr_dir + "/" + path + "/pred/" + 
                  dataset_name + "_pred_CT_ds.pkl", 'rb') as f:
            pred_ds['X'] = pickle.load(f)
            pred_ds['pairs'] = pickle.load(f)
        f.close()
        print(path, pred_ds['X'].shape)

        # predict new pairs
        y_pred = model.predict(pred_ds['X'])
        recall_score = np.round(metrics.recall_score(np.ones(len(y_pred)), 
                                                     y_pred), 2)
        recall_scores.append(recall_score)
        
    # create scores dataframe 
    recall_scores_df = pd.DataFrame({"Recall score" : recall_scores})
    # assign datasets
    datasets = ["PB", "B2B4", "CYP", "CA"]
    recall_scores_df['Dataset'] = datasets

    # figure
    plt.figure(figsize=(8,6))
    sns.pointplot(data=recall_scores_df, x = "Dataset", y = "Recall score",
                        join=True, linestyles = ':', dodge=0.1, scale=1.5) 
    
    # set title
    plt.title(dataset_name, fontsize=20)

    # x-axis
    plt.xlabel("Dataset", fontsize=20)
    plt.xticks(fontsize=20)

    # y-axis
    plt.ylabel("Recall score", fontsize=20)
    plt.ylim(0.55, 1.05)
    plt.yticks(np.arange(0.6, 1.05, 0.1), fontsize=20)

    # grid
    plt.grid(visible=True)

    return recall_scores_df