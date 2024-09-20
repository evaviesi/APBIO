#!/usr/bin/env python
# -*- coding: utf-8 -*-

# define imports 
import  os
import random
import pickle
import numpy as np
import pandas as pd

# import scikit-learn modules
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn import svm

# import plotting libraries 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# set seed 
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)


def sample_neg(X, y, dataset_name, output_path, ratio=10, 
               rnd=False, seed=SEED):

    """This function selects negative instances either using 
    the ocsvm or randomly. 

    :param X: an array of (n_samples, n_features)
    :param y: a dataframe of class labels having (inchikey, uniprot) as index 
    :param output_path: path to save dataset
    :param ratio: a value corresponding to the ratio of negatives to positives
    :param: a boolean value indicating whether to perform random selection
    :param seed: seed for reproducibility 
    :return: a numpy array of (n_samples, n_features) and a dataframe 
        of class labels having (inchikey, uniprot) as index 
    """

    # seed the global RNG 
    random.seed(seed)
    # seed the global NumPy RNG
    np.random.seed(seed)

    # get positive and negative indexes
    pos_index = np.where(y.values == 1)[0]
    neg_index = np.where(y.values == 0)[0] 
    print("#pos:", len(pos_index), "\n#neg:", len(neg_index))

    # generate random negative indexes
    if rnd:
        neg_idx_selected = random.sample(list(neg_index), 
                                         len(pos_index)*ratio)
        dataset_name = dataset_name + '_rnd'

    else:
        # define grid of parameters to be tuned  
        nu_range = [0.01, 0.03, 0.05, 0.1]     
        param_grid = dict(nu=nu_range)

        # define estimator 
        estimator = svm.OneClassSVM(kernel='linear')  

        # define type of cross-validation 
        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)

        # find the best estimator by optimizing for recall 
        grid_cv = GridSearchCV(estimator, 
                               param_grid=param_grid, 
                               scoring='recall', 
                               refit=True, 
                               cv=rkf, 
                               return_train_score=True) 
        grid_cv.fit(X[pos_index], np.ones(len(pos_index)))

        # show best estimator and recall score 
        print("Best estimator:", grid_cv.best_estimator_, 
            "\nTest recall:",  np.round(
                grid_cv.cv_results_['mean_test_score'][grid_cv.best_index_], 2))

        # compute ocsvm distances for positive and negative instances 
        pos_sign_dist = grid_cv.best_estimator_.decision_function(X[pos_index])
        neg_sign_dist = grid_cv.best_estimator_.decision_function(X[neg_index])

        # create dataframe of negatives distances
        neg_pairs = pd.DataFrame({'Distance' : neg_sign_dist}, 
                                 index=y.index[neg_index])

        # sort negative instances by ocsvm distances and select the ratio 
        neg_selected = neg_pairs.sort_values(by='Distance') \
                                .head(ratio*len(pos_index))
        neg_idx_selected = np.where(y.index.isin(neg_selected.index))[0]
        print("Number of selected negative pairs:", len(neg_idx_selected))

        # figure 
        plt.figure(figsize=(8, 6))
        # distances distribution
        plt.hist([neg_sign_dist, pos_sign_dist], 
                 color=['r', 'b'], alpha=0.5)
        plt.hist(neg_selected.values, color='g', alpha=0.5)

        # legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1),  
                   labels=['Neg', 'Pos', 'Selected Neg'], fontsize=20)
        # title
        plt.title(dataset_name, size=20)

        # x-axis
        plt.xlabel('Signed distance', size=20)
        plt.xticks(size=20)
        plt.xlim([min(neg_sign_dist), max(pos_sign_dist)])

        # y-axis
        plt.ylabel('Count of pairs', size=20)
        plt.yticks(size=20)
        plt.yscale('log')
        plt.ylim([1, 10**6])
        
        # grid
        plt.grid(visible=True)

        # save 
        plt.savefig(output_path + "/" + dataset_name + "_ratio" + str(ratio) + 
                    "_selected_neg.png", 
                    bbox_inches='tight', dpi=600)
    
    # concat positive and negative features
    X = np.concatenate((X[pos_index], X[neg_idx_selected]))

    # create positive and negative labels
    pos_lab = list(y.index[pos_index])
    neg_lab = list(y.index[neg_idx_selected])
    y = pd.DataFrame({"y" : np.array([1] * len(pos_index) + 
                                     [0] * len(neg_idx_selected))}, 
                                     index = pos_lab + neg_lab)

    # shuffle labels
    shuffle = np.random.permutation(len(X))
    X = X[shuffle]
    y = y.iloc[shuffle]

    # save X and y 
    f = open(output_path + "/" + dataset_name + "_sampled_CT_ds.pkl", "wb")  
    pickle.dump(X, f)      
    pickle.dump(y, f)
    f.close()

    return X, y