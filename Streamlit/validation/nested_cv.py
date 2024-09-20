#!/usr/bin/env python
# -*- coding: utf-8 -*-

import  sys, random, pickle, os
import numpy as np
import pandas as pd

# sklearn libraries
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
# seed the global RNG 
random.seed(SEED)
# seed the global NumPy RNG
np.random.seed(SEED)

# input file with tasks and parameters 
task_id = sys.argv[1]   # <TASK_ID>   
filename = sys.argv[2]  # <FILE>  

# dictionary with task ID as key
input_pickle = pickle.load(open(filename, 'rb'))   

# run in parallel all task IDs
model_name = input_pickle[task_id][0][0]     
dataset = input_pickle[task_id][0][1]
print(model_name, dataset)   

# load data
ds_path = '/aloy/home/eviesi/CTI_datasets/10_04_24/' + dataset + '.pkl'
print(ds_path)
with open(ds_path, 'rb') as f:
    X = pickle.load(f)
    y = pickle.load(f)
f.close()

print("Total number of ct pairs: ", y.shape[0]) 

# define output path
output_path = "/aloy/home/eviesi/CTI_datasets/10_04_24/pred/%s.%s.pkl"%(model_name, dataset.split('_')[0])

def model_eval(X, y, model_name, output_path, seed=SEED):

    # set seed
    random.seed(seed)
    np.random.seed(seed)

    # check dataset shape 
    y = y.values.ravel() 
    print("X: ", X.shape, "\ny: ", y.shape)

    # define grid of parameters for hyperparameter tuning
    C_range = [0.01, 0.1, 1.0, 10]                                             # LR
    n_neighbors_range = [5, 15, 25, 35]                                        # KNN                 
    n_estimators_range = [100, 200, 300, 400]                                  # RF 
    layer_sizes_range = [(64,), (128,), (256,), (512,)]                        # MLP 
    

    # define type of cross-validation 
    k_inner = 5
    k_outer = 10 
    rep = 2
    rskf_inner = RepeatedStratifiedKFold(n_splits=k_inner, n_repeats=rep, random_state=seed)
    rskf_outer = RepeatedStratifiedKFold(n_splits=k_outer, n_repeats=rep, random_state=seed)
    
    # initialize models 
    if model_name == 'lr':
        # increased number of iterations for convergence
        estimator = LogisticRegression(max_iter=1000, random_state=seed)   
        # the parameter grid is defined as a dictionary 
        param_grid = dict(C=C_range)

    if model_name == 'knn':
        estimator = KNeighborsClassifier(metric='cosine')
        # the parameter grid is defined as a dictionary 
        param_grid = dict(n_neighbors=n_neighbors_range)

    if model_name == 'rf':
        estimator = RandomForestClassifier(random_state=seed)
        # the parameter grid is defined as a dictionary 
        param_grid = dict(n_estimators=n_estimators_range)

    if model_name == 'mlp':
        estimator = MLPClassifier(max_iter=1000, random_state=seed)
        # the parameter grid is defined as a dictionary 
        param_grid = dict(hidden_layer_sizes=layer_sizes_range)
   

    # define grid search cross-validation 
    grid_cv = GridSearchCV(estimator, param_grid = param_grid, scoring = 'balanced_accuracy',  
                            cv = rskf_inner, refit=True)
    

    # compute validation scores 
    nested_cv_scores = cross_validate(grid_cv, X, y, scoring=['roc_auc', 'average_precision',  
                                                              'recall', 'precision', 'f1',
                                                              'balanced_accuracy'], cv = rskf_outer)  
    
    print("CV DONE")
 
    # save performance results for each model 
    f = open(output_path, "wb")  
    pickle.dump(nested_cv_scores, f)    
    f.close()   

    
# run function to evaluate model & save results 
model_eval(X, y, model_name, output_path)

print('FINISHED')