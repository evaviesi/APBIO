#!/usr/bin/env python
# -*- coding: utf-8 -*-

# define imports
import os
import re
import sys
import random
import pickle
import requests
import subprocess
import numpy as np
import pandas as pd

# import scikit-learn modules 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# set seed 
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)


def extract_features(uniprot_ids, output_path, fit=False, seed=SEED):
    
    """This function calculates FASTA sequence descriptors and low-dimensional
    features for target proteins.  
    
    :param uniprot_ids: a list of uniprot ids
    :param output_path: path to save descriptors 
    :param fit: a boolean value indicating whether to fit scaler and pca
    :param seed: seed for reproducibility 
    :return: a dataframe of target features having uniprot as index
    """
    
    # seed the global RNG 
    random.seed(seed)
    # seed the global NumPy RNG
    np.random.seed(seed)

    # get current directory path
    curr_dir = os.path.dirname(__file__)

    # remove file if it exists 
    if os.path.isfile(output_path + "/seq.fasta"):
        os.remove(output_path + "/seq.fasta")
  
    # open file 
    f_fasta = open(output_path + "/seq.fasta", "a")
    # query for FASTA
    for id in uniprot_ids:
        r = requests.get("https://rest.uniprot.org/uniprotkb/" + id + 
                         "?format=fasta")
        # replace header with uniprot id 
        fasta_string = re.sub(r'^.*?\n', '>'+str(id)+'\n', r.text) 
        # write to file 
        f_fasta.write(fasta_string) 
    f_fasta.close()

    # use iFeature package to compute target descriptors 
    descriptors = ["AAC", "DPC", "NMBroto", "Moran", "Geary", "CTDC", "CTDT",
                   "CTDD", "SOCNumber", "QSOrder", "PAAC"]
    for desc in descriptors: 
        subprocess.run([sys.executable, curr_dir + "/iFeature/iFeature.py", '--file', 
                        output_path + "/seq.fasta", '--type', desc, '--out', 
                        output_path + "/" + desc + ".csv" ])

    # read computed descriptors to create a single file
    descriptors_df = pd.DataFrame()
    for desc in descriptors:
        df = pd.read_csv(output_path + "/" + desc + ".csv", sep="\t", 
                         index_col=0)
        descriptors_df = pd.concat([descriptors_df, df], axis=1)
        descriptors_df = descriptors_df.sort_index()
    descriptors_df.index.name = "UniProtKB"
    
    # save dataframe
    descriptors_df.to_csv(output_path + "/descriptors.csv", index=True)

    if fit:
        # scale data before PCA 
        scaler = StandardScaler()
        scaled_descriptors = scaler.fit_transform(descriptors_df)
        
        # perform pca dimensionality reduction 
        pca = PCA(n_components=128,  svd_solver='full', random_state=seed)   
        reduced_descriptors = pd.DataFrame(pca.fit_transform(scaled_descriptors), 
                                           index=descriptors_df.index)
        print("Num. components: ", pca.n_components_, 
            "\nExplained variance: ", round(pca.explained_variance_ratio_
                                            .sum() * 100, 2),"%",
            "\nShape: ", reduced_descriptors.shape) 
    else:
        # load scaler and pca objects
        scaler = pickle.load(open(
            curr_dir + "/preparation_obj/scaler.pkl", "rb"))
        pca = pickle.load(open(
            curr_dir + "/preparation_obj/pca.pkl", "rb"))

        # apply scaler and pca transformation 
        scaled_descriptors = scaler.transform(descriptors_df.values)
        reduced_descriptors = pd.DataFrame(pca.transform(scaled_descriptors), 
                                           index=descriptors_df.index)
        print("Explained variance: ", round(pca.explained_variance_ratio_
                                            .sum() * 100, 2),"%", 
              "\nShape: ", reduced_descriptors.shape) 

    # save dataframe
    reduced_descriptors.to_csv(
        output_path + "/reduced_descriptors.csv", index=True)
    # read dataframe
    target_features = pd.read_csv(
        output_path + "/reduced_descriptors.csv", index_col=0)

    return target_features
