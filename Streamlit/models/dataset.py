#!/usr/bin/env python
# -*- coding: utf-8 -*-

# define imports 
import os
import random
import pickle
import numpy as np
from collections import defaultdict  

# import signaturizer module
from signaturizer import Signaturizer

# import rdkit library 
from rdkit import Chem
from rdkit.Chem import AllChem

import warnings
warnings.filterwarnings("ignore")

# set seed 
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)


def generate_dataset(ct_pairs, target_desc, dataset_name, 
                     output_path, seed=SEED):

    """This function generates a dataset of compound-target features.

    :param ct_pairs: a dataframe of compound-target pairs having 
        smiles as index, inchikey and uniprot as columns  
    :param target_desc: a dataframe of target descriptors having 
        uniprot as index 
    :param dataset_name: a string indicating the dataset name  
    :param output_path: path to save dataset
    :param seed: seed for reproducibility 
    :return: a numpy array of (n_samples, n_features) and a dictionary
        having inchikey as keys and list of uniprots as values
    """

    # seed the global RNG 
    random.seed(seed)
    # seed the global NumPy RNG
    np.random.seed(seed)

    # get current directory path
    curr_dir = os.path.dirname(__file__)

    # get molecule smiles 
    smiles = list(ct_pairs.index.unique())
    # show total number of molecules 
    print("Number of molecules:", len(smiles))
    # show total number of targets 
    print("Number of targets:", len(target_desc))

    # create list of ct pairs 
    ct_pairs = list(zip(ct_pairs.iloc[:,0], ct_pairs.iloc[:,1]))
    # create dictionary with inchikey as keys and list of uniprots as values
    pairs = defaultdict(list)
    for k, v in ct_pairs: pairs[k].append(v)

    # F1F2
    if dataset_name == "F1F2": 
        print("############### F1 + F2 ####################")
        models = ['F1', 'F2']
        pred_sign = []
        for model in models:
            print(model)
            # load signaturizer 
            sign = Signaturizer(
                curr_dir + "/signaturizers/" + model, local=True, 
                verbose=True)
            x = sign.predict(smiles)
            pred_sign.append(x.signature)
        pred_sign = np.concatenate(pred_sign, axis=1)
        print("Signatures dimension:", pred_sign.shape)

        # compute ct pairs dataset 
        print("Computing CT pairs")
        ct_pairs = []
        for i, k in enumerate(list(pairs.keys())):
            ct_pairs.append(np.array([
                np.concatenate((pred_sign[i], row)) for 
                row in target_desc.loc[pairs[k]].values]))
        X = np.concatenate(ct_pairs, axis=0)
        print("CT pairs dimension:", X.shape)


    # A1A2
    if dataset_name == "A1A2": 
        print("############### A1 + A2 ####################")
        models = ['A1', 'A2']
        pred_sign = []
        for model in models:
            print(model)
            # load signaturizer 
            sign = Signaturizer(
                curr_dir + "/signaturizers/" + model, local=True, 
                verbose=True)
            x = sign.predict(smiles)
            pred_sign.append(x.signature)
        pred_sign = np.concatenate(pred_sign, axis=1)
        print("Signatures dimension:", pred_sign.shape)

        # compute ct pairs dataset 
        print("Computing CT pairs")
        ct_pairs = []
        for i, k in enumerate(list(pairs.keys())):
            ct_pairs.append(np.array([
                np.concatenate((pred_sign[i], row)) for 
                row in target_desc.loc[pairs[k]].values]))
        X = np.concatenate(ct_pairs, axis=0)
        print("CT pairs dimension:", X.shape)


    # MFP
    if dataset_name == "MFP": 
        print("############### MFP ####################")
        mfp_molecules = []
        mfp_data = []
        radius = 2      # 2 angstroms 
        n_bits = 1024   # legnth of vector
    
        # generate rdkit molecule object
        for s in list(smiles):
            m = Chem.MolFromSmiles(s)
            mfp_molecules.append(m)
            
        # iterate through molecules
        for mol in mfp_molecules: 
            # generate Morgan Fingerprint
            mfp_vector = np.array(AllChem.GetMorganFingerprintAsBitVect(
                    mol, 
                    useChirality=True, 
                    radius=radius,
                    nBits=n_bits))
            mfp_data.append(mfp_vector)

        # compute ct pairs dataset 
        print("Computing CT pairs")
        x = np.array(mfp_data)
        ct_pairs = []
        for i, k in enumerate(list(pairs.keys())):
            ct_pairs.append(np.array([
                np.concatenate((x[i], row)) for 
                row in target_desc.loc[pairs[k]].values]))
        X = np.concatenate(ct_pairs, axis=0)
        print("CT pairs dimension:", X.shape)

    
    # APchem
    if dataset_name == "APchem":
        print("############### APDB GLOBAL ####################")
        models = ['F1', 'F2', 'M1', 'Q1']
        pred_sign = []
        for model in models:
            print(model)
            # load signaturizer 
            sign = Signaturizer(
                curr_dir + "/signaturizers/" + model, local=True, 
                verbose=True)
            x = sign.predict(smiles)
            pred_sign.append(x.signature)
        pred_sign = np.concatenate(pred_sign, axis=1)
        print("Signatures dimension:", pred_sign.shape)

        # compute ct pairs dataset 
        print("Computing CT pairs")
        ct_pairs = []
        for i, k in enumerate(list(pairs.keys())):
            ct_pairs.append(np.array(
                [np.concatenate((pred_sign[i], row)) for 
                 row in target_desc.loc[pairs[k]].values]))
        X = np.concatenate(ct_pairs, axis=0)
        print("CT pairs dimension:", X.shape)
        

    # CCchem
    if dataset_name == "CCchem":
        print("############### CC CHEMISTRY GLOBAL ####################")
        models = ['A1', 'A2', 'A3', 'A4', 'A5']
        pred_sign = []
        for model in models:
            print(model)
            # load signaturizer 
            sign = Signaturizer(
                curr_dir + "/signaturizers/" + model, local=True, 
                verbose=True)
            x = sign.predict(smiles)
            pred_sign.append(x.signature)
        pred_sign = np.concatenate(pred_sign, axis=1)
        print("Signatures dimension:", pred_sign.shape)

        # compute ct pairs dataset 
        print("Computing CT pairs")
        ct_pairs = []
        for i, k in enumerate(list(pairs.keys())):
            ct_pairs.append(np.array([np.concatenate((pred_sign[i], row)) for 
                 row in target_desc.loc[pairs[k]].values]))
        X = np.concatenate(ct_pairs, axis=0)
        print("CT pairs dimension:", X.shape)

    # save X and pairs 
    f = open(output_path + "/" + dataset_name + "_CT_ds.pkl", "wb")  
    pickle.dump(X, f)      
    pickle.dump(pairs, f)
    f.close()  

    return X, pairs