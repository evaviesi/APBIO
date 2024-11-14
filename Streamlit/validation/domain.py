#!/usr/bin/env python
# -*- coding: utf-8 -*-

# define imports
import os
import random
import pickle
import gzip
import numpy as np
import pandas as pd

# import scikit-learn modules 
from sklearn.manifold import TSNE

# import plotting libraries 
import plotly.express as px 
import warnings
warnings.filterwarnings("ignore")

# set seed 
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)


def search_out_ad(dataset_name, df, seed=SEED):

    """This function searches for molecules and targets outside 
    the applicability domain (AD).

    :param dataset_name: a string indicating the dataset name  
    :param df: a dataframe of compound-target feature vectors 
        having (inchikey, uniprot) as index
    :param seed: seed for reproducibility 
    :return: a dictionary with the applicability domain of the molecule and the
        target, the average distance to the 5 nearest neighbors and their keys   
    """

    # seed the global RNG 
    random.seed(seed)
    # seed the global NumPy RNG
    np.random.seed(seed)

    # get current directory path
    curr_dir = os.path.dirname(__file__)

    # open AD dictionary
    with open(curr_dir + "/data/AD_thr.pkl", "rb") as f:
        AD_thr = pickle.load(f)

    # define vector index
    if dataset_name == "F1F2" or dataset_name == "A1A2":
        idx = 256
    if dataset_name == "MFP":
        idx = 1024
    if dataset_name == "APchem":
        idx = 512
    if dataset_name == "CCchem":
        idx = 640

    ad_dct = {}
    # load nn model 
    neigh_model = pickle.load(open(
        curr_dir + "/data/" + dataset_name + "_mol_nn_model.pkl", "rb"))

    # compute nn for molecules 
    mol_nn = {}
    for i in range(len(df)):
        nn = neigh_model.kneighbors(
            df.values[i, 0:idx].reshape(1, -1),
            n_neighbors=5, 
            return_distance=True)  
        
        mol_nn[df.index[i][0]] = {
            "keys"      : np.array(AD_thr[dataset_name +
                                          "_all_molecules"])[nn[1]], 
            "distances" : nn[0]
        }
      
    # search for entries outside the ad  
    mol_ad = {}
    for k in mol_nn.keys():
        mol_type = "new"
        # compute nn average distance 
        mol_ts_dist = np.round(np.mean(mol_nn[k]["distances"]), 2)
        # positive
        if k in AD_thr[dataset_name + "_pos_molecules"]:
            mol_type = "positive"
        # negative
        elif k in AD_thr[dataset_name + "_neg_molecules"]:
            mol_type = "negative"
        # in
        elif k in AD_thr[dataset_name + "_all_molecules"]:
            mol_type = "in"
        # out
        elif mol_ts_dist > AD_thr[dataset_name + "_molecule_thr"]:
            mol_type = "out"
        mol_ad[k] = [mol_type, mol_ts_dist, mol_nn[k]["keys"][0]] 
    
    ad_dct["mol"] = mol_ad

    # load nn model for targets
    neigh_model = pickle.load(open(
        curr_dir + "/data/tg_nn_model.pkl", "rb"))

    # compute nn for targets
    tg_nn = {}
    for i in range(len(df)):
        nn = neigh_model.kneighbors(
            df.values[i, idx:df.shape[1]].reshape(1, -1), 
            n_neighbors=5, 
            return_distance=True)  
        
        tg_nn[df.index[i][1]] = {
            "keys"      : np.array(AD_thr["all_targets"])[nn[1]],
            "distances" : nn[0]
        }

    # search for entries outside the ad  
    tg_ad = {}
    for k in tg_nn.keys():
        tg_type = "new"
        # compute nn average distance
        tg_ts_dist = round(np.mean(tg_nn[k]["distances"]), 2)
        # positive
        if k in AD_thr[dataset_name + "_pos_targets"]:
            tg_type = "positive"
        # in
        elif k in AD_thr["all_targets"]:
            tg_type = "in"
        # out
        elif tg_ts_dist > AD_thr["target_thr"]:
            tg_type = "out"
        tg_ad[k] = [tg_type, tg_ts_dist, tg_nn[k]["keys"][0]]

    ad_dct["tg"] = tg_ad

    return ad_dct 
 

def sign_proj(dataset_name, df, nn_dct, seed=SEED):

    """
    this function computes and plots the 2D t-SNE projection 
    of compound signatures. 
    :param dataset_name: a string indicating the dataset name  
    :param df: a dataframe of compound-target feature vectors 
        having (inchikey, uniprot) as index
    :param ad_dct: a dictionary having inchikey as key and list 
        of nearest neighbor keys as values 
    :param seed: seed for reproducibility 
    :return: a plotly figure of 2D t-SNE projections 
    """

    # seed the global RNG 
    random.seed(seed)
    # seed the global NumPy RNG
    np.random.seed(seed)

    # get current directory path
    curr_dir = os.path.dirname(__file__)
    datasets_path = "/../../cti_datasets/AP_CTIs/sampled"

    # read dataset
    with gzip.open(
        curr_dir + datasets_path + "/" + dataset_name + 
        "_sampled_CT_ds.pkl.gz", "rb") as f:
        X = pickle.load(f)
        y = pickle.load(f)
    f.close()

    # define vector index
    if dataset_name == "F1F2" or dataset_name == "A1A2":
        idx = 256
    if dataset_name == "MFP":
        idx = 1024
    if dataset_name == "APchem":
        idx = 512
    if dataset_name == "CCchem":
        idx = 640

    # get molecule dataframe
    mol_df = pd.concat([
        df.iloc[:, 0:idx], 
        pd.DataFrame(X[:, 0:idx], index=y.index)
    ])
    mol_df.index = list([i[0] for i in mol_df.index])
    
    # drop duplicated molecules 
    mol_df = mol_df.reset_index() \
                   .drop_duplicates(subset="index", keep="first") \
                   .set_index("index")

    # add group to molecule dataframe (all, nn, query)
    groups = pd.DataFrame({"group" : "all compounds"}, index = mol_df.index)
    groups.loc[list(nn_dct["mol"].values())[0][2], "group"] = "nearest neighbors"
    groups.loc[df.index[0][0], "group"] = "query compound"
    
    # apply TSNE transformation
    tsne = TSNE(n_components=2, metric="cosine", random_state=seed)
    projections = pd.DataFrame(tsne.fit_transform(mol_df))
    projections.columns = ["dim1", "dim2"]

    # figure  
    hover_name = [("inchikey: "+ x) for x in groups.index]
    fig = px.scatter(
        projections, x="dim1", y="dim2",
        color=groups["group"],
        hover_name=hover_name,
        width=450, height=500,
        hover_data={"dim1": False, "dim2": False},
        color_discrete_map={
            "all compounds" : "#c8c8c8", 
            "nearest neighbors" : "#55f012", 
            "query compound" : "#f500d5"
        }
    )

    # set layout  
    fig.update_traces(marker_size=8, opacity=0.6)
    # legend 
    fig.update_layout(
        legend=dict(
            title = "Group", 
            title_font_size=16, 
            font_size=16))
    
    # x-axis
    fig.update_layout(
        xaxis = dict(
            tickfont = dict(size=16), 
            title_font_size=16))
    
    # y-axis 
    fig.update_layout(
        yaxis = dict(
            tickfont = dict(size=16), 
            title_font_size=16))
    # color 
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,255)", 
        plot_bgcolor="rgba(0,0,0,0)")

    return fig
