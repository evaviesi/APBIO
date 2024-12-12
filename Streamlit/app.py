#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Predict potential compound-target interactions (CTIs) through
compound bioactivity signatures and target sequence descriptors. 
'''

# define imports 
import streamlit as st
import os
import io
import random
import pickle
import ast
import shutil
import requests
import urllib
import numpy as np
import pandas as pd

# import rdkit library 
from rdkit import Chem

# import py3dmol module 
import py3Dmol
from stmol import showmol

# import modules
from ifeatures.fasta_feat import extract_features
from models.dataset import generate_dataset
from validation.domain import search_out_ad, sign_proj

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
# seed the global RNG 
random.seed(SEED)
# seed the global NumPy RNG
np.random.seed(SEED)

# set working directory 
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# page configuration 
st.set_page_config(layout="centered")

# import styles
with open('./static/css/main.css') as f:
    st.markdown(f'<style>{f.read()}</style>', 
                unsafe_allow_html=True)

# initialize session state variables
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

if 'input_mol' not in st.session_state:
    st.session_state.input_mol = ""

if 'input_tg' not in st.session_state:
    st.session_state.input_tg = ""

# update session state variables
def click_button():
    st.session_state.clicked = True

def get_mol():
    st.session_state.input_mol = "C1=CC=CC=C1"

def get_tg():
    st.session_state.input_tg = "Q9UHW9"

### single CTI
def change_mol_style(mol_id, style):
    """Change compound visualization."""

    cid = ""
    inchikey = ""
    # query compound cid and inchikey  
    try:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/" + \
              mol_id + "/property/InChIKey/CSV"
        res = pd.read_csv(url)
        cid, inchikey = res.CID[0], res.InChIKey[0]
    
    except urllib.error.URLError as e:
        print(f"Error searching PubChem for {mol_id}: {e}")
        st.warning("Invalid SMILES. Please try again.", icon="⚠️")

    except requests.exceptions.RequestException as e:
        print(f"Error searching PubChem for {mol_id}: {e}")
        st.warning("Request exception. Please try again.", icon="⚠️")

    except Exception as e:
        print(f"Error searching PubChem for {mol_id}: {e}")
        st.warning("Invalid URL. Please try again.", icon="⚠️")

    # compound figure
    if cid:
        # stick
        if style == "stick":      
            # get 3Dmol object 
            view = py3Dmol.view(query=f"cid:{cid}", width=350, height=300)
            view.setStyle(
                { 
                    "stick": { 
                        "thickness": 0.7 
                    }
                }
            )
        # sphere 
        elif style == "sphere":
            # get MolBlock object 
            mol = Chem.MolToMolBlock(Chem.MolFromSmiles(mol_id))
            # get 3Dmol object
            view = py3Dmol.view(width=350, height=300)
            view.addModel(mol, 'mol')
            view.setStyle(
                { 
                    "sphere": {}
                }
            )
        # show figure
        showmol(view, width=350, height=300)
        st.markdown(
            '''<div style='text-align: center;'>Compound structure</div>''', 
            unsafe_allow_html=True
        )
    else: mol_id = None
    
    return mol_id, inchikey


def change_tg_style(tg_id, style):
    """Change target visualization."""

    pdb_code = ""
    # query target pdb
    r = requests.get(
        "https://rest.uniprot.org/uniprotkb/" + tg_id + "?query=organism_id:"
        "9606+AND+database:pdb&format=tsv&fields=xref_pdb"
    )
    r =  [line for line in r.text.split("\n") if line]

    # save pdb 
    if r[0] == "Error messages":
        st.warning("Invalid UniProtKB. Please try again.", icon="⚠️")
        tg_id = None

    elif len(r) == 1: 
        url = "https://www.alphafold.ebi.ac.uk/entry/" + tg_id 
        st.markdown(
            '''No structure available in the PDB. 
            AlphaFold structure prediction: [link](%s)
            ''' % url
        )
    else: 
        pdb_code = [pdb for pdb in r[1].split(";")][0]

    # target figure
    if pdb_code:
        # get 3Dmol object
        view = py3Dmol.view(
            query=f"pdb:{pdb_code.lower()}", width=350, height=300
        )
        # cartoon
        if style == "cartoon":
            view.setStyle(
                {
                    "cartoon": {
                        "color": "spectrum",
                        "thickness": 0.2,
                    }
                }
            )
        # stick
        elif style == "stick":
            view.setStyle(
                {
                    "stick": {
                        "thickness": 0.5,
                    }
                }
            )
        # show figure
        showmol(view, width=350, height=300)
        st.markdown(
            '''<div style='text-align: center;'>Target structure</div>''', 
            unsafe_allow_html=True
        )
        
    return tg_id


@st.cache_data(ttl=300, show_spinner=False)
def generate_cti_vec(selected_space, mol_id, inchikey, tg_id):  
    """Generate compound-target interaction vector."""

    # create tmp directory
    tmp_dir = os.getcwd() + "/tmp"
    if not os.path.exists(tmp_dir): os.makedirs(tmp_dir) 

    # extract target sequence features
    target_feat = extract_features([tg_id], tmp_dir)
     
    # create dataframe
    cti_df = pd.DataFrame({"InChIkey"  : [inchikey], 
                           "UniProtKB" : [tg_id]
                           }, index = [mol_id])
    
    # generate dataset 
    X, pairs = generate_dataset(cti_df, target_feat, selected_space, tmp_dir)
    y = [(k, v) for k in pairs.keys() for v in pairs[k]]
    pred_df = pd.DataFrame(X, index=y)

    # remove tmp directory
    shutil.rmtree(tmp_dir) 

    return pred_df


def run_prediction(selected_space, df):
    """Run compound-target interaction prediction"""

    # load final model
    final_model = pickle.load(open("./models/pred_models/lr." + 
                                   selected_space + ".pkl", "rb"))
    # predict new pairs
    y_pred = final_model.predict(df.values)
    y_proba = final_model.predict_proba(df.values)

    # construct prediction dataframe
    res_df = pd.DataFrame({
        "MODEL_PREDICTION": y_pred, 
        "MODEL_CONFIDENCE": np.round(y_proba[range(len(y_pred)), y_pred], 4)
        }, index = df.index)
    
    return res_df


### multiple CTIs
def upload_file():
    """Upload CSV file"""
    
    df = None
    input_file = st.file_uploader("2\. Choose a CSV file with multiple "
                                  "compound SMILES and target UniProtKB:",
                                  type = ["csv"], 
                                  accept_multiple_files = False)
    # validate file content
    if input_file is not None:
        try:
            # read csv file 
            df = pd.read_csv(input_file)
            # check number of columns 
            if len(df.columns) != 2:
                df = None 
                st.warning("The uploaded file has an incorrect number of columns. "
                           "Please try again.", icon="⚠️")
        except Exception as e:
            print(f"Error reading file: {e}")
            st.warning("Invalid file. Please try again.", icon="⚠️")

    return df


@st.cache_data(ttl=300, show_spinner=False)
def parse_data(df):
    """Parse CSV file"""

    # query for compound inchikey and name 
    inchikeys = {}
    chemical_names = {}
    df.columns = ["SMILES", "UniProtKB"]
    smiles_lst = list(df.SMILES.unique()) 
    for smiles in smiles_lst: 
        try: 
            r_inchikey = pd.read_csv(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/" + 
                smiles + "/property/InChIKey/CSV"
            )
            inchikeys[smiles] = r_inchikey.InChIKey[0]
            r_name = pd.read_csv(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/" + 
                smiles + "/synonyms/TXT", nrows=0
            )
            chemical_names[smiles] = r_name.columns[0]

        except Exception as e:
            inchikeys[smiles] = ""
            print(f"Error searching PubChem for {smiles}: {e}")

    # query for target symbol
    uniprot_lst = list(df.UniProtKB.unique())  
    symbols = {}
    for uniprot in uniprot_lst:
        try:
            # query for target name
            r_symbol = requests.get("https://rest.uniprot.org/uniprotkb/" + 
                                    uniprot + "?fields=accession%2Cgene_names")
            r_symbol = ast.literal_eval(r_symbol.text)
            symbols[uniprot] = r_symbol["genes"][0]["geneName"]["value"]
        except Exception as e:
            print(f"Error searching UniProt for {uniprot}: {e}")

    # generate output dataset 
    parsed_df = df.copy()
    parsed_df.insert(1, "InChIKey", df.SMILES.map(inchikeys))
    parsed_df.insert(2, "Chemical_Name", df.SMILES.map(chemical_names))
    parsed_df.insert(parsed_df.shape[1], "Symbol", df.UniProtKB.map(symbols))
    # drop rows containing missing values
    parsed_df = parsed_df.dropna()

    return parsed_df


@st.cache_data(ttl=300, show_spinner=False)
def generate_cti_vecs(selected_space, df):
    """Generate compound-target interaction vectors."""

    # create tmp directory
    tmp_dir = os.getcwd() + "/tmp"
    if not os.path.exists(tmp_dir): os.makedirs(tmp_dir) 

    # extract target sequence features
    uniprot_lst = list(df["UniProtKB"].unique())

    target_feat = extract_features(uniprot_lst, tmp_dir)

    # create dataframe
    cti_df = df[["SMILES", "InChIKey", "UniProtKB"]].set_index("SMILES")

    # generate dataset 
    X, pairs = generate_dataset(cti_df, target_feat, selected_space, tmp_dir)
    y = [(k, v) for k in pairs.keys() for v in pairs[k]]
    pred_df = pd.DataFrame(X, index=y)

    # remove tmp directory
    shutil.rmtree(tmp_dir)

    return pred_df

# main        
def main():

    # set title
    st.title("APBIO")
    st.write("**Predicting compound-target interactions**")

    # define sidebar
    st.sidebar.title(
        "APBIO: bioactive profiling of air pollutants via bioactivity "
        "signatures and prediction of target interactions"
    )
    
    st.sidebar.markdown(
        '''<div style='text-align: justify; margin-bottom: 12px;'>
        The APBIO application predicts potential compound-target 
        interactions from compound signatures and target descriptors.</div>
        ''', 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        '''<div style='text-align: justify; margin-bottom: 12px;'>
        Users can input a compound structure as SMILES string and a target 
        protein as UniProtKB identifier to run a prediction.</div>
        ''', 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        '''<ul> Different chemical and bioactivity spaces can be selected
        to compute compound signatures:
        <li><b>F1F2</b>: fingerprint bits and counts;</li>
        <li><b>A1A2</b>: 2D and 3D fingerprints;</li> 
        <li><b>MFP</b>: Morgan fingerprints (1024 bits, radius 2);</li> 
        <li><b>APchem</b>: fingerprint bits and counts, molecular descriptors, 
        and quantum properties;</li> 
        <li><b>CCchem</b>: 2D and 3D fingerprints, scaffolds, 
        structural keys, and physicochemical parameters.</li></ul>
        ''', 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        '''<div style='text-align: justify; margin-bottom: 12px;'>
        Users can download the compound-target interaction dataset,
        visualize the two-dimensional (2D) t-SNE projection of the signatures of the 
        query compound and its 5 nearest neighbors (NN), and retrieve the 
        prediction results.</div>
        ''', 
        unsafe_allow_html=True
    )
     
    st.sidebar.write("#")
    b1, b2 = st.sidebar.columns(2)
    # documentation button
    doc_btn = b1.button("Documentation")
    # home button
    b2.button("Home")

    # define documentation 
    if doc_btn:
        st.subheader("Overview")
        st.markdown(
            '''<div style='text-align: justify; margin-bottom: 12px;'>
            APBIO is a tool developed for predicting compound-target 
            interactions. It computes bioactivity signatures for compounds 
            starting from the SMILES representation and FASTA sequence 
            features for targets starting from the UniProtKB identifier. 
            The prediction is given as a binary outcome of a classifier, 
            where 1 indicates interaction and 0 no interaction, together 
            with the model confidence.</div>
            ''', 
            unsafe_allow_html=True
        )

        st.subheader("Getting started")
        # single interaction 
        st.markdown(
            '''<div style='text-align: justify; font-size: 24px'>
            <b>Single interaction</b></div>
            ''', 
            unsafe_allow_html=True
        )
        st.markdown(
            '''<div style='text-align: justify;>
            If single input is selected, prediction is made for a single
            compound-target interaction:</div>
            ''', 
            unsafe_allow_html=True
        )
        st.markdown(
            '''<ol type="1";>
            <li>Insert the SMILES representation of the compound in the input 
            field and press Enter.</li>
            <li>Insert the UniProtKB identifier of the target in the input
            field and press Enter.</li>
            <li>(Optional) Change the visualization style of the compound 
            structure and the target structure.</li>
            <li>Select one or more options.</li> 
            <li>Click on the "Run" button.</li> 
            <li>Wait for the results to be displayed.</li> 
            <li>Inspect the compound-target interaction dataset and the 
            applicability domain of the compound and the target.</li>
            <li>Visually inspect the 2D t-SNE projection of compound 
            signatures.</li>
            <li>(Optional) Download the compound-target interaction dataset
            as a XLSX file via the "Download datasets" button.</li>
            <li>Inspect the dataframe of prediction results.</li>
            <li>(Optional) Download the prediction results as a XLSX file 
            via the "Download results" button.</li></ol>
            ''', 
            unsafe_allow_html=True
        )

        # multiple interactions
        st.markdown(
            '''<div style='text-align: justify; font-size: 24px'>
            <b>Multiple interactions</b></div>
            ''', 
            unsafe_allow_html=True
        )
        st.markdown(
            '''<div style='text-align: justify;>
            If single multiple interactions is selected, prediction is made 
            for multiple compound-target interactions:</div>
            ''', 
            unsafe_allow_html=True
        )
        st.markdown(
            '''<ol type="1";>
            <li>Click on the "Browse file" button to select a CSV file with 
            multiple compound SMILES and target UniProtKB (as an example, 
            you can use the file in the <i>tests</i> folder). 
            <li>Wait for the results to be displayed.</li> 
            <li>Inspect the resulting dataframe of compound and target 
            identifiers.</li>
            <li>Select one option (default is <i>None</i>).</li> 
            <li>Wait for the results to be displayed.</li>
            <li>Inspect the compound-target interaction dataset.</li>
            <li>(Optional) Download the compound-target interaction dataset 
            as a XLSX file via the "Download datasets" button.
            <li>Inspect the dataframe of prediction results.</li>
            <li>(Optional) Download the prediction results as a XLSX file 
            via the "Download results" button.</ol>
            ''', 
            unsafe_allow_html=True
        )

        # output description
        st.markdown(
            '''<div style='text-align: justify; font-size: 24px; 
            margin-bottom: 12px'><b>Results description</b></div>
            ''', 
            unsafe_allow_html=True
        )
        st.markdown(
            '''<div style='text-align: justify; font-size: 20px'>
            Compound-target interaction dataset</div>
            ''', 
            unsafe_allow_html=True
        )
        st.markdown(
            '''<div style='text-align: justify; margin-bottom: 12px'>
            The compound-target interaction dataset represents the 
            concatenation of the compound signature vector and the 
            target feature vector.</div>
            ''', 
            unsafe_allow_html=True 
        )
        st.markdown(
            '''<div style='text-align: justify; font-size: 20px'>
            Compound and target applicability domain</div>
            ''', 
            unsafe_allow_html=True
        )
        st.markdown(
            '''<div style='text-align: justify; margin-bottom: 12px'>
            Each compound is assigned a flag: "positive" or "negative" if 
            present only in the positive or negative pairs of the training 
            set, respectively, "in" or "new" if present or not in the pairs 
            of the training set, respectively, and "out" if outside the 
            applicability domain. For each compound are reported its 5 nearest 
            neighbors (NN) together with the average cosine distance between
            compound signatures. The same applies to targets. If single 
            interaction is selected, the 2D t-SNE projection 
            of compound signatures is displayed. Each signature vector is 
            colored according to the group it belongs to.</div>
            ''', 
            unsafe_allow_html=True
        ) 
        st.markdown(
            '''<div style='text-align: justify; font-size: 20px'>
            Prediction results</div>
            ''', 
            unsafe_allow_html=True
        )
        st.markdown(
            '''<div style='text-align: justify;'>
            The prediction results are the output of the classification 
            model, where class 0 means no interaction and class 1 means 
            interaction. Each result is provided with its predicition 
            confidence represented by the probability of each class.</div>
            ''', 
            unsafe_allow_html=True
        )
        st.write("#")
    

    # select input type
    input = st.radio("1\. Select the input type to predict a single " 
                     "compound-target interaction or multiple interactions:", 
                     (None, 'Single interaction', 'Multiple interactions'),
                     index=1)
    
    # multiple input
    if input == "Multiple interactions":
        
        # read file 
        st.write("#")
        df = upload_file()

        # parse data 
        space = None
        if df is not None:
            with st.spinner("Wait for data to be parsed..."):
                parsed_df = parse_data(df) 
                if parsed_df is not None:
                    st.success("Done!")
                    st.dataframe(parsed_df, height=200, 
                                 use_container_width=True)    

                    # select space
                    st.write("#")
                    space = st.radio("3\. Select a bioactivity space to "
                                     "generate the compound-target vectors "
                                     "and run the predictions:", 
                                     (None, "F1F2", "A1A2", "MFP", 
                                      "APchem", "CCchem"))

        if space:
            # generate compound-target vectors 
            pred_df = generate_cti_vecs(space, parsed_df)
            # define applicability domain
            ad_dct = search_out_ad(space, pred_df)
            # run predictions 
            result_df = run_prediction(space, pred_df)

            # show dataset
            st.write("#")
            st.subheader("%s"%(space))
            st.markdown(
                '''<div style='text-align: center;'>
                Compound-target interaction vectors</div>
                ''',
                unsafe_allow_html=True
            )
            pred_df.index.name = "Compound - Target"
            st.dataframe(pred_df.style.set_properties(**{"background-color": "#ffe3c9"
                }).format(precision=4), height=150)
        
            # show compounds applicability  
            cols = ["DOMAIN", "5NN_DIST", "5NN_KEYS"]
            mol_applicability_df = pd.DataFrame(list(ad_dct["mol"].values()),
                                                index=list(ad_dct["mol"].keys()))
            mol_applicability_df.columns = cols
            st.markdown(
                '''<div style='text-align: center;'>
                Compounds applicability domain</div>
                ''', 
                unsafe_allow_html=True
            )  
            st.dataframe(mol_applicability_df.style 
                         .set_properties(**{"background-color": "#eeffff"}) 
                         .applymap(lambda x: "color: #ff8b8b" if x=="out" else '', 
                                   subset=["DOMAIN"]) 
                         .format(precision=3, subset=["5NN_DIST"]), height=150)

            # show targets applicability 
            tg_applicability_df = pd.DataFrame(list(ad_dct['tg'].values()),
                index=list(ad_dct['tg'].keys()))
            tg_applicability_df.columns = cols
            st.markdown(
                '''<div style='text-align: center;'>
                Targets applicability domain</div>
                ''', 
                unsafe_allow_html=True
            )
            st.dataframe(tg_applicability_df.style 
                         .set_properties(**{"background-color": "#eeffff"})
                .applymap(lambda x: "color: #ff8b8b" if x=="out" else '', 
                          subset=["DOMAIN"])
                .format(precision=3, subset=["5NN_DIST"]), height=140, 
                use_container_width=True)

            # prepare results 
            pairs = list(result_df.index)
            result_df.reset_index(drop=True, inplace=True)
            # annotation 
            cols = ["InChIKey", "UniProtKB"]
            sorted_df = pd.merge(pd.DataFrame(pairs, columns=cols),
                                 parsed_df, on=cols, how="inner")
            uniprot = sorted_df.pop("UniProtKB")
            sorted_df.insert(3, "UniProtKB", uniprot)
            # applicability
            mol_appl_df = mol_applicability_df. \
                loc[sorted_df.loc[:,"InChIKey"]].reset_index(drop=True)
            mol_appl_df = mol_appl_df.add_prefix("COMPOUND_")           
            tg_appl_df = tg_applicability_df. \
                loc[sorted_df.loc[:,"UniProtKB"]].reset_index(drop=True)
            tg_appl_df = tg_appl_df.add_prefix("TARGET_") 

            # output dataframe 
            out_df = pd.DataFrame(pd.concat([sorted_df, result_df, 
                                             mol_appl_df, tg_appl_df], 
                                             axis=1))
            out_df.index = pairs
            out_df.index.name = "Compound - Target"

            # save dataset
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as w:
                pred_df.to_excel(w, sheet_name=space)
                w.save()
            # download dataset  
            st.download_button(label="Download dataset",
                               data=buffer,
                               file_name = "datasets.xlsx",
                               mime="application/vnd.ms-excel")

            # show results   
            st.write("#")
            st.subheader("Prediction results:")
            # save results
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as w:
                out_df.to_excel(w, sheet_name=space)
                w.save()
            st.markdown(
                f'''<p style='font-size:20px;'>{space}</p>
                ''', 
                unsafe_allow_html=True
            )
            st.dataframe(out_df.iloc[:,5:7].style.applymap(
                lambda x: "background-color: #ffd3c9" if x==0 \
                else "background-color: #e8f9ee", 
                subset=["MODEL_PREDICTION"]) 
                .format(precision=0, subset=["MODEL_PREDICTION"]) 
                .format(precision=3, subset=["MODEL_CONFIDENCE"]),
                height=240, use_container_width=True)
            
            # download results 
            st.download_button(label="Download results",
                               data=buffer,
                               file_name="results.xlsx",
                               mime="application/vnd.ms-excel")

    else: 
        # single input
        st.write("#")
        st.write("2\. Enter compound and target identifiers:")
        b1, b2 = st.columns(2)
        with b1:
            # input compound 
            mol_id = ""
            mol_id = st.text_input("Input a valid compound SMILES:", 
                                   value=st.session_state.input_mol, 
                                   placeholder="C1=CC=CC=C1")
            # hint button 
            st.button("Use hint", key="mol", on_click=get_mol, type="primary")
            if mol_id:
                # select a style 
                mol_style = st.radio("Select a style for visualization:", 
                                     (None, "stick", "sphere"), index=1)
                # change style for visualization
                mol_id, inchikey = change_mol_style(mol_id, mol_style)
                
        with b2:
            # input target 
            tg_id = ""
            tg_id = st.text_input("Input a valid target UniProtKB:",
                                  value=st.session_state.input_tg, 
                                  placeholder="Q9UHW9")
            # hint button 
            st.button("Use hint", key="tg", on_click=get_tg, type="primary")
            if tg_id:
                # select style
                tg_style = st.radio("Select a style for visualization:", 
                                    (None, "cartoon", "stick"), index=1)
                # set style for visualization
                tg_id = change_tg_style(tg_id, tg_style)

        st.write("#")
        if mol_id and tg_id:
            # select space
            spaces = []
            st.write("3\. Select one or more bioactivity spaces to generate "
                     "the compound-target vector and run the prediction:")
            c1, c2, c3, c4, c5, c6, c7 = st.columns(6)
            if c1.checkbox('F1F2')  : spaces.append('F1F2')
            if c2.checkbox('A1A2')  : spaces.append('A1A2')
            if c3.checkbox('MFP')   : spaces.append('MFP')
            if c4.checkbox('APchem'): spaces.append('APchem')
            if c5.checkbox('CCchem'): spaces.append('CCchem')
            
            # define button 
            c6.button("Run", on_click=click_button)
            # define slider
            tsne_option = c7.select_slider("2D t-SNE:", ["Hide", "Show"])
            if st.session_state.clicked:
                # query for compound name
                r_name = pd.read_csv(
                    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound" + 
                    "/smiles/" + mol_id + "/synonyms/TXT", nrows=0
                )
                name = ','.join(list(r_name.columns))

                # query for target symbol
                r_symbol = requests.get("https://rest.uniprot.org/uniprotkb/" + 
                                        tg_id + "?fields=accession%2Cgene_names")
                symbol = ast.literal_eval(r_symbol.text)
                symbol = symbol["genes"][0]["geneName"]["value"]

                datasets = []
                res = []
                for i, space in enumerate(spaces): 
                    # generate compound-target vector 
                    pred_df = generate_cti_vec(space, mol_id, inchikey, tg_id)
                    datasets.append(pred_df)
                    # define applicability domain 
                    ad_dct = search_out_ad(space, pred_df) 
                    applicability_df = pd.concat([
                        pd.DataFrame(list(ad_dct["mol"].values())), 
                        pd.DataFrame(list(ad_dct["tg"].values()))])
                    applicability_df.index = [list(ad_dct["mol"].keys()) + \
                                              list(ad_dct["tg"].keys())]
                    applicability_df.columns = ["DOMAIN", "5NN_DIST", "5NN_KEYS"]
                    # generate projections
                    fig = sign_proj(space, pred_df, ad_dct)
                    # run predictions 
                    result_df = run_prediction(space, pred_df)
                    # generate output dataframe 
                    out_df = pd.DataFrame(pd.concat([result_df.iloc[0],
                                                     applicability_df.iloc[0]
                                                     .add_prefix("COMPOUND_"), 
                                                     applicability_df.iloc[1]
                                                     .add_prefix("TARGET_")])).T
                    out_df.index = [(inchikey, tg_id)]
                    out_df.index.name = "Compound: %s - Target: %s"%(name, symbol) 
                    res.append(out_df)              

                    # show dataset
                    st.write("#")
                    st.subheader("%s"%(space))
                    st.markdown(
                        '''<div style='text-align: center;'>
                        Compound-target interaction vector</div>
                        ''', 
                        unsafe_allow_html=True
                    )
                    pred_df.index.name = "Compound: %s - Target: %s"%(name, symbol)
                    st.dataframe(
                        pred_df.style
                        .set_properties(**{"background-color": "#ffe3c9"})
                        .format(precision=4))
                    
                    # show applicability  
                    st.write("#") 
                    st.markdown(
                        '''<div style='text-align: center;'>
                        Compound and target applicability domain</div>
                        ''', 
                        unsafe_allow_html=True
                    )  
                    st.dataframe(applicability_df.style 
                                 .applymap(lambda _: "background-color: #eeffff",
                                           subset=([inchikey], slice(None))) 
                                 .applymap(lambda _: "background-color: #d4feff",
                                           subset=([tg_id], slice(None))) 
                                 .applymap(lambda x: 'color: #ff8b8b' if \
                                           x=="out" else '', subset=['DOMAIN'])
                                 .format(precision=3, subset=['5NN_DIST']))
                                                                    
                    # show figure 
                    if tsne_option == "Show":
                        # generate projections
                        fig = sign_proj(space, pred_df, ad_dct)
                        st.write("#")
                        st.markdown('''<div style='text-align: center;'>
                            2D t-SNE projection of compound signatures</div>
                            ''', 
                            unsafe_allow_html=True)
                        # configuration
                        config = {
                            'modeBarButtonsToRemove': 
                            ['logo', 'zoom', 'tableRotation','lasso', 'select2d',
                            'autoscale', 'orbitRotation', 'pan', 
                            'resetCameraLastSave'], 'displaylogo': False
                        }
                        st.plotly_chart(fig, config=config, 
                                        use_container_width=True)

                if spaces:
                    # save datasets
                    ds_buffer = io.BytesIO()
                    with pd.ExcelWriter(ds_buffer, engine="xlsxwriter") as w:
                        for i, space in enumerate(spaces):
                            datasets[i].to_excel(w, sheet_name=space)
                    # download datasets   
                    st.download_button(label="Download datasets",
                                       data=ds_buffer,
                                       file_name = "datasets.xlsx",
                                       mime="application/vnd.ms-excel")

                    # save and show results   
                    st.write("#")
                    st.subheader("Prediction results:")
                    res_buffer = io.BytesIO()
                    with pd.ExcelWriter(res_buffer, engine="xlsxwriter") as w:
                        for i, space in enumerate(spaces): 
                            res[i].to_excel(w, sheet_name=space)
                            st.markdown(
                                f'''<p style='font-size:20px;'>{space}</p>''', 
                                unsafe_allow_html=True
                            )
                            st.dataframe(
                                res[i].iloc[:,0:2].style.applymap(
                                    lambda x: "background-color: #ffd3c9" if x==0 \
                                    else "background-color: #e8f9ee", 
                                    subset=["MODEL_PREDICTION"]
                                    ) 
                                .format(precision=0, subset=["MODEL_PREDICTION"]) 
                                .format(precision=3, subset=["MODEL_CONFIDENCE"]),
                                use_container_width=True)
           
                    # download results 
                    st.download_button(label="Download results",
                                       data=res_buffer,
                                       file_name="results.xlsx",
                                       mime="application/vnd.ms-excel")
                    
    # define footer
    st.markdown('''<div class="footer">
            Contact: <b>eva [dot] viesi [at] univr [dot] it</b><br>
            Source code available at: <b>
            <a href=https://github.com/InfOmics/APBIO>
            https://github.com/InfOmics/APBIO</a></b></div>
            ''',
            unsafe_allow_html=True)

# call main
if __name__ == "__main__":
    main()
