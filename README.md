# APBIO
APBIO is a tool developed to predict compound-target interactions, with a 
specific focus on air pollutants and their bioactivity.
It computes bioactivity signatures for compounds starting from the SMILES 
representation (e.g., C1=CC=CC=C1) and FASTA sequence features for targets 
starting from the UniProtKB identifier (e.g., Q9UHW9). 


### Compound bioactivity signatures
Bioactivity signatures are computed via the _**signaturizer**_ package. 
For further details please visit the Chemical Checker (CC)
<a href="https://doi.org/10.1038/s41587-020-0502-7" target="_blank">paper</a>, 
the CC signaturizers <a href="https://doi.org/10.1038/s41467-021-24150-4" target="_blank">paper</a>, 
and the relative <a href="https://gitlabsbnb.irbbarcelona.org/packages" target="_blank">repositories</a>. 


### Target sequence descriptors
Sequence descriptors are calculated via the _**iFeature**_ toolkit. Specifically, 
we use the main _iFeature.py_ program and the required files in the 
_codes_ and _data_ folders. Additional information is provided in the iFeature 
<a href="https://doi.org/10.1093/bioinformatics/bty140 " target="_blank">paper</a> 
and the relative <a href="https://github.com/Superzchen/iFeature" target="_blank">repository</a>. 


### Configuration 
To run notebooks and reproduce results, you can clone this repo and set up a 
conda environment using the code snippet below:
```
$ conda create --no-default-packages -n cti -y python=3.7.16
$ conda activate cti
$ pip install -r requirements.txt
```
The main methodology can be executed via the _APBIO_pipeline.ipynb_ notebook. 

### Additional material
The datasets and additional materials related to this work can be found 
<a href="https://univr-my.sharepoint.com/:f:/g/personal/eva_viesi_univr_it/EnxlMHBp2AxNtw9XyK6c7L8Bw7TicwZaJYUU9ll1vO5GMA?e=YNOeae" target="_blank">here</a>.
If you want to perform the sampling strategy evaluation, please download and place 
the _sampled_ and _random_ folders in the following path: `/cti_datasets/AP_CTIs/`. 


### Web application
The Streamlit web app is available at: https://ap-bio.streamlit.app/.