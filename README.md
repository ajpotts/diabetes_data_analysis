# Project
Author: Amanda Potts
Fall 2023


## Setup

python3 is required to run this project.

Download the code repository from github:
> git clone git@github.com:ajpotts/diabetes_data_analysis.git

It is recommended to install python eviromnent.  On Ubuntu, it can be installed with:
> sudo apt install python3.10-venv

Install graphviz.  On Ubuntu, it can be installed with:
> sudo apt-get install graphviz

To start python environment:
> bash create_env.sh 
> source py-env/bin/activate

To install requirments without python environment (use at your own risk):
> pip install -r requirements.txt



## Download the data

Download the Presence Dataset from here:
https://www.kaggle.com/datasets/andrewmvd/early-diabetes-classification

And unzip the data into this folder:
> data/diabetes_presence

Download the Readmittance Dataset from here:
https://www.kaggle.com/datasets/sulphatet/diabetes-130us-hospitals-for-years-19992008?select=diabetic_data.csv

And unzip the data into this folder:
> data/diabetes_readmittance



## To Run

> cd main/
> python3 diabetes_data_analysis.py

## Output

Analysis files will be written to two directories.  

The Presence Dataset results will be written here:
> analysis/diabetes_presence

The Readmittance Dataset results will be written here:
> analysis/diabetes_readmittance

## Citations

Code samples for grid search over cost-complexity parameters:

Joann. (2021, March 25). \emph/{Classification Tree Growing and Pruning with Python Code (Grid Search & Cost-Complexity Function)} Medium.com. https://vzhang1999.medium.com/classification-tree-growing-and-pruning-with-python-code-grid-search-cost-complexity-function-b2e45e33a1a4


