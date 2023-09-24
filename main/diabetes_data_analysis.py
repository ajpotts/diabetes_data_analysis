#!/usr/local/bin/python2.7
# encoding: utf-8
'''
@author:     Amanda Potts

@deffield    updated: September 2023
'''

import os
import sys
import pathlib
from pathlib import Path
import pandas as pd
from model_builder import ModelBuilder

MAX_ROWS = 3000


def main():
    
    # Get path for analysis to be stored
    path = pathlib.Path().resolve()
    project_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute())
    analysis_dir = str(path) + '/analysis'

    # Get first data set and analyze
    X, y, feature_columns = get_diabetes_presence_data(project_dir, analysis_dir)
    
    config_file_diabetes_presence = project_dir + '/diabetes_presence_config.txt'
    model_builder_diabetes_presence = ModelBuilder(X, y, feature_columns, analysis_dir, 'diabetes_presence', config_file_diabetes_presence)
    
    model_builder_diabetes_presence.decision_tree()
    model_builder_diabetes_presence.svm()
    model_builder_diabetes_presence.knn()
    model_builder_diabetes_presence.boosting()
    model_builder_diabetes_presence.neural_network()
    model_builder_diabetes_presence.write_config()

    #  Get second data set and analyze
    X, y , feature_columns = get_diabetes_readmittance_data(project_dir, analysis_dir)   
     
    config_file_diabetes_readmittance = project_dir + '/diabetes_readmittance_config.txt'
    model_builder_diabetes_readmittance = ModelBuilder(X, y, feature_columns, analysis_dir, 'diabetes_readmittance', config_file_diabetes_readmittance)
    model_builder_diabetes_readmittance.decision_tree()
    model_builder_diabetes_readmittance.svm()
    model_builder_diabetes_readmittance.knn()
    model_builder_diabetes_readmittance.boosting()
    model_builder_diabetes_readmittance.neural_network()
    model_builder_diabetes_readmittance.write_config()


def get_diabetes_readmittance_data(project_dir, analysis_dir):
    
    data = project_dir + "/data/diabetes_readmittance/diabetic_data.csv"    
    df = pd.read_csv(data, sep=',')
    
    eda(df, analysis_dir, "diabetes_readmittance")
    
    drop_list = ['encounter_id',
                 'weight',
                'diag_1',
                'diag_2',
                'diag_3',
                'max_glu_serum',
                'A1Cresult',
                'repaglinide',
                'nateglinide',
                'chlorpropamide',
                'glimepiride',
                'acetohexamide',
                'glipizide',
                'glyburide',
                'tolbutamide',
                'pioglitazone',
                'rosiglitazone',
                'acarbose',
                'miglitol',
                'troglitazone',
                'tolazamide',
                'examide',
                'citoglipton',
                'glyburide-metformin',
                'glipizide-metformin',
                'glimepiride-pioglitazone',
                'metformin-rosiglitazone',
                'metformin-pioglitazone',
                ]

    df = df.drop(drop_list, axis=1)
    
    ohe_columns = ['gender',
                    'race',
                    'age',
                    'admission_type_id',
                    'discharge_disposition_id',
                    'admission_source_id',
                    'payer_code',
                    'medical_specialty',
                    'metformin',
                    'insulin',
                    'change',
                    'diabetesMed'
                    ]

    df = df[(df['readmitted'] == "NO") | (df['readmitted'] == "<30")]
    
    df = df.drop_duplicates(subset=['patient_nbr'])
    df = df.drop(['patient_nbr'], axis=1)
    
    transformed_df = pd.get_dummies(df, columns=ohe_columns, drop_first=True)

    transformed_df = transformed_df.groupby('readmitted').sample(n=8627, random_state=1)
    
    if(df.shape[0] > MAX_ROWS):
        transformed_df = transformed_df.sample(n=MAX_ROWS, random_state=1)

    X = transformed_df.drop(['readmitted'], axis=1)  # Features
    y = transformed_df['readmitted']  # Target variable
    
    feature_columns = X.columns    
    
    return X, y, feature_columns


def get_diabetes_presence_data(project_dir, analysis_dir):
    
    data = project_dir + "/data/diabetes_presence/diabetes_data.csv"    
    df = pd.read_csv(data, sep=';')
    
    eda(df, analysis_dir, "diabetes_presence")
    
    ohe_columns = [ 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring', 'itching', 'irritability', 'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'alopecia', 'obesity']

    transformed_df = pd.get_dummies(df, columns=ohe_columns, drop_first=True)

    X = transformed_df.drop(['class'], axis=1)  # Features
    y = transformed_df['class']  # Target variable
    
    feature_columns = X.columns
    
    return X, y, feature_columns


def eda(df, analysis_dir, modelname):
    
    eda_dir = analysis_dir + "/" + modelname 
    
    try:
        os.makedirs(eda_dir, exist_ok=True)
        print("Directory '%s' created successfully" , eda_dir)
    except OSError as error:
        print("Directory '%s' can not be created", eda_dir)
    
    eda_filename = eda_dir + "/eda.txt"   
    
    with open(eda_filename, "w") as text_file:
        
        text_file.write("Original Data size (num Rows): " + str(df.shape[0]) + "\n\n")
        text_file.write("Original Data size (num Columns): " + str(df.shape[1]) + "\n\n")
                      
        text_file.write(str(df.columns.tolist()))
        text_file.write("\n\n")
        
        for column in df.columns:
            
            text_file.write("\n\n")
            grouped_df = df.groupby([column]).size()
            text_file.write(str(grouped_df))
            text_file.write("\n\n")
            text_file.write(str(grouped_df / grouped_df.sum()))


if __name__ == "__main__":

    sys.exit(main())
