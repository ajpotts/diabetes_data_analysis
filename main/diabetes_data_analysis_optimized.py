#!/usr/local/bin/python2.7
# encoding: utf-8
'''
@author:     Amanda Potts

@deffield    updated: October 2023
'''

import os
import sys
import pathlib
from pathlib import Path
import six
sys.modules['sklearn.externals.six'] = six
from model_builder_rand_opt import ModelBuilderRandOpt

from diabetes_data_analysis import get_diabetes_presence_data


def main():
    
    # Get path for analysis to be stored
    path = pathlib.Path().resolve()
    project_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute())
    analysis_dir = str(path) + '/analysis'

    # Get data set and analyze
    X, y, feature_columns = get_diabetes_presence_data(project_dir, analysis_dir)

    config_file_in = project_dir + '/default_config.txt'    
    config_file_out = project_dir + '/diabetes_presence_rand_opt_config.txt'
    model_builder_diabetes_presence = ModelBuilderRandOpt(X, y, feature_columns, analysis_dir, 'diabetes_presence', config_file_in, model_out_config=config_file_out)
    
    model_builder_diabetes_presence.nn_random_opt()
    model_builder_diabetes_presence.write_config()


if __name__ == "__main__":

    sys.exit(main())
