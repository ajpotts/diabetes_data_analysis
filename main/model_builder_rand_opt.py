'''
Created on Sep 18, 2023

@author: amandapotts
'''

import os
import sys
import time
import math 
import mlrose
import logging

from six import StringIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pydotplus
import pandas as pd
import numpy as np
from yellowbrick.model_selection import LearningCurve

from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LearningCurveDisplay, ShuffleSplit, learning_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from model_config_parser import ModelConfig
from model_builder import ModelBuilder

import warnings
from pickle import NONE
warnings.filterwarnings("ignore")

NUM_JOBS = -1


class ModelBuilderRandOpt(ModelBuilder):

    def __init__(self, X, y, feature_columns, analysis_dir, model_name, model_config_file, model_out_config="out_config.txt"):
        '''
        Constructor
        '''
        
        super().__init__(X, y, feature_columns, analysis_dir, model_name, model_config_file, model_out_config)

        one_hot = OneHotEncoder()

        self.y_train_hot = one_hot.fit_transform(self.y_train.to_numpy().reshape(-1, 1)).todense()
        self.y_test_hot = one_hot.transform(self.y_test.to_numpy().reshape(-1, 1)).todense()
        
        self.hidden_nodes = [50, 10]
        self.activation = 'relu'
        self.max_iters = 1000
        self.bias = True

        self.clip_max = 10
        self.max_attempts = 100
        self.restarts = 10
        
        self.learning_rates = [0.001, 0.01, 0.1, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1]
        self.population_sizes = [200, 300]
        self.probs = [0.0001, 0.001, 0.01, 0.05, 0.1 ]
        
        self.geom_decay = [0.9, 0.99, 0.999, 0.9999, 0.99999]
        self.arith_decay = [0.01, 0.001, 0.0001]
        self.exp_decay = [0.001, 0.005, 0.05]
    
    def set_config_values(self):
        self.model_config.write_config_value("MODEL", "name", self.model_name)
        self.model_config.write_config_value("OUTPUT", "analysis_dir", self.analysis_dir)
        self.model_config.write_config_value("Training", "num_training_rows", str(self.num_training_rows)) 
        self.model_config.write_config_value("Testing", "num_testing_rows", str(self.num_testing_rows))        
        
    def nn_random_hill_climb_get_best_curve(self):

        best_rate = None
     
        best_cv = None
        
        for rate in self.learning_rates:
            print("Random Hill Climb....")
         
            print("\n\nrate: " + str(rate))
            model = mlrose.NeuralNetwork(hidden_nodes=self.hidden_nodes, activation=self.activation, \
                                     algorithm='random_hill_climb', max_iters=self.max_iters, \
                                     bias=self.bias, is_classifier=True, learning_rate=rate, \
                                     early_stopping=False, clip_max=self.clip_max, max_attempts=self.max_attempts, \
                                     restarts=self.restarts,
                                     curve=True,
                                     random_state=1)
        
            cv = cross_val_score(model, self.X_train, self.y_train_hot, cv=5, scoring='f1_weighted')
            cv_score = sum(cv) / 5
            print("CV : " + str(cv_score))
            
            if(best_cv == None or cv_score > best_cv):
                best_cv = cv_score
           
                best_rate = rate
                
        self.model_config.write_config_value("randomized_hill_climbing", "best_rate", str(best_rate))
        self.model_config.write_config_value("randomized_hill_climbing", "best_cv_score", str(best_cv))   
        
        hc_model = mlrose.NeuralNetwork(hidden_nodes=self.hidden_nodes, activation=self.activation, \
                                     algorithm='random_hill_climb', max_iters=self.max_iters, \
                                     bias=self.bias, is_classifier=True, learning_rate=best_rate, \
                                     early_stopping=False, clip_max=self.clip_max, max_attempts=self.max_attempts, \
                                     restarts=self.restarts,
                                     curve=True,
                                     random_state=1) 
        
        hc_time, hc_curve = self.mlrose_nn(hc_model, "randomized_hill_climbing")
            
        return hc_time, hc_curve
    
    def nn_random_sa_get_best_curve(self):
        
        best_rate = None
        best_decay = None
        best_cv = None
        best_schedule = "arithmetic"
        
        for rate in self.learning_rates:
            for decay in self.arith_decay:
                print("Simulated Annealing....")
                print("decay: " + str(decay))
                print("\n\nrate: " + str(rate))
                model = mlrose.NeuralNetwork(hidden_nodes=self.hidden_nodes, activation=self.activation, \
                                         algorithm='simulated_annealing', max_iters=self.max_iters, \
                                         bias=self.bias, is_classifier=True, learning_rate=rate, \
                                         early_stopping=False, clip_max=self.clip_max, max_attempts=self.max_attempts, \
                                         schedule=mlrose.GeomDecay(decay=decay),
                                        curve=True,
                                         random_state=1)
            
                cv = cross_val_score(model, self.X_train, self.y_train_hot, cv=5, scoring='f1_weighted')
                cv_score = sum(cv) / 5
                print("CV : " + str(cv_score))
                
                if(best_cv == None or cv_score > best_cv):
                    best_cv = cv_score
                    best_decay = decay
                    best_rate = rate
                    
        print("BEST CV: " + str(best_cv))
        self.model_config.write_config_value("simulated_annealing", "best_rate", str(best_rate))
        self.model_config.write_config_value("simulated_annealing", "best_schedule", str(best_schedule))
        self.model_config.write_config_value("simulated_annealing", "best_decay", str(best_decay))
        self.model_config.write_config_value("simulated_annealing", "best_cv_score", str(best_cv))
        
        sa_model = mlrose.NeuralNetwork(hidden_nodes=self.hidden_nodes, activation=self.activation, \
                                     algorithm='simulated_annealing', max_iters=self.max_iters, \
                                     bias=self.bias, is_classifier=True, learning_rate=best_rate, \
                                     early_stopping=False, clip_max=self.clip_max, max_attempts=self.max_attempts, \
                                     schedule=mlrose.GeomDecay(decay=best_decay),
                                    curve=True,
                                     random_state=1)
        
        sa_time, sa_curve = self.mlrose_nn(sa_model, "simulated_annealing")
            
        return sa_time, sa_curve

    def nn_genetic_get_best_curve(self):
        
        best_pop = None
        best_prob = None
        best_cv = None
        
        for pop in self.population_sizes:
            for prob in self.probs:
                    
                print("Genetic....")
    
                print("\n\npop: " + str(pop))
                model = mlrose.NeuralNetwork(hidden_nodes=self.hidden_nodes,
                                             activation=self.activation, \
                                         algorithm='genetic_alg',
                                         max_iters=self.max_iters, \
                                         bias=self.bias,
                                         is_classifier=True,
                                         learning_rate=0.0001, \
                                         early_stopping=False,
                                         clip_max=self.clip_max,
                                         max_attempts=self.max_attempts, \
                                         pop_size=pop,
                                         mutation_prob=prob,
                                         curve=True,
                                         random_state=1) 
            
                cv = cross_val_score(model, self.X_train, self.y_train_hot, cv=5, scoring='f1_weighted')
                cv_score = sum(cv) / 5
                print("CV : " + str(cv_score))
                
                if(best_cv == None or cv_score > best_cv):
                    best_cv = cv_score
                    best_prob = prob
                    best_pop = pop
                
        print("BEST CV: " + str(best_cv))
        self.model_config.write_config_value("genetic_algorithm", "best_pop_size", str(best_pop))
        self.model_config.write_config_value("genetic_algorithm", "best_mut_prob", str(best_prob))        
        self.model_config.write_config_value("genetic_algorithm", "best_cv_score", str(best_cv))
        
        ga_model = mlrose.NeuralNetwork(hidden_nodes=self.hidden_nodes,
                                         activation=self.activation, \
                                     algorithm='genetic_alg',
                                     max_iters=self.max_iters, \
                                     bias=self.bias,
                                     is_classifier=True,
                                     learning_rate=0.0001, \
                                     early_stopping=False,
                                     clip_max=self.clip_max,
                                     max_attempts=self.max_attempts, \
                                     pop_size=best_pop,
                                    mutation_prob=best_prob,
                                     curve=True,
                                     random_state=1)
        
        sa_time, sa_curve = self.mlrose_nn(ga_model, "genetic_algorithm")
            
        return sa_time, sa_curve
    
    def nn_random_opt(self):
        
        # hc_time, hc_curve = self.nn_random_hill_climb_get_best_curve()   
        #
        sa_time, sa_curve = self.nn_random_sa_get_best_curve() 
        
        gradient_descent_model = mlrose.NeuralNetwork(hidden_nodes=self.hidden_nodes, activation=self.activation, \
                                     algorithm='gradient_descent', max_iters=self.max_iters, \
                                     bias=self.bias, is_classifier=True, learning_rate=0.0001, \
                                     early_stopping=False, clip_max=self.clip_max, max_attempts=self.max_attempts, \
                                    curve=True,
                                     random_state=1)
        
        gradient_descent_time, gradient_descent_curve = self.mlrose_nn(gradient_descent_model, "gradient_descent")

        print("\n\nStarting Genetic Algorithm:")  
        
        ga_time, ga_curve = self.nn_genetic_get_best_curve()

        learning_curve_filename = self.analysis_dir + "rand_opt_performance.png"
        runtime_filename = self.analysis_dir + "rand_opt_run_time.png"

        blue_patch = mpatches.Patch(color='blue', label="Simulated Annealing")
        green_patch = mpatches.Patch(color='green', label="Randomized Hill Climbing")
        red_patch = mpatches.Patch(color='red', label="Gradient Descent")
        purple_patch = mpatches.Patch(color='purple', label="Genetic Algorithm")
          
        self.clear_plots()                                   
        ax = plt.gca()
        plt.plot(sa_curve, color='blue')
        plt.plot(hc_curve, color='green')       
        plt.plot(gradient_descent_curve, color='red')       
        plt.plot(ga_curve, color='purple')          
        plt.legend(handles=[blue_patch, green_patch, red_patch, purple_patch], loc="lower right") 
        plt.xlabel("Iterations")
        plt.ylabel("Fitness Score")
        plt.title("Fitness by Number of Iterations for Each Algorithm")
        plt.savefig(learning_curve_filename)  
        # plt.show()          
        
        self.clear_plots()  
        ax = plt.gca()
        fig = plt.figure(figsize=(10, 5))
        algorithms = ["Simulated Annealing", "Randomized Hill Climbing", "Gradient Descent", "Genetic Algorithm"]
        times = [sa_time, hc_time, gradient_descent_time, ga_time]
        
        # creating the bar plot
        plt.bar(algorithms, times, color='maroon',
        width=0.4)
 
        plt.xlabel("Algorithm")
        plt.ylabel("Running Time (Seconds)")
        plt.title("Running Time For Each Algorithm (Seconds)")
        plt.savefig(runtime_filename)

    def mlrose_nn(self, model, model_type):

        start_time = time.time()
        model.fit(self.X_train, self.y_train_hot)
        end_time = time.time() - start_time
        
        self.model_config.write_config_value(model_type, "training_time", str(end_time))
        
        # Predict labels for train set and assess accuracy
        y_train_pred = model.predict(self.X_train)
        
        y_train_accuracy = accuracy_score(self.y_train_hot, y_train_pred)
        self.model_config.write_config_value(model_type, "training_accuracy", str(y_train_accuracy))
        
        # F1
        training_f1 = metrics.f1_score(self.y_train_hot, y_train_pred, average='weighted')
        self.model_config.write_config_value(model_type, "training_weighed_f1", str(training_f1))
        
        # Predict labels for test set and assess accuracy
        y_test_pred = model.predict(self.X_test)
        
        y_test_accuracy = accuracy_score(self.y_test_hot, y_test_pred)
        self.model_config.write_config_value(model_type, "test_accuracy", str(y_test_accuracy))
        
        percision = metrics.precision_score(self.y_test_hot, y_test_pred, average='weighted')
        self.model_config.write_config_value(model_type, "test_weighed_percision", str(percision))
        
        # Model Recall
        recall = metrics.recall_score(self.y_test_hot, y_test_pred, average='weighted')
        self.model_config.write_config_value(model_type, "test_weighed_recall", str(recall))
        
        # F1
        test_f1 = metrics.f1_score(self.y_test_hot, y_test_pred, average='weighted')
        self.model_config.write_config_value(model_type, "test_weighed_f1", str(test_f1))
        
        return end_time, model.fitness_curve
    
