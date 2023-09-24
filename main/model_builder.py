'''
Created on Sep 18, 2023

@author: amandapotts
'''

import os
import sys
import time
import math 

from six import StringIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pydotplus
import pandas as pd
import numpy as np
from yellowbrick.model_selection import LearningCurve

from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LearningCurveDisplay, ShuffleSplit, learning_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle

from model_config_parser import ModelConfig

import warnings
warnings.filterwarnings("ignore")

NUM_JOBS = -1


class ModelBuilder(object):

    def __init__(self, X, y, feature_columns, analysis_dir, model_name, model_config_file):
        '''
        Constructor
        '''
        
        self.model_config = ModelConfig(model_config_file)
        
        self.analysis_dir = analysis_dir + '/' + model_name + '/'
        self.model_name = model_name
        
        try:
            os.makedirs(self.analysis_dir, exist_ok=True)
            print("Directory '%s' created successfully" , self.analysis_dir)
        except OSError as error:
            print("Directory '%s' can not be created", self.analysis_dir)

        self.X = X
        self.y = y
        self.feature_columns = feature_columns
        
        self.X, self.y = shuffle(self.X, self.y, random_state=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=1)  # 70% training and 30% test        
        
        sc = StandardScaler()
        
        scaler = sc.fit(self.X_train)
        self.X_train_scaled = scaler.transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        self.X_train = self.X_train_scaled
        self.X_test = self.X_test_scaled
        
        self.num_training_rows = self.X_train.shape[0]
        self.num_testing_rows = self.X_test.shape[0]
    
    def set_config_values(self):
        self.model_config.write_config_value("MODEL", "name", self.model_name)
        self.model_config.write_config_value("OUTPUT", "analysis_dir", self.analysis_dir)
        self.model_config.write_config_value("Training", "num_training_rows", str(self.num_training_rows)) 
        self.model_config.write_config_value("Testing", "num_testing_rows", str(self.num_testing_rows))        
    
    def write_config(self):
        self.set_config_values()
        self.model_config.write(open(self.analysis_dir + "out_config.txt", 'w'), space_around_delimiters=False)
    
    def decision_tree(self):
        
        feature_cols = self.feature_columns

##################################################################
##################################################################
##################################################################
###                                                            ###
###      Implement Grid Search                                 ###
###                                                            ###
##################################################################
##################################################################
##################################################################

        param_grid = {
            'criterion':['gini', 'entropy', 'log_loss'],
            'splitter':['best', 'random'],
            'max_features': ['sqrt', 'log2'],
            'max_depth':[5, 10, 20, 30, 40, 50, 100, 500],
            'min_samples_split':[5, 10, 20, 40, 60, 80, 100],
            'min_impurity_decrease': [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
            'max_leaf_nodes':[10, 20, 40, 60, 80, 100, 200]
            }

        model_type = "Decision Tree"        
        gridSearch = self.grid_search(DecisionTreeClassifier(random_state=1), param_grid, model_type)

##################################################################
##################################################################
##################################################################
###                                                            ###
###      Train Grid Search Tree                                ###
###                                                            ###
##################################################################
##################################################################
##################################################################

        gridsearch_tree = DecisionTreeClassifier(criterion=gridSearch.best_params_["criterion"],
                                        splitter=gridSearch.best_params_["splitter"],
                                        min_samples_split=gridSearch.best_params_["min_samples_split"],
                                        max_features=gridSearch.best_params_["max_features"],
                                        min_impurity_decrease=gridSearch.best_params_["min_impurity_decrease"],
                                        max_leaf_nodes=gridSearch.best_params_["max_leaf_nodes"],
                                        max_depth=gridSearch.best_params_["max_depth"],
                                        random_state=1)   
        
        gridsearch_y_pred = self.fit_model(gridsearch_tree , model_type, "gridsearch")
        self.write_decision_tree_graph(gridsearch_tree , self.analysis_dir + self.model_name + '_gridsearch_decision_tree_graph.png', feature_cols=feature_cols)
                
##################################################################
##################################################################
##################################################################
###                                                            ###
###      Train Full Tree                                       ###
###                                                            ###
##################################################################
##################################################################
##################################################################        
        
        fulltree = DecisionTreeClassifier(criterion=gridSearch.best_params_["criterion"],
                                        splitter=gridSearch.best_params_["splitter"],
                                        min_samples_split=gridSearch.best_params_["min_samples_split"],
                                        max_features=gridSearch.best_params_["max_features"],
                                        min_impurity_decrease=gridSearch.best_params_["min_impurity_decrease"],
                                        max_leaf_nodes=gridSearch.best_params_["max_leaf_nodes"],
                                        random_state=1)
        
        fulltree_y_pred = self.fit_model(fulltree, model_type, "full")
        
    
          
        self.write_decision_tree_graph(fulltree, self.analysis_dir + self.model_name + '_full_decision_tree_graph.png', feature_cols=feature_cols)     
        
##################################################################
##################################################################
##################################################################
###                                                            ###
###      Train Pruned Tree                                     ###
###                                                            ###
##################################################################
##################################################################
##################################################################        
        
        # prune the tree with cost complexity pruning — Alpha
        path = fulltree.cost_complexity_pruning_path(self.X_train, self.y_train)
        alphas, impurities = path.ccp_alphas, path.impurities
        mean, std = [], []
        for i in alphas:
            tree = DecisionTreeClassifier(ccp_alpha=i, random_state=0)
            # 5 fold cross validation for each alpha value
            scores = cross_val_score(tree, self.X_train, self.y_train, cv=5)
            mean.append(scores.mean())
            std.append(scores.std())
        
        # keep a record of the values of alpha, mean accuracy rate, standard deviation of accuracies
        eva_df = pd.DataFrame({'alpha': alphas, 'mean': mean, 'std': std})
        eva_df = eva_df.sort_values(['mean'], ascending=False)
        eva_df.head(10)
        print(eva_df.head(10))
        
        best_alpha = eva_df['alpha'][0]
        print("BEST Alpha: " + str(best_alpha))

        pruned_tree = DecisionTreeClassifier(criterion=gridSearch.best_params_["criterion"],
                                        splitter=gridSearch.best_params_["splitter"],
                                        min_samples_split=gridSearch.best_params_["min_samples_split"],
                                        max_features=gridSearch.best_params_["max_features"],
                                        min_impurity_decrease=gridSearch.best_params_["min_impurity_decrease"],
                                        max_leaf_nodes=gridSearch.best_params_["max_leaf_nodes"],
                                        ccp_alpha=0.01,
                                        random_state=1)
        
        prunedtree_y_pred = self.fit_model(pruned_tree, model_type, "pruned")
          
        self.write_decision_tree_graph(pruned_tree, self.analysis_dir + self.model_name + '_pruned_decision_tree_graph.png', feature_cols=feature_cols)
        
        self.write_config()
                
    def svm(self):
        
        model_type = "SVM"      
               
        param_grid = {'C': [0.001, 0.05, 0.1, 1],
               'kernel': ['rbf', 'linear', 'poly'],
              'degree':[2, 3, 4]
              } 
   
        gridSearch = self.grid_search(SVC(random_state=1), param_grid, model_type)
        
        gridsearch_y_pred = self.fit_model(gridSearch, model_type, "gridsearch")

        #  linear kernel
        linear_kernel = SVC(kernel='linear', random_state=1)
        linear_y_pred = self.fit_model(linear_kernel, model_type, 'linear')
        
        #  rbf kernel
        rbf_kernel = SVC(kernel='rbf', random_state=1)
        rbf_y_pred = self.fit_model(rbf_kernel, model_type, 'rbf')
        
        self.write_config()
        
        return gridsearch_y_pred

    def neural_network(self):
        
        model_type = "Neural Network"
        
        MAX_ITER = 1000
        
        mlp_gs = MLPClassifier(max_iter=MAX_ITER, early_stopping=False, random_state=1)
        
        param_grid = {
            'hidden_layer_sizes': [(10, 30, 10), (10, 10, 10), (50, 10), (30, 10), (20,), (50,)],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            'learning_rate': ['constant', 'adaptive', 'invscaling'],
            }
        
        gridSearch = self.grid_search(mlp_gs, param_grid, model_type)
        
        gridsearch_early_stopping = MLPClassifier(hidden_layer_sizes=gridSearch.best_params_["hidden_layer_sizes"],
                                        activation=gridSearch.best_params_["activation"],
                                        solver=gridSearch.best_params_["solver"],
                                        alpha=gridSearch.best_params_["alpha"],
                                        learning_rate=gridSearch.best_params_["learning_rate"],
                                        max_iter=MAX_ITER,
                                        early_stopping=True,
                                        random_state=1)
        
        self.fit_model(gridsearch_early_stopping, model_type, "mlp_early_stopping")
        self.model_config.write_config_value(model_type, "gridsearch_earlystopping_model_num_iterations", str(gridsearch_early_stopping.n_iter_))
        
        gridsearch_best = MLPClassifier(hidden_layer_sizes=gridSearch.best_params_["hidden_layer_sizes"],
                                        activation=gridSearch.best_params_["activation"],
                                        solver=gridSearch.best_params_["solver"],
                                        alpha=gridSearch.best_params_["alpha"],
                                        learning_rate=gridSearch.best_params_["learning_rate"],
                                        max_iter=MAX_ITER,
                                        random_state=1)
        
        gridsearch_y_pred = self.fit_model(gridsearch_best, model_type, "mlp")
        
        print("Best NN Grid Search Model, NUM ITER: " + str(gridsearch_best.n_iter_))
        
        self.model_config.write_config_value(model_type, "gridsearch_model_num_iterations", str(gridsearch_best.n_iter_))     
        
        self.write_config()   
        
        return gridsearch_y_pred
        
    def knn(self):
        
        model_type = "K Nearest Neighbors"
        
        knn = KNeighborsClassifier()
        
        param_grid = {
            "n_neighbors":range(0, 50),
            "weights":["uniform", "distance"],
            "p":[1, 2, 3, 4, 5]
            }

        gridSearch = self.grid_search(knn, param_grid, model_type)
        
        gridsearch_y_pred = self.fit_model(gridSearch, model_type, "knn")
        
        self.write_config()
        
        return gridsearch_y_pred    
    
    def boosting(self):

        model_type = "Boosting"
        
        NUM_ESTIMATORS = 500
        self.model_config.write_config_value(model_type, "num_iterations", str(NUM_ESTIMATORS))   

        boost = GradientBoostingClassifier(n_estimators=NUM_ESTIMATORS, random_state=1)
        
        param_grid = {
            "learning_rate": [0.01, 0.1, 1],
            "max_leaf_nodes": [4, 10, 20, 50],
            "max_depth": [5, 10, 25, 50],
            "min_samples_leaf": [5, 10, 20, 40, 60, 80, 100],
            "ccp_alpha":[0.001, 0.01, 0.1, 1]
        }

        gridSearch = self.grid_search(boost, param_grid, model_type)

        gridsearch_best = GradientBoostingClassifier(
                                        n_estimators=NUM_ESTIMATORS,
                                        learning_rate=gridSearch.best_params_["learning_rate"],
                                        max_leaf_nodes=gridSearch.best_params_["max_leaf_nodes"],
                                        max_depth=gridSearch.best_params_["max_depth"],
                                        min_samples_leaf=gridSearch.best_params_["min_samples_leaf"],
                                        ccp_alpha=gridSearch.best_params_["ccp_alpha"],
                                        random_state=1)
        
        gridsearch_y_pred = self.fit_model(gridsearch_best, model_type, "gridsearch")
        
        self.write_config()
        
        return gridsearch_y_pred 

    def grid_search(self, model, param_grid, model_type):
        print("Starting Grid Search for " + model_type)
        print("Using parameter grid: " + str(param_grid))
        
        cross_validation_n = 5
        
        self.model_config.write_config_value(model_type, "parameter_grid" , str(param_grid))
        self.model_config.write_config_value(model_type, "num_jobs" , str(NUM_JOBS))
        self.model_config.write_config_value(model_type, "cross_validation_n" , str(cross_validation_n))
             
        gridSearch = GridSearchCV(model, param_grid, cv=cross_validation_n, n_jobs=NUM_JOBS, refit=True)
        
        start_time = time.time()
        gridSearch.fit(self.X_train, self.y_train)
        end_time = time.time()
        self.model_config.write_config_value(model_type, "grid_search_time" , str(end_time - start_time))
        
        print('Grid Search score: ', gridSearch.best_score_)
        self.model_config.write_config_value(model_type, "grid_search_best_score" , str(gridSearch.best_score_))

        print('Grid Search estimator: ', str(gridSearch.best_estimator_))
        self.model_config.write_config_value(model_type, "grid_search_best_estimator" , str(gridSearch.best_estimator_))
        
        print('Grid Search parameters: ', gridSearch.best_params_)
        for key in gridSearch.best_params_:
            self.model_config.write_config_value(model_type, "grid_search_best_" + key, str(gridSearch.best_params_[key]))
            
        self.write_config()
            
        return gridSearch
    
    def clear_plots(self):
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

    def fit_model(self, model, model_type, model_subtype):
        
        start_time = time.time()
        model.fit(self.X_train, self.y_train)  # plot the tree
        end_time = time.time()
        self.model_config.write_config_value(model_type, model_subtype + "_training_time" , str(end_time - start_time))
                
        # Predict the response for test dataset
        y_pred = model.predict(self.X_test)
        
        # Model Accuracy
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        print(model_type + " " + model_subtype + " Accuracy:", str(accuracy)) 
        self.model_config.write_config_value(model_type, model_subtype + "_accuracy" , str(accuracy))
        
        # Model Precision
        percision = metrics.precision_score(self.y_test, y_pred, average='macro')
        print(model_type + " " + model_subtype + " Precision:", str(percision))
        self.model_config.write_config_value(model_type, model_subtype + "_precision" , str(percision))
        
        # Model Recall
        recall = metrics.recall_score(self.y_test, y_pred, average='macro')
        print(model_type + " " + model_subtype + " Recall:", str(recall))
        self.model_config.write_config_value(model_type, model_subtype + "_recall" , str(recall))
        
        print(classification_report(self.y_test, y_pred))
        report = classification_report(self.y_test, y_pred)
        report_filename = self.analysis_dir + "/classification_report_" + model_type.replace(" ", "_") + "_" + model_subtype + ".txt"
        
        self.clear_plots()
        
        loss_curve_filename = self.analysis_dir + "/loss_curve_" + model_type.replace(" ", "_") + "_" + model_subtype + ".png"

        if(model_type == "Neural Network"):
            blue_patch = mpatches.Patch(color='blue', label="Training Loss")
            green_patch = mpatches.Patch(color='green', label="Validation Accuracy")
            if(model.best_validation_score_ is not None):
                ax = plt.gca()
                y_min = min(min(model.validation_scores_), min(model.loss_curve_)) - 0.1
                y_max = max(max(model.validation_scores_), max(model.loss_curve_)) + 0.1
                ax.set_ylim([y_min, y_max ])
                ax.set_xlim([0, model.n_iter_])
                plt.plot(model.loss_curve_, color='blue')
                plt.plot(model.validation_scores_, color='green')
                plt.legend(handles=[blue_patch, green_patch], loc="upper right")
            
            else:
                plt.plot(model.loss_curve_, color='blue')
                plt.legend(handles=[blue_patch], loc="upper right")
            
            plt.title("Loss Curve", fontsize=14)
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.savefig(loss_curve_filename)
        
        if(model_type == "Boosting"):
            blue_patch = mpatches.Patch(color='blue', label="In-Bag Training Loss")

            plt.plot(model.train_score_, color='blue')
            plt.legend(handles=[blue_patch], loc="upper right")
            plt.title("Loss Curve", fontsize=14)
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.savefig(loss_curve_filename)
        
        cm = confusion_matrix(self.y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=model.classes_)
        
        confusion_matrix_filename = self.analysis_dir + "/confusion_matrix_" + model_type.replace(" ", "_") + "_" + model_subtype + ".png"
        disp.plot().figure_.savefig(confusion_matrix_filename , dpi=300)
 
        with open(report_filename, "w") as text_file:
            text_file.write(report + "\n\n" + str(cm))
        
        self.clear_plots()
        learning_curve_filename = self.analysis_dir + "/learning_curve_" + model_type.replace(" ", "_") + "_" + model_subtype + ".png"
        visualizer = LearningCurve(
            model,
            cv=ShuffleSplit(n_splits=50, test_size=0.2, random_state=1),
            scoring='f1_weighted',
            train_sizes=np.linspace(0.1, 1.0, 5),
            n_jobs=4,
            random_state=1
            )

        visualizer.fit(self.X, self.y)  # Fit the data to the visualizer
        visualizer.show(outpath=learning_curve_filename)  # Finalize and render the figure
        
        return y_pred
    
    def write_decision_tree_graph(self, tree, file_path, feature_cols): 
        
        dot_data = StringIO()
        export_graphviz(tree, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True , feature_names=feature_cols)
        
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        graph.write_png(file_path)        
