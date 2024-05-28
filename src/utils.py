
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

import mlflow
import pickle

from src.exception import CustomException
from src.logger import logging


# function to save an object into a pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.info("EXCEPTION OCCURRED IN UTLIS.PY")
        raise CustomException(e, sys)

 
# function to load an object from a pickle file
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info("EXCEPTION OCCURRED WHILE LOADING PICKLE FILE")
        raise CustomException(e, sys)


# fundtion to train test and evaluate the models over the data
def train_and_evaluate_model(X_train, y_train, X_test, y_test, models) -> dict:

    # training per cluster and testing per model
    cluster_size = len(X_train)//6    
    clusters_X = [X_train[i * cluster_size: (i + 1) * cluster_size] for i in range(6)]
    clusters_y = [y_train[i * cluster_size: (i + 1) * cluster_size] for i in range(6)]

    report = dict()
    for i in range(0,len(models)):
        model = list(models.values())[i]
        model.fit(clusters_X[i], clusters_y[i])
        y_pred = model.predict(X_test)
        report.update({model : accuracy_score(y_test,y_pred)})

    logging.info(f"Model Training Completed. Respective accuracies attained by following models : \n {report}")

    best_model_score = max(sorted(report.values()))

    best_model = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]
    
    # hyperparameter tuning
    param_dict = {
    'n_estimators' : [100, 200, 300, 400, 500],
    'criterion' : ["gini", "entropy", "log_loss"],
    'max_depth' : [3, 10, 13, 20, 23, 30, 33, 40, 42],
    'min_samples_leaf' : [3, 10, 13, 20, 23, 30, 33, 40, 42],
    'max_features' : ["sqrt", "log2"],
    'n_jobs' : [-1],
    'random_state' : [3, 10, 13, 20, 23, 30, 33, 40, 42]
    }
    random_search = RandomizedSearchCV(
        estimator=best_model,
        param_distributions=param_dict,
        n_iter=25,
        n_jobs=-1,
        cv=5,
        verbose=1,
        random_state=42
    )
    random_search.fit(clusters_X[0], clusters_y[0])

    return best_model_score, random_search.best_estimator_