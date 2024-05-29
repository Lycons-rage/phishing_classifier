
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
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
def train_and_evaluate_model(X_train, y_train, X_test, y_test, models, run_id) -> dict:
    
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

    # registering and logging the best model got after performing hyperparameter tuning using randomized search cv
    model_name = random_search.best_estimator_.__class__.__name__
    mlflow.sklearn.log_model(random_search.best_estimator_, model_name)
    model_uri = f"run:/{run_id}/{model_name}"
    mlflow.register_model(model_uri=model_uri, name=model_name)

    log_params(random_search.best_params_)
    log_metrics(best_model_score)

    return best_model_score, random_search.best_estimator_


# logging parameters using mlflow
def log_params(parameters:dict):
    for key in parameters.keys():
        mlflow.log_param(key, parameters[key])


# logging metrics using mlflow
def log_metrics(score):
    mlflow.log_metric("Accuracy - ",score)


# load the preprocessor as well as model object
def load_object(file_path):
    try:
        with open(file_path, "r+") as op:
            preprocessor_pickel = pickle.load(op)

        try:
            mlflow_client = mlflow.tracking.MlflowClient()
            registered_model = mlflow_client.list_registered_models()
            model_name = registered_model[0].name

            loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")

        except Exception as e:
            logging.info(f"Exception : {e}")
            raise CustomException(e,sys)    
        return preprocessor_pickel, loaded_model
    
    except Exception as e:
        logging.info(f"Exception : {e}")
        raise CustomException(e,sys)