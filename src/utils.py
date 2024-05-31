# importing required libraries
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
from mlflow.models import infer_signature
import pickle

import re
from urllib.parse import urlparse
from collections import Counter
import math

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
    mlflow.sklearn.log_model(
        sk_model=random_search.best_estimator_,
        artifact_path=f"{model_name}",
        registered_model_name=f"{model_name}"
    )
    mlflow_client = mlflow.MlflowClient()
    mlflow_client.create_registered_model(name=model_name)
    mlflow_client.create_model_version(
        name=model_name,
        source=f"mlruns/0/{run_id}/artifacts/{model_name}",
        run_id=run_id
    )
    # we need to store this model name 
    file_path = os.path.join("artifacts","model_name.txt")
    with open(file_path, "w") as file:
        file.write(model_name) 

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
        with open(file_path, "rb") as file_obj:
            try:
                # #mlflow_client = mlflow.tracking.MlflowClient()
                # registered_model = mlflow.registered_model.list_registered_models()
                # model_name = registered_model[0].name
                model_file_path = os.path.join("artifacts", "model_name.txt")
                with open(model_file_path, "r") as model_file_obj:
                    model_name = model_file_obj.read()
                    loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")

            except Exception as e:
                logging.info(f"Exception : {e}")
                raise CustomException(e,sys)    
            return pickle.load(file_obj), loaded_model
    
    except Exception as e:
        logging.info(f"Exception : {e}")
        raise CustomException(e,sys)


# get entropy of url string    
def entropy(s) -> float:
    """Calculate the entropy of a string."""
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log2(count/lns) for count in p.values())


# data preparation
def extract_features(url:str) -> dict:
    # Initialize the data_dict
    data_dict = {
        'url_length' : [len(url)],
        'number_of_dots_in_url' : [url.count('.')],
        'having_repeated_digits_in_url': [int(bool(re.search(r'(\d)\1', url)))],
        'number_of_digits_in_url': [len(re.findall(r'\d', url))], 
        'number_of_special_char_in_url': [len(re.findall(r'[~`!@#$%^&*()_\-+=\'";:<>,./?|\{}[\]]', url))],
        'number_of_hyphens_in_url': [url.count('-')], 
        'number_of_underline_in_url': [url.count('_')],
        'number_of_slash_in_url': [url.count('/')], 
        'number_of_questionmark_in_url': [url.count('?')],
        'number_of_equal_in_url': [url.count('=')], 
        'number_of_at_in_url': [url.count('@')],
        'number_of_dollar_in_url': [url.count('$')], 
        'number_of_exclamation_in_url': [url.count('!')],
        'number_of_hashtag_in_url': [url.count('#')], 
        'number_of_percent_in_url': [url.count('%')], 
        'domain_length': [len(urlparse(url).netloc)],
        'number_of_dots_in_domain': [urlparse(url).netloc.count('.')], 
        'number_of_hyphens_in_domain': [urlparse(url).netloc.count('-')],
        'having_special_characters_in_domain': [int(bool(re.search(r'[^a-zA-Z0-9.-]', urlparse(url).netloc)))],
        'number_of_special_characters_in_domain': [len(re.findall(r'[^a-zA-Z0-9.-]', urlparse(url).netloc))],
        'having_digits_in_domain': [int(bool(re.search(r'\d', urlparse(url).netloc)))],
        'number_of_digits_in_domain': [len(re.findall(r'\d', urlparse(url).netloc))], 
        'having_repeated_digits_in_domain': [int(bool(re.search(r'(\d)\1', urlparse(url).netloc)))],
        'number_of_subdomains': [urlparse(url).netloc.count('.') - 1],
        'having_dot_in_subdomain': [int(bool(re.search(r'\.', urlparse(url).netloc.split('.')[0])))],
        'having_hyphen_in_subdomain': [int(bool(re.search(r'-', urlparse(url).netloc.split('.')[0])))],
        'average_subdomain_length': [sum(len(part) for part in urlparse(url).netloc.split('.')) / len(urlparse(url).netloc.split('.'))],
        'average_number_of_dots_in_subdomain': [urlparse(url).netloc.split('.').count('.') / len(urlparse(url).netloc.split('.'))],
        'average_number_of_hyphens_in_subdomain': [urlparse(url).netloc.split('.').count('-') / len(urlparse(url).netloc.split('.'))],
        'having_special_characters_in_subdomain': [int(bool(re.search(r'[^a-zA-Z0-9-]', urlparse(url).netloc.split('.')[0])))],
        'number_of_special_characters_in_subdomain': [len(re.findall(r'[^a-zA-Z0-9-]', urlparse(url).netloc.split('.')[0]))],
        'having_digits_in_subdomain': [int(bool(re.search(r'\d', urlparse(url).netloc.split('.')[0])))],
        'number_of_digits_in_subdomain': [len(re.findall(r'\d', urlparse(url).netloc.split('.')[0]))],
        'having_repeated_digits_in_subdomain': [int(bool(re.search(r'(\d)\1', urlparse(url).netloc.split('.')[0])))],
        'having_path': [int(bool(urlparse(url).path))],
        'path_length': [len(urlparse(url).path)],
        'having_query': [int(bool(urlparse(url).query))],
        'having_fragment': [int(bool(urlparse(url).fragment))],
        'having_anchor': [int(bool(urlparse(url).fragment))],
        'entropy_of_url': [entropy(url)],
        'entropy_of_domain': [entropy(urlparse(url).netloc)]
    }
    return data_dict