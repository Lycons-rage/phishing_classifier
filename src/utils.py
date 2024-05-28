
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

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

    return report