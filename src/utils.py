
import os
import sys
import pickle
import pandas as pd
import numpy as np

import mlflow
import pickle

from src.exception import CustomException
from src.logger import logging


# function to save an object into a pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with mlflow.start_run() as run:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
                mlflow.log_artifact(file_path)

                run_id = run.info.run_id
                artifact_uri = mlflow.get_artifact_uri()

    except Exception as e:
        logging.info("EXCEPTION OCCURRED IN UTLIS.PY")
        raise CustomException(e, sys)
    

def train_and_evaluate_model(X_train, y_train, X_test, y_test, models):
    pass