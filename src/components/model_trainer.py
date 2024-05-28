import os
import sys

# for resolving any path conflict
current = os.path.dirname(os.path.realpath("data_transformation.py"))
parent = os.path.dirname(current)
sys.path.append(current)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, train_and_evaluate_model

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainerPhase:
    def __init__(self):
        self.modelTrainerConfig = ModelTrainerConfig()

    def modelTraining(self, train_arr, test_arr):
        try:
            logging.info("Model Training Starts")
            logging.info("Splitting the data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'RandomForest' : RandomForestClassifier(n_jobs=-1),
                'SVC' : SVC(kernel='rbf'),
                'GBM' : GradientBoostingClassifier(),
                'ADA' : AdaBoostClassifier(),
                'LogisticReg' : LogisticRegression(penalty='l2', n_jobs=-1),
                'DecisionTree' : DecisionTreeClassifier()
            }

            model_report:dict = train_and_evaluate_model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=models
            )

            print(model_report)
            print(30*"=")
            logging.info(f"Model Report :\n{model_report}")
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            print(f"BEST MODEL FOUND\n{30*"-"}\nMODEL NAME : {best_model_name}\nSCORE : {best_model_score}")
            print(30*"=")
            logging.info(f"BEST MODEL FOUND\nMODEL NAME : {best_model_name}\nSCORE : {best_model_score}")

            save_object(
                file_path = self.modelTrainerConfig.trained_model_file_path,
                obj = best_model_name
            )

        except Exception as e:
            logging.info(f"Exception : {e}")
            raise CustomException(e,sys)