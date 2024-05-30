import sys
import os 

# for resolving any path conflict
current = os.path.dirname(os.path.realpath("data_transformation.py"))
parent = os.path.dirname(current)
sys.path.append(current)

import pandas as pd
import numpy as np
import mlflow

from src.exception import CustomException
from src.logger import logging
from src.utils import extract_features, load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict_data(self, features):
        try:
            with mlflow.start_run():
                # Loading the preprocessor and registered model
                preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")
                preprocessor,model = load_object(preprocessor_file_path)

                # Data Scaling
                data = preprocessor.transform(features)
                # Prediction
                prediction = model.predict(data)
                return prediction
            
        except Exception as e:
            logging.info(f"Exception : {e}")
            raise CustomException(e,sys)
        

# data will be a url and we need to extract data from the url based on our raw dataset features
# such a prepared dataset will then be fed into the model for the final prediction
class CustomDataPreparation:
    def __init__(self, url):
        self.url=url

    def prepare_data(self):
        return extract_features(self.url)

    def get_data_as_dataframe(self, data_dict):
        return pd.DataFrame(data_dict)