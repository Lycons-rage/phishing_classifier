import os
import sys
current = os.path.dirname(os.path.realpath("data_transformation.py"))
parent = os.path.dirname(current)
sys.path.append(current)

from distutils.errors import PreprocessError
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformationPhase:
    def __init__(self):
       self.dataTransformationConfig = DataTransformationConfig()

    def getDataTransformationObject(x):
        try:
            logging.info("Data Transformation Initiated")
            
            transform_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer()),
                    ("scaling", StandardScaler())
                ]
            )       
            logging.info("Pipeline Established")

            features = ['url_length', 'number_of_dots_in_url', 'having_repeated_digits_in_url',
                    'number_of_digits_in_url', 'number_of_special_char_in_url',
                    'number_of_hyphens_in_url', 'number_of_underline_in_url',
                    'number_of_slash_in_url', 'number_of_questionmark_in_url',
                    'number_of_equal_in_url', 'number_of_at_in_url',
                    'number_of_dollar_in_url', 'number_of_exclamation_in_url',
                    'number_of_hashtag_in_url', 'number_of_percent_in_url', 'domain_length',
                    'number_of_dots_in_domain', 'number_of_hyphens_in_domain',
                    'having_special_characters_in_domain',
                    'number_of_special_characters_in_domain', 'having_digits_in_domain',
                    'number_of_digits_in_domain', 'having_repeated_digits_in_domain',
                    'number_of_subdomains', 'having_dot_in_subdomain',
                    'having_hyphen_in_subdomain', 'average_subdomain_length',
                    'average_number_of_dots_in_subdomain',
                    'average_number_of_hyphens_in_subdomain',
                    'having_special_characters_in_subdomain',
                    'number_of_special_characters_in_subdomain',
                    'having_digits_in_subdomain', 'number_of_digits_in_subdomain',
                    'having_repeated_digits_in_subdomain', 'having_path', 'path_length',
                    'having_query', 'having_fragment', 'having_anchor', 'entropy_of_url',
                    'entropy_of_domain']
            preprocessor = ColumnTransformer([
                ("transform_pipeline", transform_pipeline, features)
            ])
            logging.info("Pipeline Completed")

            return preprocessor
        
        except Exception as e:
            logging.info(f"Exception : {e}")
            raise CustomException(e,sys)
    
    def initiateDataTransformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("DATA READ SUCCESSFULLY")
            logging.info(f"Train Data Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Data Head : \n{test_df.head().to_string()}")
            
            logging.info("OBTAINING PREPROCESSOR OBJECT")

            preprocessor_obj = self.getDataTransformationObject()

            X_train_df = train_df.drop("Type", axis=1)
            y_train_df = train_df["Type"]

            X_test_df = test_df.drop("Type", axis=1)
            y_test_df = test_df["Type"]

            logging.info("Applying preprocessor transformation on training and testing data frames")
            X_train_arr = preprocessor_obj.fit_transform(X_train_df)
            X_test_arr = preprocessor_obj.transform(X_test_df)

            train_arr = np.c_[X_train_arr, np.array(y_train_df)]
            test_arr = np.c_[X_test_arr, np.array(y_test_df)]

            save_object(
                file_path = self.dataTransformationConfig.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            logging.info("PICKLE FILE CREATED")

            return (
                train_arr, 
                test_arr,
                self.dataTransformationConfig.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info(f"Exception : {e}")
            raise CustomException(e,sys)