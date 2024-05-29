import os
import sys
current = os.path.dirname(os.path.realpath("data_ingestion.py"))
parent = os.path.dirname(current)
sys.path.append(current)
from src.exception import CustomException
from src.logger import logging
from src.mlflow_logs import Mlflow_logs

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import mlflow
import random


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train_data.csv')
    test_data_path:str = os.path.join('artifacts','test_data.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw_data.csv')
# permission denied on this address     ^

class DataIngestionPhase:
    def __init__(self):
        self.dataIngestionConfig = DataIngestionConfig

    def dataIngestion(self):
        logging.info("Data Ingestion Phase Start")
        try:
            df = pd.read_csv("notebooks/data/Dataset.csv")
            if os.path.exists(self.dataIngestionConfig.raw_data_path):
               pass
            else:
                os.makedirs(self.dataIngestionConfig.raw_data_path, exist_ok=True)
                df.to_csv(os.path.join(self.dataIngestionConfig.raw_data_path), index=False)

            logging.info("Train Test Split")    
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.dataIngestionConfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.dataIngestionConfig.test_data_path, index=False, header=True)

            logging.info("Data Ingestion Completed Successfully")    
            return(
                self.dataIngestionConfig.train_data_path,
                self.dataIngestionConfig.test_data_path
            )
            
        except Exception as e:
            logging.info(f"Exception : {e}")
            raise CustomException(e,sys)


c = DataIngestionPhase()
c.dataIngestion()