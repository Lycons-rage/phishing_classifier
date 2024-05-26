import os
import sys

# for resolving any path conflict
current = os.path.dirname(os.path.realpath("data_transformation.py"))
parent = os.path.dirname(current)
sys.path.append(current)

from src.components.data_ingestion import DataIngestionPhase
from src.components.data_transformation import DataTransformationPhase
from src.components.model_trainer import ModelTrainerPhase

if __name__ == "__main__":
    ingestion_obj = DataIngestionPhase()
    train_path, test_path = ingestion_obj.dataIngestion()
    transformation_obj = DataTransformationPhase()
    train_arr, test_arr, _ = transformation_obj.initiateDataTransformation(train_path, test_path)
    training_obj = ModelTrainerPhase()
    training_obj.modelTraining(train_arr=train_arr, test_arr=test_arr)