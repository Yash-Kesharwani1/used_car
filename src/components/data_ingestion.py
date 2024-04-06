import os
import sys

# We are importing CustomException Class to handle the exception
from src.exception import CustomException

# We are importing loggingg to create the log file and to track the code from which it is executed
from src.logger import logging

# importing pandas as we are playing with datafram
import pandas as pd

# Importing train_test_split class to split the data into train and test dataframe
from sklearn.model_selection import train_test_split

# 
from dataclasses import dataclass

@dataclass
# Any input is given through this DataIngestionConfig to Ingestion
class DataIngestionConfig:
    # This will set a path where to store the train data
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

# 
class DataIngestion:
    # This thing initialize the DataIngestionConfig by creating a ingestion_config object
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    # This funtion will initiate the data ingestion i.e. takes the data from the file and store it in train.csv , test.csv and data.csv
    def initiate_data_ingestion(self):

        # Create the logging to show we have entered into data ingestion component
        logging.info("Entered the data ingestionn method or compoenet")

        try:
            # This will take the data from the initial data path as dataframe
            df = pd.read_csv('notebook/autos.csv')

            # Writing logging to confirm that data is read from dataset i.e autos.csv
            logging.info("Read the dataset as dataframe")

            # This will create a directory at the path which is given to it 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            # This will create a csv file to store the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Writing a logging to initiate train test split
            logging.info("Train test split initiated.")

            # This line split the data into train and test set.
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # This will create a csv file to store the train data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # This will create a csv file to store the test data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            #
            logging.info('Ingestion of the data is completed')
            
            # We return train_data_path and test_data_path for data transformtion to fetch the data from the correct location.
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        # This will raise exception 
        except Exception as e:
            raise CustomException(e, sys)