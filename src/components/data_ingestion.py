import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts','raw.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Data Ingestion is initiated')
            df = pd.read_csv(os.path.join('notebooks','data','data.csv'))
            logging.info('Data loaded as a dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info('Raw data file is saved')

            train_data,test_data = train_test_split(df,random_state=1,test_size=.25)
            logging.info('Data splited into train and test part')

            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False)
            logging.info('Train data is saved into CSV file')

            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info('Test data is saved into CSV file')

            logging.info('Data Ingestion is completed.')

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info('There is some issue at data ingestion')
            raise CustomException(e,sys)

