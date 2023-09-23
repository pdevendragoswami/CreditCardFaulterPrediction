import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
import pandas as pd
import numpy as np

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path =  os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformation_obj(self):
        try:
            logging.info('Data transformation Objecy is initiated')

            numerical_cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
             'PAY_1', 'PAY_2','PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
             'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
             'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
             
            numerical_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                    ]
            )

            logging.info('Pipeline is created')

            preprocessor = ColumnTransformer(
                [('numerical_pipeline',numerical_pipeline,numerical_cols)])

            logging.info('preprocessor object is created')

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_trasformation(self,train_data_path,test_data_path):
        try:
            logging.info('Data transformation is initiated')

            df_train_data = pd.read_csv(train_data_path)
            df_test_data = pd.read_csv(test_data_path)

            logging.info('Train and test data is read as Dataframe')

            target_feature = 'default payment next month'

            df_train_data_input_feature = df_train_data.drop(target_feature,axis=1)
            df_train_data_target_feature = df_train_data[target_feature]

            df_test_data_input_feature = df_test_data.drop(target_feature,axis=1)
            df_test_data_target_feature = df_test_data[target_feature]

            preprocessor_obj = self.get_data_transformation_obj()

            input_feature_train_array = preprocessor_obj.fit_transform(df_train_data_input_feature)
            input_feature_test_array = preprocessor_obj.transform(df_test_data_input_feature)

            train_array = np.c_[input_feature_train_array,df_train_data_target_feature]
            test_array = np.c_[input_feature_test_array,df_test_data_target_feature]

            logging.info('Data transformation is completed')

            save_obj(
                file_path=self.data_transformation_config.preprocesser_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Pickle file is saved')

            return(
                train_array,test_array,self.data_transformation_config.preprocesser_obj_file_path
            )


        except Exception as e:
            logging.info('There is some issue in data transformation')
            raise CustomException(e,sys)


