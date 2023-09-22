import os
import sys
from src.exception import CustomException
from src.logger import logging
import pickle

def save_obj(file_path:str,obj:str):
    try:
        logging.info('Saving Object is initiated')
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

        logging.info('Object is saved')
    except Exception as e:
        logging.info('There is some issue at Save Object')
        raise CustomException(e,sys)
        

