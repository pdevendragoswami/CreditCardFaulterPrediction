import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict_value(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_obj(preprocessor_path)
            
            model = load_obj(model_path)

            scaled_data = preprocessor.transform(features)

            predicted_value = model.predict(scaled_data)

            return predicted_value

        
        except Exception as e:
            logging.info('There is some issue at predict values')
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 limit_bal:int,
                 sex:int,
                 education:int,
                 marriage:int,
                 age:int,
                 pay_1:int,
                 pay_2:int,
                 pay_3:int,
                 pay_4:int,
                 pay_5:int,
                 pay_6:int,
                 bill_amt1:float,
                 bill_amt2:float,
                 bill_amt3:float,
                 bill_amt4:float,
                 bill_amt5:float,
                 bill_amt6:float,
                 pay_amt1:float,
                 pay_amt2:float,
                 pay_amt3:float,
                 pay_amt4:float,
                 pay_amt5:float,
                 pay_amt6:float):
        
        self.limit_bal=limit_bal
        self.sex=sex
        self.education=education
        self.marriage=marriage
        self.age=age
        self.pay_1=pay_1
        self.pay_2=pay_2
        self.pay_3=pay_3
        self.pay_4=pay_4
        self.pay_5=pay_5
        self.pay_6=pay_6
        self.bill_amt1=bill_amt1
        self.bill_amt2=bill_amt2
        self.bill_amt3=bill_amt3
        self.bill_amt4=bill_amt4
        self.bill_amt5=bill_amt5
        self.bill_amt6=bill_amt6
        self.pay_amt1=pay_amt1
        self.pay_amt2=pay_amt2
        self.pay_amt3=pay_amt3
        self.pay_amt4=pay_amt4
        self.pay_amt5=pay_amt5
        self.pay_amt6=pay_amt6

    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict =  {
                'LIMIT_BAL':[self.limit_bal],
                'SEX':[self.sex],
                'EDUCATION':[self.education],
                'MARRIAGE':[self.marriage],
                'AGE':[self.age],
                'PAY_1':[self.pay_1],
                'PAY_2':[self.pay_2],
                'PAY_3':[self.pay_3],
                'PAY_4':[self.pay_4],
                'PAY_5':[self.pay_5],
                'PAY_6':[self.pay_6],
                'BILL_AMT1':[self.bill_amt1],
                'BILL_AMT2':[self.bill_amt2],
                'BILL_AMT3':[self.bill_amt3],
                'BILL_AMT4':[self.bill_amt4],
                'BILL_AMT5':[self.bill_amt5],
                'BILL_AMT6':[self.bill_amt6],
                'PAY_AMT1':[self.pay_amt1],
                'PAY_AMT2':[self.pay_amt2],
                'PAY_AMT3':[self.pay_amt3],
                'PAY_AMT4':[self.pay_amt4],
                'PAY_AMT5':[self.pay_amt5],
                'PAY_AMT6':[self.pay_amt6],
            }

            df = pd.DataFrame(custom_data_input_dict)

            logging.info('Data converted into DF')

            return df


        except Exception as e:
            logging.info('There is some issue at get data as dataframe')
            raise CustomException(e, sys)
        