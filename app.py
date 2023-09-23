from flask import Flask,render_template,request,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
from src.logger import logging
import os
import sys

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods= ['GET','POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('form.html')

        else:
            data = CustomData(
                limit_bal=int(request.form.get('limit_bal')),
                sex = int(request.form.get('sex')),
                education = int(request.form.get('education')),
                marriage = int(request.form.get('education')),
                age = int(request.form.get('education')),
                pay_1 = (int(request.form.get('pay_1'))),
                pay_2 = (int(request.form.get('pay_2'))),
                pay_3 = (int(request.form.get('pay_3'))),
                pay_4 = (int(request.form.get('pay_4'))),
                pay_5 = (int(request.form.get('pay_5'))),
                pay_6 = (int(request.form.get('pay_6'))),
                bill_amt1 = float(request.form.get('bill_amt1')),
                bill_amt2 = float(request.form.get('bill_amt2')),
                bill_amt3 = float(request.form.get('bill_amt3')),
                bill_amt4 = float(request.form.get('bill_amt4')),
                bill_amt5 = float(request.form.get('bill_amt5')),
                bill_amt6 = float(request.form.get('bill_amt6')),
                pay_amt1 = float(request.form.get('pay_amt1')),
                pay_amt2 = float(request.form.get('pay_amt2')),
                pay_amt3 = float(request.form.get('pay_amt3')),
                pay_amt4 = float(request.form.get('pay_amt4')),
                pay_amt5 = float(request.form.get('pay_amt5')),
                pay_amt6 = float(request.form.get('pay_amt6')))

            final_data = data.get_data_as_dataframe()
            predict_pipeline_obj = PredictPipeline()
            pred_value = predict_pipeline_obj.predict_value(final_data)

            results = pred_value[0]

            return render_template('results.html',final_result = results)

    except  Exception as e:
        logging.info('There is some issue at predict_datapoint')
        raise CustomException(e,sys)



if __name__ == "__main__":
    app.run(host='0.0.0.0')