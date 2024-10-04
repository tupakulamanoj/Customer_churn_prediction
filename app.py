from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
with open('Think_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('Think_preprocessor.pkl', 'rb') as file:
    preprocess = pickle.load(file)
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = int(request.form['age'])
        Gender=request.form['gender']
        monthly_charges = float(request.form['monthly-charges'])
        TotalCharges=request.form['total-charges']
        tenure = int(request.form['tenure'])
        contract_type = request.form['contract-type']
        internet_service = request.form['internet-service']
        tech_support = request.form['tech-support']
        payment_method = request.form['payment-method']
        paperless_billing = request.form['paperless-billing']
        average_monthly_charges=request.form['average-monthly-charges']
        customer_lifetime_value=request.form['customer-lifetime-value']

        # Prepare feature data for prediction

        
        features = pd.DataFrame([{
            'Age': age,
            'Gender':Gender,
            'MonthlyCharges': monthly_charges,
            'ContractType': contract_type,
            'InternetService': internet_service,
            'TechSupport': tech_support,
            'Tenure': tenure,
            'PaymentMethod': payment_method,
            'PaperlessBilling': paperless_billing,
            'MonthlyCharges':monthly_charges,
            'TotalCharges':TotalCharges,
            'average_monthly_charges':average_monthly_charges,
            'customer_lifetime_value':customer_lifetime_value
        }])
        print(features)
        m=preprocess.transform(features)
        prediction = model.predict(m)[0]
        result = 'Yes Churn' if prediction == 1 else 'No Churn'
        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)