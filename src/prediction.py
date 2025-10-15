import os
import numpy as np
import pandas as pd
import joblib

def load_models(model_path='model/fraud_detection_models.pkl', scaler_path='model/scaler_and_data.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler file not found. Please ensure the model and scaler are trained and saved.")

    models = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return models, scaler

def preprocess_input(data, scaler):
    df = pd.DataFrame([data])
    df.fillna(0, inplace=True)
    scaled_data = scaler.transform(df)
    return np.array(scaled_data)

def fraud_prediction(data,model):
    y_pred = model.predict(data)
    return y_pred

def anamoly_detection(data, model):
    y_pred = model.predict(data)
    y_pred = np.where(y_pred == -1, 1, 0)  # convert -1 to 1 (anomaly) and 1 to 0 (normal)
    return y_pred

def main():

    input_data={
        # num_items,total_amount,vat_percent,vat_amount,total_gross,payment_terms,is_fraud,amount_per_day

        "num_items": 5,
        "total_amount": 1500.75,
        "vat_percent": 20,
        "vat_amount": 300.15,
        "total_gross": 1800.90,
        "payment_terms": 30,
        "amount_per_day": 60.03

    }

    model_supervised, scaler_and_data = load_models()
    scaler = scaler_and_data[4]
    processed_data = preprocess_input(input_data, scaler)
    fraud_result = fraud_prediction(processed_data, model_supervised[1])
    print(fraud_result)

    model_unsupervised, _ = load_models(model_path='model/one_class_svm_model.pkl', scaler_path='model/scaler.pkl')
    anamoly_result = anamoly_detection(processed_data, model_unsupervised)
    print(anamoly_result)


if __name__ == "__main__":
    main()

