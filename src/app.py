import streamlit as st
from utils import predict
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from utils import explain_supervised_prediction, explain_unsupervised_prediction
import shap
def main():
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon=":credit_card:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.example.com/help',
            'Report a bug': "https://www.example.com/bug",
            'About': "# This is a fraud detection system app built with Streamlit." 
        }
    )
    st.title("Fraud Detection System")
    st.write("This app predicts if a transaction is fraudulent or not.")
    st.write("Enter the transaction details below:")
    # num_items,total_amount,vat_percent,vat_amount,total_gross,payment_terms,is_fraud,amount_per_day

    num_items = st.number_input("Number of Items", min_value=1, max_value=1000, value=5)
    total_amount = st.number_input("Total Amount", min_value=0.0, max_value=1e6, value=1500.75)
    vat_percent = st.number_input("VAT Percent", min_value=0.0, max_value=90.0, value=20.0)
    vat_amount=total_amount * (vat_percent / 100)
    total_gross = total_amount + vat_amount
    payment_terms = st.selectbox("Payment Terms", options=[15, 30, 45, 60], index=1)
    amount_per_day = total_amount / payment_terms

    input_data = {
        "num_items": num_items,
        "total_amount": total_amount,
        "vat_percent": vat_percent,
        "vat_amount": vat_amount,
        "total_gross": total_gross,
        "payment_terms": payment_terms,
        "amount_per_day": amount_per_day
    }
    st.write("### Input Data")
    st.json(input_data)

    if st.button("Predict Fraud"):
        model_path = os.path.join("model", "fraud_detection_models.pkl")
        scaler_path = os.path.join("model", "scaler_and_data.pkl")
        prediction = predict(model_path, scaler_path, input_data, fraud=1)
        result = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"
        st.write(f"### Prediction: {result}")

    if st.button("Show Fraud Explanation"):
            model_path = os.path.join("model", "fraud_detection_models.pkl")
            scaler_path = os.path.join("model", "scaler_and_data.pkl")
            shap_values = explain_supervised_prediction(model_path, scaler_path, input_data)
            st.write("### SHAP Values for Fraud Prediction")

            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, feature_names=list(input_data.keys()), show=False,plot_type="bar")
            st.pyplot(fig)


    
    if st.button("show Anamoly"):
        model_path = os.path.join("model", "one_class_svm_model.pkl")
        scaler_path = os.path.join("model", "scaler.pkl")
        prediction = predict(model_path, scaler_path, input_data, fraud=0)
        result = "Anomaly Detected" if prediction[0] == 1 else "No Anomaly"
        st.write(f"### Anamoly Detection: {result}")

    if st.button("Show Anamoly Explanation"):
            model_path = os.path.join("model", "one_class_svm_model.pkl")
            scaler_path = os.path.join("model", "scaler.pkl")
            shap_values = explain_unsupervised_prediction(model_path, scaler_path, input_data)
            st.write("### SHAP Values for Anamoly Detection")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, feature_names=list(input_data.keys()), show=False,plot_type="dot")
            st.pyplot(fig)


if __name__ == "__main__":
    main()


