from prediction import preprocess_input, load_models, fraud_prediction, anamoly_detection
import pandas as pd
import numpy as np
import os
import joblib
import shap


def predict(model_path,scaler_path,input_data,fraud=1):
    models, scaler = load_models(model_path, scaler_path)
    
    if fraud == 1:
        model = models[1]
        scaler = scaler[4]

        processed_data = preprocess_input(input_data, scaler)

        return fraud_prediction(processed_data, model)
    else:
        model = models
        processed_data = preprocess_input(input_data, scaler)

        return anamoly_detection(processed_data, model)


def explain_supervised_prediction(model_path, scaler_path, input_data):
    models, scaler = load_models(model_path, scaler_path)
    model = models[1]
    scaler = scaler[4]

    processed_data = preprocess_input(input_data, scaler)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(processed_data, check_additivity=False,approximate=True)

    shap.summary_plot(shap_values, processed_data)
    shap.summary_plot(shap_values, processed_data, plot_type="bar")
    shap.summary_plot(shap_values, processed_data, plot_type="dot")
    return shap_values

def explain_unsupervised_prediction(model_path, scaler_path, input_data):
    model, scaler = load_models(model_path, scaler_path)
    processed_data = preprocess_input(input_data, scaler)

    explainer = shap.KernelExplainer(model.predict, shap.sample(processed_data, 100))
    shap_values = explainer(processed_data)

    shap.summary_plot(shap_values, processed_data)
    shap.summary_plot(shap_values, processed_data, plot_type="bar")
    shap.summary_plot(shap_values, processed_data, plot_type="dot")
    return shap_values