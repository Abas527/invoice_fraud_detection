import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import shap

models={
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
}

def train_model(x_train,y_train,model_name="RandomForest"):
    model=models.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return y_pred

def explain_model(model, x_train):
    explainer = shap.Explainer(model, x_train)
    shap_values = explainer(x_train)
    shap.summary_plot(shap_values, x_train)
    shap.summary_plot(shap_values, x_train, plot_type="bar")

    shap.summary_plot(shap_values, x_train, plot_type="dot")

def main():
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed.csv')
    x_train, x_test, y_train, y_test, scaler = joblib.load('model/scaler_and_data.pkl')

    #model 1
    model_name="RandomForest"
    model1 = train_model(x_train, y_train, model_name)
    print(f"Evaluating {model_name}")
    evaluate_model(model1, x_test, y_test)

    # explain_model(model1, x_train)
    explain_model(model1, x_test)

    #model 2
    model_name="XGBoost"
    model2 = train_model(x_train, y_train, model_name)
    print(f"Evaluating {model_name}")
    evaluate_model(model2, x_test, y_test)

    # explain_model(model2, x_train)
    explain_model(model2, x_test)

    # Save the model
    joblib.dump((model1, model2), 'model/fraud_detection_models.pkl')

if __name__ == "__main__":
    main()
