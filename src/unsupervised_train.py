import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def preprocess(df):

    normal_data=df[df['is_fraud']==0]

    x_train, x_test, y_train, y_test = train_test_split(
        normal_data.drop(columns=['is_fraud']),
        normal_data['is_fraud'],
        test_size=0.2,
        random_state=42
    )

    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    return x_train, x_test,y_test, scaler


def train(x_train):
    model=OneClassSVM(gamma='auto',kernel='rbf',nu=0.1).fit(x_train)
    return model

def evaluate_model(model, x_test, y_test):
    
    y_pred = model.predict(x_test)
    # print(y_pred)
    y_pred=np.where(y_pred==-1,1,0) # convert -1 to 1 (anomaly) and 1 to 0 (normal)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return y_pred

def main():
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed.csv')
    x_train, x_test,y_test, scaler = preprocess(df)

    model = train(x_train)

    # Save the model and scaler
    joblib.dump(model, 'model/one_class_svm_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')


    print("Evaluating OneClassSVM")
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()