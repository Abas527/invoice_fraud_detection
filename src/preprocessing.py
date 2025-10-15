import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import datetime
import joblib

def preprocess(df):

    #label encoding vendor and client name
    le = LabelEncoder()
    df['vendor_name'] = le.fit_transform(df['vendor_name'])
    df['client_name'] = le.fit_transform(df['client_name'])

    #create new features
    df["amount_per_day"]=df["total_gross"]/(pd.to_datetime(df["due_date"]) - pd.to_datetime(df["issue_date"])).dt.days

    #drop columns not needed
    df.drop(columns=['id', 'vendor_name', 'client_name', 'due_date', 'issue_date',"fraud_type","client_tax_id","vendor_tax_id","invoice_number"], inplace=True)
    df.fillna(0, inplace=True)

    #label encoding isfraud
    df["is_fraud"] = df["is_fraud"].map({False: 0, True: 1})

    return df

def train_test_split_data(df=pd.read_csv('data/preprocessed.csv')):
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=['is_fraud']),
        df['is_fraud'],
        test_size=0.2,
        random_state=42,
        stratify=df['is_fraud']
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    joblib.dump((x_train, x_test, y_train, y_test, scaler), 'model/scaler_and_data.pkl')

    return x_train, x_test, y_train, y_test, scaler



def main():
    # Load data
    df = pd.read_csv('data/raw.csv')

    # Preprocess data
    df1=preprocess(df)
    x_train, x_test, y_train, y_test, scaler = train_test_split_data(df1)

    # Save preprocessed data
    df1.to_csv('data/preprocessed.csv', index=False)

    


if __name__ == "__main__":
    main()