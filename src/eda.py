import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

def eda(df):
    print(df.head())
    print(df.info())
    print(df.isnull().sum())
    print(df.describe())
    print(df['is_fraud'].value_counts())

    #plotting class distribution
    # plt.figure(figsize=(6,4))
    # df['is_fraud'].value_counts().plot(kind='bar')
    # plt.title('Class Distribution')
    # plt.show()
    plt.show()


if __name__ == "__main__":
    data_path = os.path.join( "data", "raw.csv")
    df = pd.read_csv(data_path, parse_dates=['issue_date', 'due_date'])
    eda(df)