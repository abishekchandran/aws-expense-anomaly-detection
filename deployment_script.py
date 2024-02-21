from prophet import Prophet
import pandas as pd
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()


# Access the environment variables
aws_expense_data_path = os.getenv("AWS_EXPENSE_DATA_PATH")

def trainAndPredict(data,date_to_predict):
    model = Prophet()
    model.fit(data)
    future = pd.DataFrame({'ds': [date_to_predict]})
    predicted_price = model.predict(future)
    return predicted_price

def normalize_value(value, series):
    return (value - series.min()) / (series.max() - series.min())

def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


#NORMALISED DATA

try:
    df = pd.read_csv(aws_expense_data_path)
    df.rename(columns={'Date': 'ds', 'Expense ($)': 'y'}, inplace=True)

    df['ds'] = pd.to_datetime(df['ds'])
    today_date = pd.to_datetime('today').date()
    today_rows = df[df['ds'].dt.date == today_date]

    today_timestamp = today_rows['ds'].values[0]

    date_to_predict = np.datetime_as_string(today_timestamp, unit='D')


    actual_price = today_rows['y'].values[0] 
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d')
    df = df[df['ds'] != date_to_predict]

    normalized_actual_price = normalize_value(actual_price, df['y'])

    normalize_df = normalize_dataframe(df['y'])
    df['y'] = normalize_df

    forecast = trainAndPredict(df,date_to_predict)

    predicted_price = forecast['yhat'].iloc[0]

    threshold = 0.30


    if abs(predicted_price - normalized_actual_price) > threshold:
        print(f" {date_to_predict} Anomaly Detected: Deviation between predicted and actual prices exceeds the threshold.")
    else:
        print(f" {date_to_predict} No Anomaly Detected: Predicted and actual prices are within the threshold.")
        
except Exception as e:
    print(f"An error occurred: {e}")
    
