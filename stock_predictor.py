import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def fetch_data(ticker='AAPL', start_date='2020-01-01'):
    """Fetches historical stock data from Yahoo Finance."""
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date)
    return data

def prepare_data(data):
    """Prepares the data for training."""
    # Ensure we have the columns we need.
    
    df = data.copy()
    
    # Feature Engineering: Use Open, High, Low, Volume to predict Close
    # We predict the NEXT day's closing price.
    
    df['Target'] = df['Close'].shift(-1)
    
    # Drop the last row as it will have NaN target
    df = df.dropna()
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']
    
    return X, y, df.index

def train_model(X, y):
    """Trains a Random Forest Regressor."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def predict_next_day(model, last_data):
    """Predicts the closing price for the next trading day."""
    # last_data should be a dataframe (1 row) with features (OHLCV)
    prediction = model.predict(last_data)
    return prediction[0]
