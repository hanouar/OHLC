# Import Libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------
# 1. Training and Saving Models
# ------------------------

def download_data():
    data = yf.download('GC=F', start='2000-01-01', end='2025-02-21')
    data = data[['Open', 'High', 'Low', 'Close']]
    return data

def compute_indicators(df):
    # Moving Averages
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    return df

def train_and_save_models():
    data = download_data()
    data = compute_indicators(data)
    data.dropna(inplace=True)

    # Split OHLC and Indicators
    ohlc = data[['Open', 'High', 'Low', 'Close']]
    indicators = data[['SMA50', 'SMA200', 'RSI', 'MACD', 'Signal_Line', 'MACD_Hist']]

    # Scale Data
    scaler_ohlc = MinMaxScaler()
    scaler_indicators = MinMaxScaler()

    scaled_ohlc = scaler_ohlc.fit_transform(ohlc)
    scaled_indicators = scaler_indicators.fit_transform(indicators)

    # Save Scalers
    joblib.dump(scaler_ohlc, 'scaler_ohlc.pkl')
    joblib.dump(scaler_indicators, 'scaler_indicators.pkl')

    # Prepare LSTM Dataset
    sequence_length = 60
    X_lstm, y_lstm = [], []

    for i in range(sequence_length, len(scaled_ohlc)):
        X_lstm.append(scaled_ohlc[i-sequence_length:i])
        y_lstm.append(scaled_ohlc[i])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    # Build LSTM Model
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], 4)),
        LSTM(50),
        Dense(4)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=1)
    lstm_model.save('lstm_model.h5')

    # Generate LSTM Predictions for XGBoost
    y_pred_lstm = lstm_model.predict(X_lstm)

    # Prepare XGBoost Dataset (Align Features)
    shifted_indicators = scaled_indicators[sequence_length-1:-1]  # Adjust indices
    X_gb = np.hstack((y_pred_lstm, shifted_indicators))
    y_gb = scaled_ohlc[sequence_length:]

    # Train XGBoost Model
    xgb_model = XGBRegressor(n_estimators=100)
    xgb_model.fit(X_gb, y_gb)
    joblib.dump(xgb_model, 'xgb_model.pkl')

# ------------------------
# 2. Loading Models and Making Predictions
# ------------------------

def load_models_and_predict():
    # Load Models and Scalers
    lstm_model = load_model('lstm_model.h5', custom_objects={'mse': MeanSquaredError()})
    xgb_model = joblib.load('xgb_model.pkl')
    scaler_ohlc = joblib.load('scaler_ohlc.pkl')
    scaler_indicators = joblib.load('scaler_indicators.pkl')

    # Fetch Latest Data
    new_data = yf.download('GC=F', period='260d')  # Sufficient history
    new_data = compute_indicators(new_data)
    new_data.dropna(inplace=True)

    # Preprocess New Data
    ohlc_new = new_data[['Open', 'High', 'Low', 'Close']]
    indicators_new = new_data[['SMA50', 'SMA200', 'RSI', 'MACD', 'Signal_Line', 'MACD_Hist']]

    scaled_ohlc_new = scaler_ohlc.transform(ohlc_new)
    scaled_indicators_new = scaler_indicators.transform(indicators_new)

    # Prepare LSTM Input
    X_lstm_new = scaled_ohlc_new[-sequence_length:]
    X_lstm_new = np.array([X_lstm_new])

    # Predict with LSTM
    pred_lstm = lstm_model.predict(X_lstm_new)

    # Prepare XGBoost Input
    latest_indicators = scaled_indicators_new[-1].reshape(1, -1)
    X_gb_new = np.hstack((pred_lstm, latest_indicators))

    # Predict with XGBoost
    pred_gb = xgb_model.predict(X_gb_new)

    # Inverse Transform Predictions
    pred_ohlc_lstm = scaler_ohlc.inverse_transform(pred_lstm)
    pred_ohlc_gb = scaler_ohlc.inverse_transform(pred_gb)

    return pred_ohlc_lstm, pred_ohlc_gb

# ------------------------
# Streamlit App
# ------------------------

st.title("Gold Price Prediction using LSTM and XGBoost")

if st.button("Train and Save Models"):
    with st.spinner("Training models..."):
        train_and_save_models()
    st.success("Models trained and saved successfully!")

if st.button("Load Models and Make Predictions"):
    with st.spinner("Loading models and making predictions..."):
        pred_ohlc_lstm, pred_ohlc_gb = load_models_and_predict()
    st.success("Predictions made successfully!")

    st.subheader("LSTM Predicted OHLC (Open, High, Low, Close):")
    st.write(pred_ohlc_lstm[0])

    st.subheader("XGBoost Predicted OHLC (Open, High, Low, Close):")
    st.write(pred_ohlc_gb[0])

# Optional: Plot the predictions
if st.checkbox("Show Plot"):
    pred_ohlc_lstm, pred_ohlc_gb = load_models_and_predict()
    plt.figure(figsize=(10, 6))
    plt.plot(pred_ohlc_lstm[0], label='LSTM Predicted OHLC')
    plt.plot(pred_ohlc_gb[0], label='XGBoost Predicted OHLC')
    plt.legend()
    st.pyplot(plt)