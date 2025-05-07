#using Python 3.9

import streamlit as st # streamlit run "C:\Users\Sam's Folder\Final_Project_App\main.py"
from datetime import date
from datetime import timedelta
import yfinance as yf
from plotly import graph_objs as go
from keras.models import load_model
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

START ="2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Price Prediction Web App Prototype")
st.subheader("Samuel Nadarajah-Stookin w1834997")

value_error = st.text("ValueError: enter valid symbol to progress...")

stocks = st.text_input("Enter stock symbol e.g. AAPL, AAL, MSFT:", "", key="stock_symbol_input")
stocks = stocks.upper()

n_days = st.slider("Days of prediction: ", 1 , 120)
period = n_days

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

df = load_data(stocks)
df.columns = [col[0] for col in df.columns]

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

#_ _ _ _ _ _ _ _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

st.subheader("Current data for {}".format(stocks))
st.write(df)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name='Open Price'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'))
    fig.layout.update(title_text="Actual open and close price of {}".format(stocks), xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

window = 128

@st.cache_resource(show_spinner=True)
def prediction_model(data):
    close_data = data[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    split_fraction = 0.8
    split_index = int(len(scaled_data) * split_fraction)

    train_data = scaled_data[:split_index]
    val_data = scaled_data[split_index - window:]

    def create_sequences(data, window):
        x, y = [], []
        for i in range(window, len(data)):
            x.append(data[i - window:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    x_train, y_train = create_sequences(train_data, window)
    x_val, y_val = create_sequences(val_data, window)


    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(window, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=75))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=2)

    val_predictions = model.predict(x_val)

    val_predictions_rescaled = scaler.inverse_transform(val_predictions)
    y_val_rescaled = scaler.inverse_transform(y_val.reshape(-1, 1))

    st.subheader("Training MSE Over Epochs")
    mse = history.history['loss']

    st.write(mse)
    fig3, ax3 = plt.subplots()
    epochs = range(1, len(mse) + 1)
    ax3.plot(epochs, mse, marker='o', label='MSE')
    for i, val in enumerate(mse):
        ax3.text(epochs[i], val, f"{val:.4f}", ha='center', va='bottom', fontsize=8)

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MSE')
    ax3.set_title('Mean Squared Error Over Epochs')
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

    plt.figure(figsize=(10, 5))
    plt.plot(y_val_rescaled, label='Actual Price', linewidth=2)
    plt.plot(val_predictions_rescaled, label='Predicted Price', linestyle='--')
    plt.title('Actual closing price vs predicted validation set')
    plt.xlabel('Data point')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    return model, scaler

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

model, scaler = prediction_model(df)

last_sequence = df[['Close']].values[-window:]
last_scaled = scaler.transform(last_sequence)
input_seq = last_scaled.reshape(1, window, 1)

predictions = []
for i in range(n_days):
    pred = model.predict(input_seq)[0][0]
    predictions.append(pred)
    pred_reshaped = np.array(pred).reshape(1, 1, 1)
    input_seq = np.append(input_seq[:, 1:, :], pred_reshaped, axis=1)

forecasted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

future_dates = [df['Date'].iloc[-1] + timedelta(days=i+1) for i in range(n_days)]
forecast_df = pd.DataFrame({'Date': future_dates, 'Predictions': forecasted_prices.flatten()})
forecast_df.set_index('Date', inplace=True)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

st.subheader("Predicted close price by LSTM model compared to actual close price")
st.write(forecast_df)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Actual Close Price'))
fig3.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predictions'], name='Forecasted Price'))
fig3.update_layout(title_text=f"Predicted vs Actual for {stocks}", xaxis_rangeslider_visible=True)
st.plotly_chart(fig3)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _