#using Python 3.9

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as sk
import datetime as dt
import keras
from keras.models import Sequential #Keras version 2.15.0 with Tensorflow version 2.15.0
from keras.layers import LSTM,Dropout,Dense #Keras version 2.15.0 with Tensorflow version 2.15.0
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import EarlyStopping #Keras version 2.15.0 with Tensorflow version 2.15.0

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

data = pd.read_csv('MSFT_20yrs.csv')                                        #Read the dataset CSV file
print(data.head())

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.index = data['Date']                                             #Convert date string into timestamp

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

plt.figure(figsize=(16, 8))
plt.plot(data['High'], label='High')
plt.plot(data['Low'], label='Low')
plt.ylabel('Price (Billions) [USD]')                                  #Figure for high and low price of MSFT
plt.legend()
plt.gcf().autofmt_xdate()
plt.title(f"High and Low price of MSFT")
plt.show()

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

plt.figure(figsize=(16, 8))
plt.plot(data['Adj Close'], label='Adj Close')
plt.ylabel('Adj Close (billions) [USD]')                              #Figure for closing price of MSFT
plt.title('Closing price of MSFT')
plt.show()

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

plt.figure(figsize=(16, 8))
plt.plot(data['Volume'], label="Volume")
plt.ylabel('Volume of Sales')                                         #Figure for sales volume of MSFT
plt.title('Sales Volume of MSFT')
plt.show()

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

plt.figure(figsize=(16, 8))
data['Daily Return'] = data['Adj Close'].pct_change()

plt.plot(data['Daily Return'], label='Daily Return')
plt.title('Daily Return of MSFT')                                     #Figure for daily return of MSFT
plt.show()

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

closing_difference = data['Adj Close']

returns = closing_difference.pct_change()                             #Read the closing difference for each day
print(returns.head())

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

train_data = data[:3500]
test_data = data[3500:]

close_data = data.iloc[:, 5:6]
print(close_data.head())

training_set = close_data.iloc[:3500, :].values

test_set = close_data.iloc[3500:, :].values

scaler = MinMaxScaler(feature_range=(0,1))                            #Splitting the data into training and testing sets
training_set_scaled = scaler.fit_transform(training_set)              #Normalising the data

x_train = []
y_train = []

for i in range(120, 3500):
    x_train.append(training_set_scaled[i-120: i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train.shape, y_train.shape)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

model= Sequential()

model.add(LSTM(units=100,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(rate=0.2))
model.add(LSTM(units=50))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='Adam')

model.summary()                                                       #Building LSTM model

history = model.fit(x_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=2)

plt.figure(figsize=(16, 8))
plt.plot(history.history['loss'], label = 'loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

real_stock_price = test_data.iloc[:, 5:6].values

data_total= pd.concat([train_data['Adj Close'], test_data['Adj Close']],  axis=0)
inputs= data_total[len(data_total)-len(test_data)-60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(60, 186):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = model.predict(x_test)                         #Testing the model

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


plt.figure(figsize=(16,8))
plt.plot(real_stock_price, label='MSFT Closing Price')
plt.plot(predicted_stock_price, label='Predicted MSFT Closing price')
plt.title('MSFT Closing Actual vs Predicted')
plt.ylabel('Price (Billions) [USD]')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.title("Loss vs epochs")
plt.show()

model.save('C:/Users/Sam\'s Folder/FinalProject_LSTM_App/.venv/lstm_model.keras')


#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _