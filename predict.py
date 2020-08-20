
import math
import yfinance as yf
import numpy as np
import pandas as pd
import pandas_datareader as pdd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense , LSTM
import matplotlib.pyplot as plt
df = pdd.DataReader("USD",data_source="yahoo")
plt.figure(figsize=(16,8))
plt.figure(figsize=(16,8))
plt.title("Close Price History")
plt.plot(df["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD($)",fontsize=18)
data = df.filter(["Close"])
dataset=data.values
len(dataset)
training_data_size = math.ceil(len(dataset)*.7)
training_data_size
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data
train_data =scaled_data[0:training_data_size,:]
x_train = []
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=60:
        print(x_train)
        print(y_train)
x_train,y_train =np.array(x_train),np.array(y_train)
x_train =np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape= (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mean_squared_error")
model.fit(x_train,y_train,batch_size=1 ,epochs=1)
test_data = scaled_data[training_data_size - 60:,:]
x_test=[]
y_test= dataset[training_data_size:,:]
for i in range (60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
x_test =np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predictions =model.predict(x_test)
predictions =scaler.inverse_transform(predictions)
rmse =np.sqrt(np.mean(predictions-y_test**2))
train = data[:training_data_size]
valid = data[training_data_size:]
valid["predictions"]=predictions
plt.figure(figsize=(16,8))
plt.title("MClose Price History")
plt.plot(train["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD($)",fontsize=18)
plt.plot(valid[["Close","predictions"]])
plt.legend(["Train","Val","predictions"],loc="lower right")
plt.show







