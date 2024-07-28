# Import Required Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow

from datetime import datetime as dt
from datetime import timedelta
from pylab import rcParams

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error

# Import the Dataset

df = pd.read_csv('/Users/mathieujosserand/PycharmProjects/Stock Prediction and Forecasting Using LSTM/Ressources/TTE.csv')
# df.head()

# Analysis of data

df2 = df.reset_index()['Close']
# plt.plot(df2)

# Data Preprocessing

scaler = MinMaxScaler()
df2 = scaler.fit_transform(np.array(df2).reshape(-1,1))
# print(df2.shape)

# Train-Test Split

train_size = int(len(df2)*0.65)
test_size = len(df2) - train_size
train_data,test_data = df2[0:train_size,:],df2[train_size:len(df2),:1]

def create_dataset(dataset, time_step):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
                   a = dataset[i:(i+time_step),0]
                   dataX.append(a)
                   dataY.append(dataset[i + time_step,0])
    return np.array(dataX),np.array(dataY)

# calling the create dataset function to split the data into
# input output datasets with time step 100
time_step = 100
prediction_window = 90

X_train,Y_train =  create_dataset(train_data,time_step)
X_test,Y_test =  create_dataset(test_data,time_step)

X_future = np.zeros(shape=(1, time_step))
X_future[0,0:time_step] = X_test[len(X_test)-1,:]
np.array(X_future).reshape(-1, 1)

# checking values
# print(X_train.shape)
# print(X_train)
# print(X_test.shape)
# print(Y_test.shape)

# Creating and fitting LSTM model

model = Sequential()
model.add(LSTM(50,return_sequences = True,input_shape = (X_train.shape[1],1)))
model.add(LSTM(50,return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error',optimizer = 'adam')

model.summary()

model.fit(X_train,Y_train,validation_data = (X_test,Y_test),epochs = 100,batch_size = 64,verbose = 1)

# Prediction and checking performance matrix

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Next month's predictions
future_predict = np.empty(shape=(0,0))

for i in range(prediction_window):

    future_predict = np.append(future_predict, model.predict(X_future))
    # print(future_predict)
    X_future = np.append(X_future, future_predict[i])
    X_future = np.delete(X_future, 0)
    X_future = np.reshape(X_future, (1, 100))
    # print(X_future)

future_predict = np.reshape(future_predict,(-1,1))

# transform to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
future_predict = scaler.inverse_transform(future_predict)

# RMSE performance matrix
print(math.sqrt(mean_squared_error(Y_train,train_predict)))
print(math.sqrt(mean_squared_error(Y_test,test_predict)))

# Graph Plotting

look_back = 100

trainPredictPlot = np.empty((len(df2) + prediction_window,1))
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back : len(train_predict)+look_back,:] = train_predict

testPredictPlot = np.empty((len(df2) + prediction_window,1))
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict)+(look_back)*2 + 1 : len(df2) - 1,:] = test_predict

futurePredictPlot = np.empty((len(df2) + prediction_window,1))
futurePredictPlot[:,:] = np.nan
futurePredictPlot[len(df2) : len(futurePredictPlot),:] = future_predict

trainPlot = np.empty((len(df2) + prediction_window,1))
trainPlot[:,:] = np.nan
trainPlot[:len(df2),:] = scaler.inverse_transform(df2)

# Creating list of dates for x-axis
datelist_df = list(df['Date'])
datelist_df = [dt.strptime(date, '%Y-%m-%d').date() for date in datelist_df]

# initializing date
start_date = datelist_df[len(datelist_df) - 1] + timedelta(days=1)

# adding future dates
for day in range(prediction_window):
    date = start_date + timedelta(days=day)
    datelist_df.append(date)

rcParams['figure.figsize'] = 14, 5

plt.plot(datelist_df, trainPlot, color='skyblue', label='Actual Stock Price')
plt.plot(datelist_df, trainPredictPlot, color='orange', label='Training predictions')
plt.plot(datelist_df, testPredictPlot, color='yellowgreen', label='Predicted Stock Price')
plt.plot(datelist_df, futurePredictPlot, color='red', label="Next Month's Predicted Stock Price")


plt.legend(shadow=True)
plt.title('Predictions and Actual Stock Prices', family='Arial', fontsize=12)
plt.xlabel('Timeline', family='Arial', fontsize=10)
plt.ylabel('Stock Price Value', family='Arial', fontsize=10)
plt.xticks(rotation=45, fontsize=8)

plt.show()