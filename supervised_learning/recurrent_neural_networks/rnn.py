# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set = sc.fit_transform(training_set)

# Getting the inputs and outputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshaping
X_train = np.reshape(X_train , (1257 , 1 , 1))

# Building the RNN

# Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# initializing the RNN
regressor = Sequential()

# adding input and hidden layers using LSTM
regressor.add(LSTM(units = 4,
                   activation = 'sigmoid',
                   input_shape = (None , 1)))

# adding output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam',
                  loss = 'mean_squared_error')

# Fitting RNN to training set
regressor.fit(X_train, y_train, batch_size =32, epochs = 200)

# Making predictions and visualising the results

#Getting real stock price
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting predicted stock price
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20,1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real stock price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted stock price')
plt.title("Real vs Predicted stock price")
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()

# Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))
