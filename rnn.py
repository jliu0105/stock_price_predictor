# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
data_scale = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = data_scale.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train_val = []
y_train_val = []
for i in range(60, 1258):
    X_train_val.append(training_set_scaled[i-60:i, 0])
    y_train_val.append(training_set_scaled[i, 0])
X_train_val, y_train_val = np.array(X_train_val), np.array(y_train_val)

# Reshaping
X_train_val = np.reshape(X_train_val, (X_train_val.shape[0], X_train_val.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor_mod = Sequential()

# Add first LSTM layer and some Dropout regularisation
regressor_mod.add(LSTM(units = 55, return_sequences = True, input_shape = (X_train_val.shape[1], 1)))
regressor_mod.add(Dropout(0.2))

# Add a second LSTM layer and some Dropout regularisation
regressor_mod.add(LSTM(units = 55, return_sequences = True))
regressor_mod.add(Dropout(0.2))

# Add a third LSTM layer and some Dropout regularisation
regressor_mod.add(LSTM(units = 55, return_sequences = True))
regressor_mod.add(Dropout(0.2))

# Add a fourth LSTM layer and some Dropout regularisation
regressor_mod.add(LSTM(units = 55))
regressor_mod.add(Dropout(0.2))

# Add the output layer
regressor_mod.add(Dense(units = 1))

# Compile RNN
regressor_mod.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit RNN to Training set
regressor_mod.fit(X_train_val, y_train_val, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
total_dataset = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = total_dataset[len(total_dataset) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = data_scale.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predict_stock_price = regressor_mod.predict(X_test)
predict_stock_price = data_scale.inverse_transform(predict_stock_price)

# Graph the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predict_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()