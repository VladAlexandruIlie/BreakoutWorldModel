##
# LSTM example. Sien wave prediction
# source: https://github.com/kmsravindra/ML-AI-experiments/blob/master/AI/LSTM-time_series/LSTM%20-%20Sine%20wave%20predictor.ipynb
##
import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

series = pd.read_csv('sine-wave.csv', header=None )
#print("head ", series.head())
#print("shape: ", series.shape)

plt.figure(figsize=(20,6))
plt.plot(series.values)
plt.show()

plt.figure(figsize=(20,6))
plt.plot(series.values[:50])
plt.show()

# normalize features -
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(series.values)
series = pd.DataFrame(scaled)

window_size = 50

series_s = series.copy()
for i in range(window_size):
    series = pd.concat([series, series_s.shift(-(i+1))], axis = 1)
series.dropna(axis=0, inplace=True)
#print("head ", series.head())
#print("shape: ", series.shape)

nrow = round(0.8*series.shape[0])

train = series.iloc[:nrow, :]
test = series.iloc[nrow:,:]

train = shuffle(train)
train_X = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]
train_X = train_X.values
train_y = train_y.values
test_X = test_X.values
test_y = test_y.values

print("train_X: ", train_X.shape, " --> ")
print("train_y: ", train_y.shape)
print("test_X: ", test_X.shape, " --> ")
print("test_y: ", test_y.shape)

train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)

print("train_X: ", train_X.shape)
print("train_y: ", train_y.shape)
print("test_X: ", test_X.shape)
print("test_y: ", test_y.shape)

# Define the LSTM model
model = Sequential()
model.add(LSTM(input_shape = (50,1), output_dim= 50, return_sequences = True))
#model.add(LSTM(input_shape = (50,1), return_sequences = True, output_dim= 50))
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="adam")
model.summary()

start = time.time()
model.fit(train_X,train_y,batch_size=512,nb_epoch=3,validation_split=0.1)
print("> Compilation Time : ", time.time() - start)

# Doing a prediction on all the test data at once
preds = model.predict(test_X)
preds = scaler.inverse_transform(preds)

#actuals = scaler.inverse_transform(test_y)
actuals = test_y
mean_squared_error(actuals,preds)

plt.plot(actuals)
plt.plot(preds)
plt.show()

plt.plot(preds)
plt.show()


def moving_test_window_preds(n_future_preds):

    ''' n_future_preds - Represents the number of future predictions we want to make
                         This coincides with the number of windows that we will move forward
                         on the test data
    '''
    preds_moving = []                                    # Use this to store the prediction made on each test window
    moving_test_window = [test_X[0,:].tolist()]          # Creating the first test window
    moving_test_window = np.array(moving_test_window)    # Making it an numpy array

    for i in range(n_future_preds):
        preds_one_step = model.predict(moving_test_window) # Note that this is already a scaled prediction so no need to rescale this
        preds_moving.append(preds_one_step[0,0]) # get the value from the numpy 2D array and append to predictions
        preds_one_step = preds_one_step.reshape(1,1,1) # Reshaping the prediction to 3D array for concatenation with moving test window
        moving_test_window = np.concatenate((moving_test_window[:,1:,:], preds_one_step), axis=1) # This is the new moving test window, where the first element from the window has been removed and the prediction  has been appended to the end

    return preds_moving

preds_moving = moving_test_window_preds(500)
plt.plot(actuals)
plt.plot(preds_moving)
plt.show()

preds_moving = moving_test_window_preds(10)
plt.plot(actuals)
plt.plot(preds_moving)
plt.show()

preds_moving = moving_test_window_preds(2)
plt.plot(actuals)
plt.plot(preds_moving)
plt.show()
