import pandas as pd
import numpy as np
DJ = pd.read_csv("Dow_Jones.csv")
DJ_data = DJ[DJ['Close'] != 0]

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Time series plot of Dow Jones data
figure(figsize=(15, 10), dpi=80)
DJ_x = DJ_data.index
DJ_y = DJ_data['Close']
plt.plot(DJ_x,DJ_y)
plt.show()

# install the package which is necessary for frequency decomposition
! pip install EMD-signal

DJ_Signal = DJ_data['Close'].to_numpy()
DJ_T = DJ_data.index.to_numpy()

from PyEMD import EEMD
# Assign EEMD to `eemd` variable
eemd = EEMD()
# Say we want detect extrema using parabolic method
emd = eemd.EMD
emd.extrema_detection="parabol"
# Execute EEMD on Signal
IMFs = eemd.eemd(DJ_Signal, DJ_T)

nIMFs = len(IMFs)
# visualize the subsequences getting from the original signal
plt.figure(figsize=(12,9))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(DJ_T, DJ_Signal, 'r')

for n in range(nIMFs):
  plt.subplot(nIMFs+1, 1, n+2)
  plt.plot(DJ_T, IMFs[n], 'g')
  plt.ylabel("IMF %i" %(n+1))
  plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time")
plt.tight_layout()
plt.savefig('eemd_example', dpi=120)
plt.show()

# Think about is to use linear regression to get coefficients for each subsequence as the weights to perform weighted sum during the fusion.
column_names = ["IMF1", "IMF2", "IMF3", "IMF4", "IMF5", "IMF6", "IMF7", "IMF8", "IMF9", "IMF10", "IMF11"]
IMF_df = pd.DataFrame(columns = column_names)

for i in range(len(column_names)):
  IMF_df[column_names[i]] = IMFs[i]
  
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(IMF_df,DJ_Signal)
coefficient = list(lin_reg.coef_) # Weights used in the following fusion technique
intercept = lin_reg.intercept_

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

# Create dataset by looking specific number of steps back
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def perform_LSTM(dataset, look_back, layer=4):
  
  dataset = dataset.astype('float32')
  dataset = np.reshape(dataset, (-1, 1))
  
  # Normalize the data -- using Min and Max values in each subsequence to normalize the values
  scaler = MinMaxScaler()
  dataset = scaler.fit_transform(dataset)
  
  # Split data into training and testing set
  train_size = int(len(dataset) * 0.8)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size, :], dataset[train_size:, :]
  
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)

  trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
  
  # create and fit the LSTM network
  model = Sequential()
  model.add(LSTM(layer, input_shape=(1, look_back)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

  # make predictions
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)

  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict)
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict)
  testY = scaler.inverse_transform([testY])
  testing_error = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

  return testPredict, testY, testing_error

testPredict_1, testY_1, testing_error_1 = perform_LSTM(IMFs[0], 5)
testing_error_1

testPredict_2, testY_2, testing_error_2 = perform_LSTM(IMFs[1], 5)
testing_error_2

testPredict_3, testY_3, testing_error_3 = perform_LSTM(IMFs[2], 5)
testing_error_3

testPredict_4, testY_4, testing_error_4 = perform_LSTM(IMFs[3], 5)
testing_error_4

testPredict_5, testY_5, testing_error_5 = perform_LSTM(IMFs[4], 5)
testing_error_5

testPredict_6, testY_6, testing_error_6 = perform_LSTM(IMFs[5], 5)
testing_error_6

testPredict_7, testY_7, testing_error_7 = perform_LSTM(IMFs[6], 5)
testing_error_7

testPredict_8, testY_8, testing_error_8 = perform_LSTM(IMFs[7], 5)
testing_error_8

testPredict_9, testY_9, testing_error_9 = perform_LSTM(IMFs[8], 1)
testing_error_9

testPredict_10, testY_10, testing_error_10 = perform_LSTM(IMFs[9], 1)
testing_error_10

testPredict_11, testY_11, testing_error_11 = perform_LSTM(IMFs[10], 1)
testing_error_11

IMF_predict_list = []
IMF_predict_list = IMF_predict_list + [testPredict_1, testPredict_2, testPredict_3, testPredict_4, 
                                       testPredict_5, testPredict_6, testPredict_7, testPredict_8, testPredict_9, 
                                       testPredict_10, testPredict_11]

# Because the subsequences do not have the same look_back values
# What we are doing here is to make sure we combine the corresponding values in the same length. 
start_list = [0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4] 
IMF_pred_equal_length = []
for i in range(len(IMF_predict_list)):
  IMF_pred_equal_length.append(IMF_predict_list[i][start_list[i]:,0])
  print(len(IMF_pred_equal_length[-1]))
  
# This nested for loop is used to generate final prediction by adding up the corresponding values in each subsequence
final_prediction = []
for i in range(len(IMF_pred_equal_length[0])):
  element = 0
  for j in range(len(IMF_pred_equal_length)):  
    element += IMF_pred_equal_length[j][i] * coefficient[j]
  final_prediction.append(element)

  
DJ_data = DJ_y.astype('float32')
DJ_data = np.reshape(DJ_data.to_numpy(), (-1, 1))

train_size = int(len(DJ_data) * 0.8)
test_size = len(DJ_data) - train_size
DJ_train, DJ_test = DJ_data[0:train_size], DJ_data[train_size:]
DJ_testX, DJ_testY = create_dataset(DJ_test, 5)

# Calculate the RMSE
math.sqrt(mean_squared_error(DJ_testY.tolist(), final_prediction))

figure(figsize=(15, 10), dpi=80)
x = np.linspace(1, len(final_prediction)+1, len(final_prediction), endpoint=True)
# plot lines
plt.plot(x, final_prediction, label = "Predicted Value")
plt.plot(x, DJ_testY.tolist(), label = "Actual Value")
plt.legend()
plt.show()
