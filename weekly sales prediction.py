#Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random
import time
start_time = time.time()

timesteps_val=32
# Importing the training set
dataset_feature = pd.read_csv('../walmart_data/features.csv')
dataset_train = pd.read_csv('../walmart_data/train.csv')

dataset_feature=dataset_feature.drop(["MarkDown1","MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"],1)

feature_train=pd.merge(dataset_train, dataset_feature, how = 'left',left_on = ['Store', 'Date','IsHoliday'], right_on = ['Store', 'Date','IsHoliday'])

feature_train['Date'] = pd.to_datetime(feature_train['Date'])
feature_train['IsHoliday'] = feature_train['IsHoliday'].astype(int)
feature_train = feature_train.sort_values(by=['Date','Store','Dept'])
true_feature_train = feature_train.drop(['Date','Store','Dept'],1)
# print(true_feature_train["Weekly_Sales"][20])
# exit()
# print(feature_train.head())
# exit()
#feature_train.isnull().sum(axis=0)
#print(feature_train.head())
#exit()
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaled_input = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaled_input.fit_transform(true_feature_train)




# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(timesteps_val, 421570):
    #print(X_train.Shape)
    # x1=[]
    # for j in  range(i-20, i):
    #     x1= x1+ list(training_set_scaled[j,])
    #print(len(x1))
    X_train.append(training_set_scaled[i-timesteps_val:i, ])
    y_train.append(training_set_scaled[i, ])
    #print(y_train)
    # exit()
#print(X_train)
#exit()
X_train, y_train = np.array(X_train), np.array(y_train)

# print(X_train.shape,y_train.shape)
# exit()

# Reshaping
#X_train = np.reshape(X_train, (X_train.shape[0], (X_train.shape[1]//6),6))
# print(X_train[1])
# print(X_train.shape)
# exit()
dataset_test = pd.read_csv('../walmart_data/test.csv')
feature_test = pd.merge(dataset_test, dataset_feature, how = 'left',left_on = ['Store', 'Date','IsHoliday'], right_on = ['Store', 'Date','IsHoliday'])
feature_test['Date'] = pd.to_datetime(feature_test['Date'])
feature_test['IsHoliday'] = feature_test['IsHoliday'].astype(int)
feature_test = feature_test.sort_values(by=['Date','Store','Dept'])
date_store_dept = feature_test[['Date','Store','Dept']]
true_feature_test = feature_test.drop(['Date','Store','Dept'],1)



#print(inputs.shape)
#print(date_store_dept.values[1])


dataset_total = pd.concat((true_feature_train, true_feature_test),ignore_index=True, axis = 0,sort=True)
print(dataset_total.head())
# print(dataset_total["Weekly_Sales"][1])
#exit()
# print(dataset_total.isnull().sum(axis=0))
# for i in range(len(true_feature_train), len(dataset_total)):
#     x= random.randint(0,(len(true_feature_train)-1))
#     dataset_total.at[i,"Weekly_Sales"]= true_feature_train["Weekly_Sales"][x]
#     dataset_total.at[i,"Unemployment"]= true_feature_train["Unemployment"][x]
#     dataset_total.at[i,"CPI"]= true_feature_train["CPI"][x]

# print(dataset_total.shape)
# print(true_feature_train.shape)
# print(true_feature_test.shape)
# print("len")
# print(len(dataset_total))
# print(len(true_feature_train))
# print(len(true_feature_test))
# print(len(dataset_total)-len(true_feature_test)-20)

# print(dataset_total.isnull().sum(axis=0))
# exit()

inputs = dataset_total[len(dataset_total) - len(true_feature_test) - timesteps_val:]
# print(inputs.head())



#inputs = inputs.reshape(-1,6)
#print(inputs.shape)
#print(input)
inputs = scaled_input.transform(inputs)
X_test = []


# for i in range(20, inputs.shape[0]):
#     X_test.append(inputs[i-20:i, ])
# X_test = np.array(X_test)
# print(X_test.shape)
# exit()
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# n = date_store_dept["Date"]
#
# datetimeObj = datetime.datetime.strptime(n.values, '%d/%b/%Y')
# print(datetimeObj.values)
# exit()

dfST = date_store_dept['Date'].dt.date

# print(datetimeObj)
# print(X_test.shape)
# with open("submission.csv", "w") as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerow(["Id","Weekly_Sales"])
#     for i in range(len(true_feature_test)):
#         x = str(date_store_dept["Store"][i])+'_' + str(date_store_dept["Dept"][i])+ "_" +str(dfST[i])
#         #y = predicted_weekly_sales[i,0]
#         #x = str(x)+"," + str("jgfkdsbg")
#         y = "heelo"
#         writer.writerow([x,y])

# exit()
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 1, return_sequences = True, input_shape = (X_train.shape[1], 6)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 2, return_sequences = True))
regressor.add(Dropout(0.2))

# # Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 3, return_sequences = True))
regressor.add(Dropout(0.2))

# # Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 4, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 5))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 6))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = timesteps_val)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
#dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
#real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
#X_test = X_test[]
predicted_weekly_sales= np.zeros((115064,6))
for i in range(0, inputs.shape[0]-timesteps_val):
    X_test=(inputs[i:timesteps_val+i,])
    X_test=np.array(X_test)
    X_test=X_test.reshape(1, timesteps_val,6)
    predicted_weekly_sales[i,] = regressor.predict(X_test)
    inputs[timesteps_val+i,]=predicted_weekly_sales[i,]
predicted_weekly_sales = scaled_input.inverse_transform(predicted_weekly_sales)

# with open("submission.csv", "w") as f:
#     writer = csv.writer(f, delimiter=',')
#
#     writer.writerow("Id,Weekly_Sales")
#     for i in range(len(true_feature_test)):
#         x = str(date_store_dept["Store"][i])+'_' + str(date_store_dept["Dept"][i])+ "_" +str(date_store_dept["Date"][i])
#         y = predicted_weekly_sales[i,0]
#         x = str(x)+"," + str(y)
#         writer.writerow(x)


with open("submission14.csv", "w") as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["Id","Weekly_Sales"])
    for i in range(len(true_feature_test)):
        x = str(date_store_dept["Store"][i])+'_' + str(date_store_dept["Dept"][i])+ "_" +str(dfST[i])
        y = predicted_weekly_sales[i,0]
        #x = str(x)+"," + str("jgfkdsbg")
        # y = "heelo"
        writer.writerow([x,y])


# Visualising the results
# plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
# plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
# plt.title('Google Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Google Stock Price')
# plt.legend()
# plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
