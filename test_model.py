from data_preprocess import * 
from deep_model import * 
from plot import * 
import numpy as np 
import pandas as pd
import talib as ta
import os
import tensorflow as tf 
from tensorflow.keras.optimizers import RMSprop, Adam, SGD 
from tensorflow.keras.losses import Huber, MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy
from sklearn.metrics import accuracy_score, confusion_matrix

list_data = ["forecast price/data/BTC_USDT_1d.csv"]
list_acc = []
report = pd.DataFrame()

for symbol in list_data:
    print(symbol)
    for i in range(14, 15):
        data = pd.read_csv(symbol, parse_dates=True, index_col="time")
        # data["ROC"] = ta.ROC(data["close"], timeperiod=20)
        # data["DIFF"] = data["high"] - data["low"]

        data.dropna(inplace=True)
        preprocess = PreprocessingDeep(data, int(len(data)*0.8), i)
        xtrain, ytrain, xtest, ytest, time_train, time_test, ctr, cte = preprocess.preprocess()
        print(ctr)

        #load our model
        forecast_models = ModelDeep
        gru_model = forecast_models.lstm(xtrain) 
        gru_model.summary()

        #compile our model 
        gru_model.compile(loss=BinaryCrossentropy(), optimizer=Adam(5e-3), metrics=["accuracy"])
        gru_model.fit(xtrain, ctr[14:], epochs=100, shuffle=True, validation_data=(xtest, cte[14:]), batch_size=128)

        #make prediction 
        # test_prediction = gru_model.predict(xtest).reshape(-1, )
        # train_prediction = gru_model.predict(xtrain).reshape(-1, )
        # print(test_prediction.shape, ytest.shape)

        #plot test and prediction 
        # plt.figure(figsize=(16, 12))
        # plt.plot(time_train, ytrain, label="actual train")
        # plt.plot(time_train, train_prediction, label="prediction train")
        # plt.plot(time_test, ytest, label="actual test")
        # plt.plot(time_test, test_prediction, label="prediction test")
        # plt.grid()
        # plt.legend()
        # plt.savefig(symbol+"_lstm.png")

        #accuracy of trend 
#         prediction_df = pd.DataFrame(test_prediction)
#         test_df = pd.DataFrame(ytest)

#         prediction_df = prediction_df.pct_change()
#         test_df = test_df.pct_change()

#         prediction_df.dropna(inplace=True)
#         test_df.dropna(inplace=True)

#         prediction_arr = prediction_df.to_numpy()
#         test_arr = test_df.to_numpy()

#         prediction_arr = np.where(prediction_arr>0, 1, 0)
#         test_arr = np.where(test_arr>0, 1, 0)
#         acc = accuracy_score(prediction_arr, test_arr)
#         list_acc.append([acc, symbol, i])

#     report = pd.concat([report, pd.DataFrame(list_acc)])

# report.to_csv("prediction_acc_mt.csv") 