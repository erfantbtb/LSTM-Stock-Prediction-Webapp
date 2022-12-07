import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import tensorflow as tf
import streamlit as st
import talib as ta 

from data_preprocess import *
from deep_model import *
from plot import *
from pattern import *

from tensorflow.keras.layers import Conv1D, LSTM, GRU, Dense, Input
from tensorflow.keras.losses import Huber, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

plt.style.use('seaborn-dark')

header = st.container()
dataset = st.container()
learn = st.container()
visualize = st.container()

with header:
    st.title("Welcome to my Forecasting WebApp!")
    st.markdown("##### In this app we use data with n window to predict tomorrows close value.")

with dataset:
    f = st.file_uploader("choose a file")
    print(f)

    if f is not None:       
        timeperiod = st.slider("choose your money flow index indicator input", min_value=10, max_value=50, value=14, step=1)
        data = pd.read_csv("data/"+f.name, parse_dates=True, index_col="time")

        data["MFI"] = ta.MFI(data["high"], data["low"], data["close"], data["volume"], timeperiod=timeperiod)
        data["SIN"] = np.sin(data["close"])
        data["COS"] = np.cos(data["close"])
        data["sma"] = data["close"].rolling(50).mean()
        data["hl"] = data["high"] - data["low"]
        data["oc"] = data["close"] - data["open"]
        data["roc"] = ta.ROC(data["close"], timeperiod=20)
        data["ob"] = ta.OBV(data["close"], data["volume"])
        data["atr"] = ta.ATR(data["high"], data["low"], data["close"], timeperiod=14)
        data["session"] = ta.HT_DCPHASE(data["close"])

        # pattern = Pattern(data)
        # data = pattern.candle_stick()
        # print(len(data.columns))

        data.dropna(inplace=True)
        data.drop(columns=data.columns[0], axis=1, inplace=True)
        st.table(data.head())

with learn:
    col_split_time, col_window_size, col_past_days = st.columns(3)
    split_time = col_split_time.text_input("please write your split value", "1200")
    window_size = col_window_size.slider("choose your window size", min_value=10, max_value=60, value=28)
    n_change = col_past_days.text_input("please determine your pct_change value", "28")

    pp = PreprocessingDeep(data, int(split_time), window_size, n_change=int(n_change))
    xtrain, ytrain, xtest, ytest, time_train, time_test = pp.preprocess(method="min_max_scaler")

    print(xtrain.shape)

    dm = ModelDeep
    model_lstm = dm.bidirectional_lstm(xtrain)
    model_lstm.compile(loss=Huber(), optimizer=Adam(learning_rate=1e-2), metrics=["mse"])
    model_lstm.fit(xtrain, ytrain, epochs=100, validation_data=(xtest, ytest), shuffle=True, batch_size=128)

    train_forecast = model_lstm.predict(xtrain).reshape(-1)
    test_forecast = model_lstm.predict(xtest).reshape(-1)

with visualize:
    scaled_data = (data["close"] - data["close"].min())/(data["close"].max() - data["close"].min())
    train_forecast_scaled = train_forecast*(data["close"].max() - data["close"].min()) + data["close"].min()
    test_forecast_scaled = test_forecast*(data["close"].max() - data["close"].min()) + data["close"].min()

    fig, ax = plt.subplots(figsize=(20, 16))
    ax.plot(time_train, train_forecast_scaled, label="forecast_train")
    ax.plot(time_test, test_forecast_scaled, label="forecast_test", color="red")
    ax.plot(data.index, data["close"], label="real")
    ax.grid()
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.set_title("Forecasting Plot")
    st.pyplot(fig)
