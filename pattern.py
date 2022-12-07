import talib
import pandas as pd 
import yfinance as yf 
import numpy as np 
import matplotlib.pyplot as plt 

class Pattern:
    def __init__(self, data):
        self.data = data
        self.candle_names = talib.get_function_groups()['Pattern Recognition']

    def candle_stick(self):
        # extract OHLC 
        op = self.data['open']
        hi = self.data['high']
        lo = self.data['low']
        cl = self.data['close']

        # create columns for each pattern
        for candle in self.candle_names:
            # below is same as;
            # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
            self.data[candle] = getattr(talib, candle)(op, hi, lo, cl)

        return self.data 
