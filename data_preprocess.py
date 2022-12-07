import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler, normalize

class PreprocessingDeep:
    def __init__(self, data, split_time, window_size, method_scale="MinMaxScaler", n_change=28):
        self.data = data 
        self.split_time = split_time 
        self.window_size = window_size 
        self.method = method_scale
        self.n_change = n_change

    def windowed_dataset(series, window_size):
        """Generates dataset windows

        Args:
        series (array of float) - contains the values of the time series
        window_size (int) - the number of time steps to include in the feature

        Returns:
        feature_set (NP array) - NP array containing time windows 
        label_set (NP array) - NP array containing labels
        """
    
        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(series)
        
        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

        # Create features and labels 
        feature_set = dataset.map(lambda window: window[:-1])
        label_set = dataset.map(lambda window: window[-1, 3])
        
        return feature_set, label_set 

    def preprocess(self, method="pct_change"):
        """Gives windowed train and test data to us

        Returns:
        train_windowed_feature (NP array) - NP array containing time windows for train
        train_windowed_label (NP array) - NP array containing labels for train
        test_windowed_feature (NP array) - NP array containing time windows for test
        test_windowed_label (NP array) - NP array containing labels for test        
        """
        #split data to train and test
        if method == "min_max_scaler":
            self.data = (self.data - self.data.min(axis=0))/(self.data.max(axis=0) - self.data.min(axis=0))

        else:
            self.data = self.data.pct_change(self.n_change)
            self.data.dropna(inplace=True)

        # classification_target = pd.DataFrame(np.where(self.data["close"].pct_change()>0, 1, 0))

        time = self.data.index
        time_train = time[:self.split_time]
        x_train = self.data[:self.split_time]
        time_test = time[self.split_time:]
        x_test = self.data[self.split_time:]
        # ctr = classification_target[:self.split_time]
        # cte = classification_target[self.split_time:]

        #create windowed data from train
        train_windowed_feature, train_windowed_label = PreprocessingDeep.windowed_dataset(x_train, self.window_size)

        train_windowed_feature_list = []
        train_windowed_label_list = []

        for element in train_windowed_feature:
            train_windowed_feature_list.append(element.numpy())
            
        for element in train_windowed_label:
            train_windowed_label_list.append(element.numpy())

        train_windowed_feature = np.array(train_windowed_feature_list)
        train_windowed_label = np.array(train_windowed_label_list)

        #create windowed data from test
        test_windowed_feature, test_windowed_label = PreprocessingDeep.windowed_dataset(x_test, self.window_size)

        test_windowed_feature_list = []
        test_windowed_label_list = []

        for element in test_windowed_feature:
            test_windowed_feature_list.append(element.numpy())
            
        for element in test_windowed_label:
            test_windowed_label_list.append(element.numpy())

        test_windowed_feature = np.array(test_windowed_feature_list)
        test_windowed_label = np.array(test_windowed_label_list)
        
        return train_windowed_feature,train_windowed_label,test_windowed_feature,test_windowed_label,time_train[self.window_size:],time_test[self.window_size:]

