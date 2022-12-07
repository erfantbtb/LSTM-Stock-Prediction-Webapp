import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

class PlotDeep:
    def plot_series(x, y, format="-", start=0, end=None, 
                title=None, xlabel=None, ylabel=None, legend=None ):
        """
        Visualizes time series data

        Args:
        x (array of int) - contains values for the x-axis
        y (array of int or tuple of arrays) - contains the values for the y-axis
        format (string) - line style when plotting the graph
        start (int) - first time step to plot
        end (int) - last time step to plot
        title (string) - title of the plot
        xlabel (string) - label for the x-axis
        ylabel (string) - label for the y-axis
        legend (list of strings) - legend for the plot
        """
        plt.figure(figsize=(16, 12))
        if type(y) is tuple:
            for y_curr in y:
                plt.plot(x[start:end], y_curr[start:end], format)

            else:
                plt.plot(x[start:end], y[start:end], format)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)

        if legend:
            plt.legend(legend)

        plt.show()
    
    def plot_learning_process(metric, loss):
        epochs = range(len(loss)) 
        PlotDeep.plot_series(
            x=epochs, 
            y=(mae, loss), 
            title='Metric and Loss', 
            xlabel='Epochs',
            ylabel='Metric and Loss',
            legend=['Metric', 'Loss']
            )

        zoom_split = int(epochs[-1] * 0.2)
        epochs_zoom = epochs[zoom_split:]
        metric_zoom = metric[zoom_split:]
        loss_zoom = loss[zoom_split:]
        PlotDeep.plot_series(
            x=epochs_zoom, 
            y=(metric_zoom, loss_zoom), 
            title='Zoomed Metric and Loss', 
            xlabel='Epochs',
            ylabel='Metric and Loss',
            legend=['Metric_slow', 'Loss_slow']
            )

    def plot_predictions(time_train, time_test, train_forecast, test_forecast, x_train, x_test):
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.plot(time_train, train_forecast, label="forecast_train")
        ax.plot(time_test, test_forecast, label="forecast_test")
        ax.plot(time_train, x_train[:,3], label="real_train")
        ax.plot(time_test, x_test[:,3], label="real_test")
        ax.grid()
        ax.legend()        
        return fig, ax
