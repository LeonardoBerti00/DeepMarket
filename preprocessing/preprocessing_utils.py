import pandas as pd
import numpy as np
import os


def z_score_orderbook(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """ DONE: remember to use the mean/std of the training set, to z-normalize the test set. """
    if (mean_size is None) or (std_size is None):
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()

    #do the same thing for prices
    if (mean_prices is None) or (std_prices is None):
        mean_prices = data.iloc[:, 0::2].stack().mean() #price
        std_prices = data.iloc[:, 0::2].stack().std() #price

    #apply the z score to the original data
    data.iloc[:, 1::2] = (data.iloc[:, 1::2] - mean_size) / std_size
    #do the same thing for prices
    data.iloc[:, 0::2] = (data.iloc[:, 0::2] - mean_prices) / std_prices

    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    return data, mean_size, mean_prices, std_size,  std_prices


def normalize_messages(data, mean_size=None, mean_prices=None, std_size=None,  std_prices=None, mean_time=None, std_time=None):

    #apply z score to prices and size column
    if (mean_size is None) or (std_size is None):
        mean_size = data["size"].mean()
        std_size = data["size"].std()

    if (mean_prices is None) or (std_prices is None):
        mean_prices = data["price"].mean()
        std_prices = data["price"].std()

    if (mean_time is None) or (std_time is None):
        mean_time = data["time"].mean()
        std_time = data["time"].std()

    #apply the z score to the original data
    data["time"] = (data["time"] - mean_time) / std_time
    data["size"] = (data["size"] - mean_size) / std_size
    data["price"] = (data["price"] - mean_prices) / std_prices

    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    data["event_type"] = data["event_type"]-1.0

    return data, mean_size, mean_prices, std_size,  std_prices, mean_time, std_time
