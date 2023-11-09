import pandas as pd
import numpy as np
import os


def z_score_orderbook(data, mean_vol=None, mean_prices=None, std_vol=None,  std_prices=None):
    """ DONE: remember to use the mean/std of the training set, to z-normalize the test set. """
    if (mean_vol is None) or (std_vol is None):
        mean_vol = data.iloc[:, 1::2].stack().mean()
        std_vol = data.iloc[:, 1::2].stack().std()

    #do the same thing for prices
    if (mean_prices is None) or (std_prices is None):
        mean_prices = data.iloc[:, 0::2].stack().mean() #price
        std_prices = data.iloc[:, 0::2].stack().std() #price

    #apply the z score to the original data
    data.iloc[:, 1::2] = (data.iloc[:, 1::2] - mean_vol) / std_vol
    #do the same thing for prices
    data.iloc[:, 0::2] = (data.iloc[:, 0::2] - mean_prices) / std_prices

    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    return data, mean_vol, mean_prices, std_vol,  std_prices


def normalize_messages(data, mean_vol=None, mean_prices=None, std_vol=None,  std_prices=None):

    #apply z score to prices and volume column
    if (mean_vol is None) or (std_vol is None):
        mean_vol = data["volume"].mean()
        std_vol = data["volume"].std()

    if (mean_prices is None) or (std_prices is None):
        mean_prices = data["price"].mean()
        std_prices = data["price"].std()

    data["volume"] = (data["volume"] - mean_vol) / std_vol
    data["price"] = (data["price"] - mean_prices) / std_prices

    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    data["direction"] = data["direction"].replace(-1.0, 0.0)
    data["event_type"] = data["event_type"]-1.0
    return data, mean_vol, mean_prices, std_vol,  std_prices
