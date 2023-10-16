
import pandas as pd
import numpy as np
from enum import Enum
from matplotlib import pyplot as plt

def stationary_normalize_data(data, normalization_mean=None, normalization_std=None):
    """ DONE: remember to use the mean/std of the training set, to z-normalize the test set. """

    col_choice = {"volumes": get_volume_column_name(data.columns),
                  "prices":  get_price_column_name(data.columns)}

    print("Normalization... (using means", normalization_mean, "and stds", normalization_std, ")")

    means_dict, stds_dict = dict(), dict()
    for col_name in col_choice:
        cols = col_choice[col_name]

        if normalization_mean is None and normalization_std is None:
            means_dict[col_name] = data.loc[:, cols].stack().mean()
            stds_dict[col_name] = data.loc[:, cols].stack().std()

        elif normalization_mean is not None and normalization_std is not None:
            means_dict[col_name] = normalization_mean[col_name]
            stds_dict[col_name] = normalization_std[col_name]

        data.loc[:, cols] = (data.loc[:, cols] - means_dict[col_name]) / stds_dict[col_name]
        data.loc[:, cols] = data.loc[:, cols]


    data = data.fillna(method="bfill")
    data = data.fillna(method="ffill")
    return data, means_dict, stds_dict

def get_volume_column_name(columns):
    return [f for f in columns if "vbuy" in f or "vsell" in f]


def get_price_column_name(columns):
    return [f for f in columns if "pbuy" in f or "psell" in f]


def get_sell_column_name(columns):
    return [f for f in columns if "vsell" in f or "psell" in f]


def get_buy_column_name(columns):
    return [f for f in columns if "pbuy" in f or "vbuy" in f]
