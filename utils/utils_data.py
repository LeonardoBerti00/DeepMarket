import pandas as pd
import numpy as np
import os

import torch

import constants as cst


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
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)
    # order_type = 0 -> limit order
    # order_type = 1 -> cancel order
    # order_type = 2 -> market order

    return data, mean_size, mean_prices, std_size,  std_prices, mean_time, std_time


def reset_indexes(dataframes):
    # reset the indexes of the messages and orderbooks
    for i in range(len(dataframes)):
        dataframes[i][0] = dataframes[i][0].reset_index(drop=True)
        dataframes[i][1] = dataframes[i][1].reset_index(drop=True)
    return dataframes


def preprocess_data(dataframes, n_lob_levels):
    dataframes = reset_indexes(dataframes)

    # take only the first n_lob_levels levels of the orderbook and drop the others
    for i in range(len(dataframes)):
        dataframes[i][1] = dataframes[i][1].iloc[:, :n_lob_levels * cst.LEN_LEVEL]

    # take the indexes of the dataframes that are of type 2, 5, 6, 7 and drop it
    for i in range(len(dataframes)):
        indexes_to_drop = dataframes[i][0][dataframes[i][0]["event_type"].isin([2, 5, 6, 7])].index
        dataframes[i][0] = dataframes[i][0].drop(indexes_to_drop)
        dataframes[i][1] = dataframes[i][1].drop(indexes_to_drop)

    dataframes = reset_indexes(dataframes)

    # drop index column in messages
    for i in range(len(dataframes)):
        dataframes[i][0] = dataframes[i][0].drop(columns=["order_id"])

    # do the difference of time row per row in messages and subsitute the values with the differences
    for i in range(len(dataframes)):
        first = dataframes[i][0]["time"].iloc[0]
        dataframes[i][0]["time"] = dataframes[i][0]["time"].diff()
        dataframes[i][0].loc[0, "time"] = first - 34200

    # add depth column to messages
    for i in range(len(dataframes)):
        dataframes[i][0]["depth"] = 0

    # we compute the depth of the orders with respect to the orderbook
    for i in range(0, len(dataframes)):
        for j in range(1, dataframes[i][0].shape[0]):
            order_price = dataframes[i][0]["price"].iloc[j]
            direction = dataframes[i][0]["direction"].iloc[j]
            type = dataframes[i][0]["event_type"].iloc[j]
            if type == 1:
                index = j
            else:
                index = j - 1
            if direction == 1:
                bid_side = dataframes[i][1].iloc[index, 2::4]
                depth = np.where(bid_side == order_price)[0][0]
            else:
                ask_side = dataframes[i][1].iloc[index, 0::4]
                depth = np.where(ask_side == order_price)[0][0]

            dataframes[i][0].loc[j, "depth"] = depth

    # we eliminate the first row of every dataframe because we can't deduce the depth
    for i in range(len(dataframes)):
        dataframes[i][0] = dataframes[i][0].iloc[1:, :]
        dataframes[i][1] = dataframes[i][1].iloc[1:, :]

    # we transform the execution of a sell limit order in a buy market order and viceversa
    for i in range(len(dataframes)):
        dataframes[i][0]["direction"] = dataframes[i][0]["direction"] * dataframes[i][0]["event_type"].apply(
            lambda x: -1 if x == 4 else 1)

    return dataframes[0][1], dataframes[0][0]


def unnormalize(x, mean, std):
    return x * std + mean


def one_hot_encode_type(data):
    encoded_data = torch.zeros(data.shape[0], data.shape[1] + 2, dtype=torch.float32)
    encoded_data[:, 0] = data[:, 0]
    # encoding order type
    one_hot_order_type = torch.nn.functional.one_hot((data[:, 1]).to(torch.int64), num_classes=3).to(
        torch.float32)
    encoded_data[:, 1:4] = one_hot_order_type
    encoded_data[:, 4:] = data[:, 2:]
    return encoded_data



'''
def to_original_lob(event_and_lob, seq_size):
    lob = event_and_lob[:, cst.LEN_EVENT:]

    lob[:, 0::2] = unnormalize(lob[:, 0::2], cst.TSLA_LOB_MEAN_PRICE_10, cst.TSLA_LOB_STD_PRICE_10)
    lob[:, 1::2] = unnormalize(lob[:, 1::2], cst.TSLA_LOB_MEAN_SIZE_10, cst.TSLA_LOB_STD_SIZE_10)
    lob = lob[seq_size - 2:, :]
    #assert (generated_events.shape[0]+1 == lob.shape[0])
    # round price and size

    lob[:, 0::2] = np.around(lob[:, 0::2], decimals=0)
    lob[:, 1::2] = np.around(lob[:, 1::2], decimals=0)

    return lob


def check_constraints(file_recon, file_lob, seq_size):
    generated_events = np.load(file_recon)
    event_and_lob = np.load(file_lob)
    lob = to_original_lob(event_and_lob, seq_size)
    print()
    print("numbers of lob ", lob.shape[0])
    print("numbers of gen events ", generated_events.shape[0])
    num_violations_price_del = 0
    num_violations_price_exec = 0
    num_violations_size = 0
    num_non_violations_price_del = 0
    num_non_violations_price_exec = 0
    num_add = 0
    for i in range(generated_events.shape[0]):
        price = generated_events[i, 3]
        size_event = generated_events[i, 2]
        type = generated_events[i, 1]
        if (type == 2 or type == 3 or type == 4):    #it is a cancellation
            #take the index of the lob with the same value of the price
            index = np.where(lob[i, :] == price)[0]
            if (index.shape[0] == 0):
                if (type == 2 or type == 3):
                    num_violations_price_del += 1
                else:
                    num_violations_price_exec += 1
            else:
                size_limit_order = lob[i, index[0] + 1]
                if (size_limit_order < size_event):
                    num_violations_size += 1
                else:
                    if (type == 2 or type == 3):
                        num_non_violations_price_del += 1
                    else:
                        num_non_violations_price_exec += 1
        else:
            num_add += 1
    print("number of violations for price deletion ", num_violations_price_del)
    print("number of violations for price execution ", num_violations_price_exec)
    print("number of violations for size ", num_violations_size)
    print("number of non violations for price deletion ", num_non_violations_price_del)
    print("number of non violations for price execution ", num_non_violations_price_exec)
    print("number of add orders ", num_add)
    print()
    
    
def from_event_exec_to_order(data):
    #transform execution event in add order of the opposite side
    for i in range(data.shape[0]):
        if (data[i, 1] == 3):
            data[i, 1] = 0
            # if the event is an execution of a sell limit order then we transform it in a buy market order
            if (data[i, 2] < 0):
                data[i, 2] = -data[i, 2]
            # if the event is an execution of a sell limit order then we transform it in a buy market order
            elif data[i, 2] > 0:
                data[i, 2] = data[i, 2]
            else:
                raise ValueError("Execution order with size 0")
    return data
'''