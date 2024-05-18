import pandas as pd
import numpy as np
import os

import torch
import pandas
import constants as cst

def z_score_market_features(data, mean_spread=None, mean_returns=None, mean_vol_imb=None, mean_abs_vol=None, std_spread=None, std_returns=None, std_vol_imb=None, std_abs_vol=None):
    data = data.reset_index(drop=True)
    if (mean_spread is None) or (std_spread is None):
        mean_spread = data["spread"].mean()
        std_spread = data["spread"].std()
    
    if (mean_returns is None) or (std_returns is None):
        #concatenates returns_1 and returns_5
        mean_returns = pd.concat([data["returns_1"], data["returns_50"]]).mean()
        std_returns = pd.concat([data["returns_1"], data["returns_50"]]).std()
        
    if (mean_vol_imb is None) or (std_vol_imb is None):
        mean_vol_imb = pd.concat([data["volume_imbalance_1"], data["volume_imbalance_5"]]).mean()
        std_vol_imb = pd.concat([data["volume_imbalance_1"], data["volume_imbalance_5"]]).std()
        
    if (mean_abs_vol is None) or (std_abs_vol is None):
        mean_abs_vol = pd.concat([data["absolute_volume_1"], data["absolute_volume_5"]]).mean()
        std_abs_vol = pd.concat([data["absolute_volume_1"], data["absolute_volume_5"]]).std()
    
    data["spread"] = (data["spread"] - mean_spread) / std_spread
    data["returns_1"] = (data["returns_1"] - mean_returns) / std_returns
    data["returns_50"] = (data["returns_50"] - mean_returns) / std_returns
    data["volume_imbalance_1"] = (data["volume_imbalance_1"] - mean_vol_imb) / std_vol_imb
    data["volume_imbalance_5"] = (data["volume_imbalance_5"] - mean_vol_imb) / std_vol_imb
    data["absolute_volume_1"] = (data["absolute_volume_1"] - mean_abs_vol) / std_abs_vol
    data["absolute_volume_5"] = (data["absolute_volume_5"] - mean_abs_vol) / std_abs_vol
    print()
    print("mean spread ", mean_spread)
    print("std spread ", std_spread)
    print("mean returns ", mean_returns)
    print("std returns ", std_returns)
    print("mean vol imb ", mean_vol_imb)
    print("std vol imb ", std_vol_imb)
    print("mean abs vol ", mean_abs_vol)
    print("std abs vol ", std_abs_vol)
    print(data[:10])
    print()
    return data, mean_spread, mean_returns, mean_vol_imb, mean_abs_vol, std_spread, std_returns, std_vol_imb, std_abs_vol



def normalize_order_cgan(data, mean_size=None, mean_depth=None, mean_cancel_depth=None, mean_size_100=None, std_size=None, std_depth=None, std_cancel_depth=None, std_size_100=None):
    data = data.reset_index(drop=True)
    if (mean_size is None) or (std_size is None):
        mean_size = data["size"].mean()
        std_size = data["size"].std()
    
    if (mean_depth is None) or (std_depth is None):
        mean_depth = data["depth"].mean()
        std_depth = data["depth"].std()
        
    if (mean_cancel_depth is None) or (std_cancel_depth is None):
        mean_cancel_depth = data["cancel_depth"].mean()
        std_cancel_depth = data["cancel_depth"].std()
        
    if (mean_size_100 is None) or (std_size_100 is None):
        mean_size_100 = data["quantity_100"].mean()
        std_size_100 = data["quantity_100"].std()
        
    data["size"] = (data["size"] - mean_size) / std_size
    data["depth"] = (data["depth"] - mean_depth) / std_depth
    data["cancel_depth"] = (data["cancel_depth"] - mean_cancel_depth) / std_cancel_depth
    data["quantity_100"] = (data["quantity_100"] - mean_size_100) / std_size_100
    
    data["event_type"] = data["event_type"]-1.0
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)
    data["event_type"] = data["event_type"]-1.0
    # order_type = -1 -> limit order
    # order_type = 0 -> cancel order
    # order_type = 1 -> market order
    print("mean size order cgan", mean_size)
    print("std size order cgan", std_size)
    print("mean depth order cgan", mean_depth)
    print("std depth order cgan", std_depth)
    print("mean cancel depth order cgan", mean_cancel_depth)
    print("std cancel depth order cgan", std_cancel_depth)
    print("mean size 100 order cgan", mean_size_100)
    print("std size 100 order cgan", std_size_100)
    print(data[:5])
    
    return data, mean_size, mean_depth, mean_cancel_depth, mean_size_100, std_size, std_depth, std_cancel_depth, std_size_100


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


def normalize_messages(data, mean_size=None, mean_prices=None, std_size=None,  std_prices=None, mean_time=None, std_time=None, mean_depth=None, std_depth=None):

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

    if (mean_depth is None) or (std_depth is None):
        mean_depth = data["depth"].mean()
        std_depth = data["depth"].std()

    #apply the z score to the original data
    data["time"] = (data["time"] - mean_time) / std_time
    data["size"] = (data["size"] - mean_size) / std_size
    data["price"] = (data["price"] - mean_prices) / std_prices
    data["depth"] = (data["depth"] - mean_depth) / std_depth

    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    data["event_type"] = data["event_type"]-1.0
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)
    # order_type = 0 -> limit order
    # order_type = 1 -> cancel order
    # order_type = 2 -> market order

    return data, mean_size, mean_prices, std_size,  std_prices, mean_time, std_time, mean_depth, std_depth


def reset_indexes(dataframes):
    # reset the indexes of the messages and orderbooks
    for i in range(len(dataframes)):
        dataframes[i][0] = dataframes[i][0].reset_index(drop=True)
        dataframes[i][1] = dataframes[i][1].reset_index(drop=True)
    return dataframes


def preprocess_data(dataframes, n_lob_levels, chosen_model):
    dataframes = reset_indexes(dataframes)

    # take only the first n_lob_levels levels of the orderbook and drop the others
    for i in range(len(dataframes)):
        dataframes[i][1] = dataframes[i][1].iloc[:, :n_lob_levels * cst.LEN_LEVEL]

    # take the indexes of the dataframes that are of type 
    # 2 (partial deletion), 5 (execution of a hidden limit order), 
    # 6 (cross trade), 7 (trading halt) and drop it
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
                bid_price = bid_side[0]
                depth = (bid_price - order_price) // 100
                if depth < 0:
                    depth = 0
            else:
                ask_side = dataframes[i][1].iloc[index, 0::4]
                ask_price = ask_side[0]
                depth = (order_price - ask_price) // 100
                if depth < 0:
                    depth = 0
            dataframes[i][0].loc[j, "depth"] = depth

    # we eliminate the first row of every dataframe because we can't deduce the depth
    for i in range(len(dataframes)):
        dataframes[i][0] = dataframes[i][0].iloc[1:, :]
        dataframes[i][1] = dataframes[i][1].iloc[1:, :]


    dataframes = reset_indexes(dataframes)
    if chosen_model == cst.Models.CGAN:
        for i in range(len(dataframes)):
            dataframes[i][0]["cancel_depth"] = 0
            dataframes[i][0]["quantity_100"] = 0
            dataframes[i][0]["quantity_type"] = 0
            
        for i in range(len(dataframes)):
            dataframes[i][0]["quantity_100"] = dataframes[i][0]["size"].apply(lambda x: x // 100 if x % 100 == 0 else 0)
            dataframes[i][0]["quantity_type"] = dataframes[i][0]["size"].apply(lambda x: -1 if x % 100 == 0 else 1)
            for j in range(1, dataframes[i][0].shape[0]):
                if dataframes[i][0].loc[j, "event_type"] == 3:
                    dataframes[i][0].loc[j, "cancel_depth"] = dataframes[i][1].iloc[j-1, 0::2].tolist().index(dataframes[i][0].loc[j, "price"]) // 2
        #eliminate columns price and time from messages
        for i in range(len(dataframes)):
            dataframes[i][0] = dataframes[i][0].drop(columns=["price", "time"])
        
        for i in range(len(dataframes)):
            dataframes[i][1] = dataframes[i][1].shift(1).fillna(0)
        
        for i in range(len(dataframes)):
            lob_sizes = dataframes[i][1].iloc[:, 1::2]
            lob_prices = dataframes[i][1].iloc[:, 0::2]
            dataframes[i][1]["volume_imbalance_1"] = lob_sizes.iloc[:, 1] / (lob_sizes.iloc[:, 1] + lob_sizes.iloc[:, 0])
            dataframes[i][1]["volume_imbalance_5"] = (lob_sizes.iloc[:, 1] + lob_sizes.iloc[:, 3] + lob_sizes.iloc[:, 5] + lob_sizes.iloc[:, 7] + lob_sizes.iloc[:, 9]) / (lob_sizes.iloc[:, :10].sum(axis=1))
            dataframes[i][1]["absolute_volume_1"] = lob_sizes.iloc[:, 1] + lob_sizes.iloc[:, 0]
            dataframes[i][1]["absolute_volume_5"] = lob_sizes.iloc[:, :10].sum(axis=1)
            dataframes[i][1]["spread"] = lob_prices.iloc[:, 0] - lob_prices.iloc[:, 1]

        for i in range(len(dataframes)):
            order_sign_imbalance_256 = pd.Series(0, index=dataframes[i][1].index)
            order_sign_imbalance_128 = pd.Series(0, index=dataframes[i][1].index)
            returns_50 = pd.Series(0, index=dataframes[i][1].index)
            returns_1 = pd.Series(0, index=dataframes[i][1].index)
            lob_prices = dataframes[i][1].iloc[:, 0::2]
            mid_prices = (lob_prices.iloc[:, 0] + lob_prices.iloc[:, 1]) / 2
            for j in range(len(dataframes[i][1])-256):
                order_sign_imbalance_256.iloc[j] = dataframes[i][0]["direction"].iloc[j:j+256].sum()
                order_sign_imbalance_128.iloc[j] = dataframes[i][0]["direction"].iloc[j+128:j+256].sum()
                returns_1.iloc[j] = mid_prices[j+255] / mid_prices[j+254] - 1
                returns_50.iloc[j] = mid_prices[j+255] / mid_prices[j+205] - 1
            dataframes[i][1] = dataframes[i][1].iloc[255:]
            dataframes[i][0] = dataframes[i][0].iloc[255:]
            dataframes[i][1]["order_sign_imbalance_256"] = order_sign_imbalance_256[:-255]
            dataframes[i][1]["order_sign_imbalance_128"] = order_sign_imbalance_128[:-255]
            dataframes[i][1]["returns_1"] = returns_1[:-255]
            dataframes[i][1]["returns_50"] = returns_50[:-255]
            dataframes[i][1] = dataframes[i][1][["volume_imbalance_1", "volume_imbalance_5", "absolute_volume_1", "absolute_volume_5", "spread", "order_sign_imbalance_256", "order_sign_imbalance_128", "returns_1", "returns_50"]]
        
        dataframes = reset_indexes(dataframes)
        for i in range(len(dataframes)):
            #transform nan values in 0
            dataframes[i][1] = dataframes[i][1].fillna(0)
            dataframes[i][0] = dataframes[i][0].fillna(0)
    # we transform the execution of a sell limit order in a buy market order and viceversa
    for i in range(len(dataframes)):
        dataframes[i][0]["direction"] = dataframes[i][0]["direction"] * dataframes[i][0]["event_type"].apply(
            lambda x: -1 if x == 4 else 1)

    return dataframes[0][1], dataframes[0][0]


def unnormalize(x, mean, std):
    return x * std + mean


def one_hot_encoding_type(data):
    encoded_data = torch.zeros(data.shape[0], data.shape[1] + 2, dtype=torch.float32)
    encoded_data[:, 0] = data[:, 0]
    # encoding order type
    one_hot_order_type = torch.nn.functional.one_hot((data[:, 1]).to(torch.int64), num_classes=3).to(
        torch.float32)
    encoded_data[:, 1:4] = one_hot_order_type
    encoded_data[:, 4:] = data[:, 2:]
    return encoded_data


def tanh_encoding_type(data):
    data[:, 1] = torch.where(data[:, 1] == 1.0, 2.0, torch.where(data[:, 1] == 2.0, 1.0, data[:, 1]))
    data[:, 1] = data[:, 1] - 1
    return data


def to_sparse_representation(lob, n_levels):
    if not isinstance(lob, np.ndarray):
        lob = np.array(lob)
    sparse_lob = np.zeros(n_levels * 2)
    for j in range(lob.shape[0] // 2):
        if j % 2 == 0:
            ask_price = lob[0]
            current_ask_price = lob[j*2]
            depth = (current_ask_price - ask_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)] = lob[j*2+1]
        else:
            bid_price = lob[2]
            current_bid_price = lob[j*2]
            depth = (bid_price - current_bid_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)+1] = lob[j*2+1]
    return sparse_lob


'''
def to_original_lob(event_and_lob, seq_size):
    lob = event_and_lob[:, cst.LEN_ORDER:]

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