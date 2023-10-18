import os
from data_preprocessing.preprocessing_utils import z_score_orderbook
import pandas as pd

import numpy as np
from config import Configuration
import constants as cst



class LOBSTERDataBuilder:
    def __init__(
        self,
        stock_name,
        data_dir,
        n_lob_levels,
        date_trading_days,
        split_rates,
    ):
        self.n_lob_levels = n_lob_levels
        self.data_dir = data_dir
        self.date_trading_days = date_trading_days
        self.stock_name = stock_name
        self.split_rates = split_rates

    def pre_processing(self):
        path = "{}/{}/{}_{}_{}".format(
            self.data_dir,
            self.stock_name,
            self.stock_name,
            self.date_trading_days[0],
            self.date_trading_days[1],
        )

        dataframes = self._prepare_dataset(path)
        for i in range(len(dataframes)):
            if (i == 0):
                dataframes[i][1], mean_vol, mean_prices, std_vol, std_prices = z_score_orderbook(dataframes[i][1])
            else:
                dataframes[i][1], _, _, _, _ = z_score_orderbook(dataframes[i][1], mean_vol, mean_prices, std_vol, std_prices)

        #self._save(orderbook, messages, path)

    def _save(self, orderbook, messages, path):
        # save as numpy orderbook and messages that are pandas dataframes
        np.save(path + "orderbook.npy", orderbook)
        np.save(path + "messages.npy", messages)

    def _prepare_dataset(self, path):

        COLUMNS_NAMES = {"orderbook": ["sell1", "vsell1", "buy1", "vbuy1",
                                       "sell2", "vsell2", "buy2", "vbuy2",
                                       "sell3", "vsell3", "buy3", "vbuy3",
                                       "sell4", "vsell4", "buy4", "vbuy4",
                                       "sell5", "vsell5", "buy5", "vbuy5",
                                       "sell6", "vsell6", "buy6", "vbuy6",
                                       "sell7", "vsell7", "buy7", "vbuy7",
                                       "sell8", "vsell8", "buy8", "vbuy8",
                                       "sell9", "vsell9", "buy9", "vbuy9",
                                       "sell10", "vsell10", "buy10", "vbuy10"],
                         "message": ["time", "event_type", "order_id", "size", "price", "direction"]}

        self.num_trading_days = len(os.listdir(path))//2
        split_days = self._split_days()
        split_days = [i * 2 for i in split_days]
        dataframes = []

        # iterate over files in the data directory of self.STOCK_NAME
        for i, filename in enumerate(os.listdir(path)):
            f = os.path.join(path, filename)
            if os.path.isfile(f):

                if i < split_days[0]:     #then we are creating the df for the training set
                    if (i % 2) == 0:
                        if i == 0:
                            train_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            train_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])

                    else:
                        if i == 1:
                            orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            if (len(orderbooks) != len(train_messages)):
                                raise ValueError("Orderbook length is different than train_messages")
                        else:
                            orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            train_messages = pd.concat([train_messages, train_message], axis=0)
                            orderbooks = pd.concat([orderbooks, orderbook], axis=0)

                elif split_days[0] <= i < split_days[1]:        #then we are creating the df for the validation set
                    if (i % 2) == 0:
                        if (i == split_days[0]):
                            dataframes.append([train_messages, orderbooks])
                            val_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            val_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                    else:
                        if i == split_days[0]+1:
                            orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            if (len(orderbooks) != len(val_messages)):
                                raise ValueError("Orderbook length is different than val_messages")
                        else:
                            orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            val_messages = pd.concat([val_messages, val_message], axis=0)
                            orderbooks = pd.concat([orderbooks, orderbook], axis=0)

                else:                #then we are creating the df for the test set

                    if (i % 2) == 0:
                        if (i == split_days[1]):
                            dataframes.append([val_messages, orderbooks])
                            test_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            test_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])

                    else:
                        if i == split_days[1]+1:
                            orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            if (len(orderbooks) != len(test_messages)):
                                raise ValueError("Orderbook length is different than test_messages")
                        else:
                            orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            test_messages = pd.concat([test_messages, test_message], axis=0)
                            orderbooks = pd.concat([orderbooks, orderbook], axis=0)

            else:
                raise ValueError("File {} is not a file".format(f))

        dataframes.append([messages, orderbooks])

        # azzerate the indexes of the messages and orderbooks
        for i in range(len(dataframes)):
            dataframes[i][0] = dataframes[i][0].reset_index(drop=True)
            dataframes[i][1] = dataframes[i][1].reset_index(drop=True)

        # take the indexes of the dataframes that are of type 5, 6, 7 and drop it
        for i in range(len(dataframes)):
            indexes = dataframes[i][0][dataframes[i][0]["event_type"].isin([5, 6, 7])].index
            dataframes[i][0] = dataframes[i][0].drop(indexes)
            dataframes[i][1] = dataframes[i][1].drop(indexes)

        # take only the first n_lob_levels levels of the orderbook and drop the others
        for i in range(len(dataframes)):
            dataframes[i][1] = dataframes[i][1].iloc[:, :self.n_lob_levels * 4]

        # drop the orderbooks with missing values checking all the columns and save the indexes and drop them
        for i in range(len(dataframes)):
            indexes_null_values = dataframes[i][1][dataframes[i][1].isnull().any(axis=1)].index
            if len(indexes_null_values) > 0:
                print("Dropping {} orderbooks with missing values".format(len(indexes_null_values)))
                dataframes[i][0] = dataframes[i][0].drop(indexes_null_values)
                dataframes[i][1] = dataframes[i][1].drop(indexes_null_values)

        # do the same for messages
        for i in range(len(dataframes)):
            indexes_null_values = dataframes[i][0][dataframes[i][0].isnull().any(axis=1)].index
            if len(indexes_null_values) > 0:
                print("Dropping {} orderbooks with missing values".format(len(indexes_null_values)))
                dataframes[i][0] = dataframes[i][0].drop(indexes_null_values)
                dataframes[i][1] = dataframes[i][1].drop(indexes_null_values)

        # drop index column in messages
        for i in range(len(dataframes)):
            dataframes[i][0] = dataframes[i][0].drop(columns=["order_id"])

        # do the difference of time row per row in messages and subsitute the values with the differences
        for i in range(len(dataframes)):
            dataframes[i][0]["time"] = dataframes[i][0]["time"].diff()
            dataframes[i][0][0]["time"] = 0

        # azzerate the indexes of the messages and orderbooks
        for i in range(len(dataframes)):
            dataframes[i][0] = dataframes[i][0].reset_index(drop=True)
            dataframes[i][1] = dataframes[i][1].reset_index(drop=True)

        return dataframes

    def _split_days(self):
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        return [train, val, test]