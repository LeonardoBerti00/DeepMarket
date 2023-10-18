import os
from data_preprocessing.preprocessing_utils import z_score_orderbook, normalize_messages
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
        self.dataframes = []

    def prepare_save_datasets(self):
        path = "{}/{}/{}_{}_{}".format(
            self.data_dir,
            self.stock_name,
            self.stock_name,
            self.date_trading_days[0],
            self.date_trading_days[1],
        )

        self._prepare_dataframes(path)
        self._normalize_dataframes()

        path_where_to_save = "{}/{}".format(
            self.data_dir,
            self.stock_name,
        )

        self.train_set = pd.concat(self.dataframes[0], axis=1).values
        self.val_set = pd.concat(self.dataframes[1], axis=1).values
        self.test_set = pd.concat(self.dataframes[2], axis=1).values

        self._save(path_where_to_save)


    def _normalize_dataframes(self):
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][1], mean_vol, mean_prices, std_vol, std_prices = z_score_orderbook(self.dataframes[i][1])
            else:
                self.dataframes[i][1], _, _, _, _ = z_score_orderbook(self.dataframes[i][1], mean_vol, mean_prices, std_vol, std_prices)

        #do the same thing with messages
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][0], mean_vol, mean_prices, std_vol, std_prices = normalize_messages(self.dataframes[i][0])
            else:
                self.dataframes[i][0], _, _, _, _ = normalize_messages(self.dataframes[i][0], mean_vol, mean_prices, std_vol, std_prices)

        #print mean value and std value for each dataframe and for each column
        '''
        for i in range(len(self.dataframes)):
            for j in range(len(self.dataframes[i])):
                print("Dataframe {}".format(i))
                print("Column {}".format(j))
                # print the mean for each column with the name of dataframe[i][j]
                print("Mean: {}".format(np.mean(self.dataframes[i][j])))
                #print also the name of the column
                print("Std: {}".format(np.std(self.dataframes[i][j])))
                print("")
        '''


    def _save(self, path_where_to_save):
        np.save(path_where_to_save + "/train.npy", self.train_set)
        np.save(path_where_to_save + "/val.npy", self.val_set)
        np.save(path_where_to_save + "/test.npy", self.test_set)


    def _prepare_dataframes(self, path):

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
                         "message": ["time", "event_type", "order_id", "volume", "price", "direction"]}

        self.num_trading_days = len(os.listdir(path))//2
        split_days = self._split_days()
        split_days = [i * 2 for i in split_days]
        self._create_dataframes_splitted(path, split_days, COLUMNS_NAMES)
        self._preprocess_dataframes()


    def _reset_indexes(self):
        # reset the indexes of the messages and orderbooks
        for i in range(len(self.dataframes)):
            self.dataframes[i][0] = self.dataframes[i][0].reset_index(drop=True)
            self.dataframes[i][1] = self.dataframes[i][1].reset_index(drop=True)


    def _preprocess_dataframes(self):

        self._reset_indexes()

        # take only the first n_lob_levels levels of the orderbook and drop the others
        for i in range(len(self.dataframes)):
            self.dataframes[i][1] = self.dataframes[i][1].iloc[:, :self.n_lob_levels * cst.LEN_LEVEL]

        # take the indexes of the dataframes that are of type 5, 6, 7 and drop it
        for i in range(len(self.dataframes)):
            indexes = self.dataframes[i][0][self.dataframes[i][0]["event_type"].isin([5, 6, 7])].index
            self.dataframes[i][0] = self.dataframes[i][0].drop(indexes)
            self.dataframes[i][1] = self.dataframes[i][1].drop(indexes)

        self._drop_missing_values()

        # drop index column in messages
        for i in range(len(self.dataframes)):
            self.dataframes[i][0] = self.dataframes[i][0].drop(columns=["order_id"])

        # do the difference of time row per row in messages and subsitute the values with the differences
        for i in range(len(self.dataframes)):
            self.dataframes[i][0]["time"] = self.dataframes[i][0]["time"].diff()
            self.dataframes[i][0]["time"].iloc[0] = 0

        self._reset_indexes()


    def _split_days(self):
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        return [train, val, test]


    def _create_dataframes_splitted(self, path, split_days, COLUMNS_NAMES):

        # iterate over files in the data directory of self.STOCK_NAME
        for i, filename in enumerate(os.listdir(path)):
            f = os.path.join(path, filename)
            if os.path.isfile(f):

                if i < split_days[0]:  # then we are creating the df for the training set
                    if (i % 2) == 0:
                        if i == 0:
                            train_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            train_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])

                    else:
                        if i == 1:
                            train_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            if (len(train_orderbooks) != len(train_messages)):
                                raise ValueError("train_orderbook length is different than train_messages")
                        else:
                            train_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            train_messages = pd.concat([train_messages, train_message], axis=0)
                            train_orderbooks = pd.concat([train_orderbooks, train_orderbook], axis=0)

                elif split_days[0] <= i < split_days[1]:  # then we are creating the df for the validation set
                    if (i % 2) == 0:
                        if (i == split_days[0]):
                            self.dataframes.append([train_messages, train_orderbooks])
                            val_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            val_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                    else:
                        if i == split_days[0] + 1:
                            val_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            if (len(val_orderbooks) != len(val_messages)):
                                raise ValueError("val_orderbook length is different than val_messages")
                        else:
                            val_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            val_messages = pd.concat([val_messages, val_message], axis=0)
                            val_orderbooks = pd.concat([val_orderbooks, val_orderbook], axis=0)

                else:  # then we are creating the df for the test set

                    if (i % 2) == 0:
                        if (i == split_days[1]):
                            self.dataframes.append([val_messages, val_orderbooks])
                            test_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            test_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])

                    else:
                        if i == split_days[1] + 1:
                            test_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            if (len(test_orderbooks) != len(test_messages)):
                                raise ValueError("test_orderbook length is different than test_messages")
                        else:
                            test_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            test_messages = pd.concat([test_messages, test_message], axis=0)
                            test_orderbooks = pd.concat([test_orderbooks, test_orderbook], axis=0)

            else:
                raise ValueError("File {} is not a file".format(f))

        self.dataframes.append([test_messages, test_orderbooks])


    def _drop_missing_values(self):
        # drop the orderbooks with missing values checking all the columns and save the indexes and drop them
        for i in range(len(self.dataframes)):
            indexes_null_values = self.dataframes[i][1][self.dataframes[i][1].isnull().any(axis=1)].index
            if len(indexes_null_values) > 0:
                print("Dropping {} orderbooks with missing values".format(len(indexes_null_values)))
                self.dataframes[i][0] = self.dataframes[i][0].drop(indexes_null_values)
                self.dataframes[i][1] = self.dataframes[i][1].drop(indexes_null_values)

        # do the same for messages
        for i in range(len(self.dataframes)):
            indexes_null_values = self.dataframes[i][0][self.dataframes[i][0].isnull().any(axis=1)].index
            if len(indexes_null_values) > 0:
                print("Dropping {} orderbooks with missing values".format(len(indexes_null_values)))
                self.dataframes[i][0] = self.dataframes[i][0].drop(indexes_null_values)
                self.dataframes[i][1] = self.dataframes[i][1].drop(indexes_null_values)
