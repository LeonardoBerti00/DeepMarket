import os
from utils.utils_data import z_score_orderbook, normalize_messages, preprocess_data, z_score_market_features, normalize_order_cgan
import pandas as pd
import numpy as np
import constants as cst


class LOBSTERDataBuilder:
    def __init__(
        self,
        stock_name,
        data_dir,
        date_trading_days,
        split_rates,
        chosen_model
    ):
        self.n_lob_levels = cst.N_LOB_LEVELS
        self.data_dir = data_dir
        self.date_trading_days = date_trading_days
        self.stock_name = stock_name
        self.split_rates = split_rates
        self.dataframes = []
        self.chosen_model = chosen_model

    def prepare_save_datasets(self):
        path = "{}/{}/{}_{}_{}".format(
            self.data_dir,
            self.stock_name,
            self.stock_name,
            self.date_trading_days[0],
            self.date_trading_days[1],
        )

        self._prepare_dataframes(path)

        path_where_to_save = "{}/{}".format(
            self.data_dir,
            self.stock_name,
        )

        self.train_set = pd.concat(self.dataframes[0], axis=1).values
        self.val_set = pd.concat(self.dataframes[1], axis=1).values
        self.test_set = pd.concat(self.dataframes[2], axis=1).values

        self._save(path_where_to_save)


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
                         "message": ["time", "event_type", "order_id", "size", "price", "direction"]}
        self.num_trading_days = len(os.listdir(path))//2
        split_days = self._split_days()
        split_days = [i * 2 for i in split_days]
        self._create_dataframes_splitted(path, split_days, COLUMNS_NAMES)
        # to conclude the preprocessing we normalize the dataframes
        if (self.chosen_model == cst.Models.CGAN):
            self._normalize_dataframes_gan()
        else:
            self._normalize_dataframes_TRADES()


    def _create_dataframes_splitted(self, path, split_days, COLUMNS_NAMES):

        # iterate over files in the data directory of self.STOCK_NAME
        for i, filename in enumerate(sorted(os.listdir(path))):
            f = os.path.join(path, filename)
            print(f)
            if os.path.isfile(f):
                # then we create the df for the training set
                if i < split_days[0]:
                    if (i % 2) == 0:
                        if i == 0:
                            train_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            train_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])

                    else:
                        if i == 1:
                            train_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            train_orderbooks, train_messages = preprocess_data([train_messages, train_orderbooks], self.n_lob_levels, self.chosen_model)
                            if (len(train_orderbooks) != len(train_messages)):
                                raise ValueError("train_orderbook length is different than train_messages")
                        else:
                            train_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            train_orderbook, train_message = preprocess_data([train_message, train_orderbook], self.n_lob_levels, self.chosen_model)
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
                            val_orderbooks, val_messages = preprocess_data([val_messages, val_orderbooks], self.n_lob_levels, self.chosen_model)
                            if (len(val_orderbooks) != len(val_messages)):
                                raise ValueError("val_orderbook length is different than val_messages")
                        else:
                            val_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            val_orderbook, val_message = preprocess_data([val_message, val_orderbook], self.n_lob_levels, self.chosen_model)
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
                            test_orderbooks, test_messages = preprocess_data([test_messages, test_orderbooks], self.n_lob_levels, self.chosen_model)

                            if (len(test_orderbooks) != len(test_messages)):
                                raise ValueError("test_orderbook length is different than test_messages")
                        else:
                            test_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            test_orderbook, test_message = preprocess_data([test_message, test_orderbook], self.n_lob_levels, self.chosen_model)
                            test_messages = pd.concat([test_messages, test_message], axis=0)
                            test_orderbooks = pd.concat([test_orderbooks, test_orderbook], axis=0)

            else:
                raise ValueError("File {} is not a file".format(f))

        self.dataframes.append([test_messages, test_orderbooks])


    def _normalize_dataframes_TRADES(self):
        # divide all the price, both of lob and messages, by 100
        for i in range(len(self.dataframes)):
            self.dataframes[i][0]["price"] = self.dataframes[i][0]["price"] / 100
            self.dataframes[i][1].loc[:, ::2] /= 100

        #apply z score to orderbooks
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][1], mean_size, mean_prices, std_size, std_prices = z_score_orderbook(self.dataframes[i][1])
            else:
                self.dataframes[i][1], _, _, _, _ = z_score_orderbook(self.dataframes[i][1], mean_size, mean_prices, std_size, std_prices)

        #apply z-score to size and prices of messages with the statistics of the train set
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][0], mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth = normalize_messages(self.dataframes[i][0])
            else:
                self.dataframes[i][0], _, _, _, _, _, _, _, _ = normalize_messages(self.dataframes[i][0], mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth)

    def _normalize_dataframes_gan(self):
        #apply z score to orderbooks
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][1], mean_spread, mean_returns, mean_vol_imb, mean_abs_vol, std_spread, std_returns, std_vol_imb, std_abs_vol = z_score_market_features(self.dataframes[i][1])
            else:
                self.dataframes[i][1], _, _, _, _, _, _, _, _ = z_score_market_features(self.dataframes[i][1], mean_spread, mean_returns, mean_vol_imb, mean_abs_vol, std_spread, std_returns, std_vol_imb, std_abs_vol)

        #apply z-score to size and prices of messages with the statistics of the train set
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][0], mean_size, mean_depth, mean_cancel_depth, mean_size_100, std_size, std_depth, std_cancel_depth, std_size_100 = normalize_order_cgan(self.dataframes[i][0])
            else:
                self.dataframes[i][0], _, _, _, _, _, _, _, _ = normalize_order_cgan(self.dataframes[i][0], mean_size, mean_depth, mean_cancel_depth, mean_size_100, std_size, std_depth, std_cancel_depth, std_size_100)

    def _save(self, path_where_to_save):
        if self.chosen_model == cst.Models.CGAN:
            np.save(path_where_to_save + "/train_cgan.npy", self.train_set)
            np.save(path_where_to_save + "/val_cgan.npy", self.val_set)
            np.save(path_where_to_save + "/test_cgan.npy", self.test_set)
        else:
            np.save(path_where_to_save + "/train.npy", self.train_set)
            np.save(path_where_to_save + "/val.npy", self.val_set)
            np.save(path_where_to_save + "/test.npy", self.test_set)


    def _split_days(self):
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        print(f"There are {train} days for training, {val - train} days for validation and {test - val} days for testing")
        return [train, val, test]



