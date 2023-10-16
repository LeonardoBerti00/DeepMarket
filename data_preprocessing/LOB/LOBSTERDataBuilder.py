import os
import data_preprocessing.preprocessing_utils as ppu
import pandas as pd

import numpy as np
from config import Configuration
import constants as cst


class LOBSTERDataBuilder:
    def __init__(
        self,
        stock_name,
        data_dir,
        config: Configuration,
        dataset_type,
        n_lob_levels=None,
        normalization_mean=None,
        normalization_std=None,
        crop_trading_day_by=0,
        window_size_forward=None,
        window_size_backward=None,
        num_snapshots=100,
        label_threshold_pos=None,
        label_threshold_neg=None,
        label_dynamic_scaler=None,
        is_data_preload=None,
        start_end_trading_day=None,
    ):
        self.config = config
        self.dataset_type = dataset_type
        self.n_lob_levels = n_lob_levels
        self.is_data_preload = is_data_preload
        self.data_dir = data_dir
        self.start_end_trading_day = start_end_trading_day
        self.crop_trading_day_by = crop_trading_day_by

        self.normalization_means = normalization_mean
        self.normalization_stds = normalization_std

        self.window_size_forward = window_size_forward
        self.window_size_backward = window_size_backward
        self.label_dynamic_scaler = label_dynamic_scaler

        self.num_snapshots = num_snapshots

        self.label_threshold_pos = label_threshold_pos
        self.label_threshold_neg = label_threshold_neg

        # to store the datasets
        self.stock_name = stock_name

        self.dir_name = "{}/{}_{}_{}".format(
            self.data_dir,
            self.stock_name,
            self.start_end_trading_day[0],
            self.start_end_trading_day[1],
        )

        self.__data_un_gathered = None
        self.__data, self.__samples_x, self.__samples_y = None, None, None   # NX40, MX100X40, MX1
        self.__prepare_dataset()  # KEY CALL

    def __prepare_dataset(self):
        COLUMNS_NAMES = {"orderbook": ["sell", "vsell", "buy", "vbuy"],
                         "message": ["time", "event_type", "order_id", "size", "price", "direction"]}

        # iterate over files in the data directory of self.STOCK_NAME
        for i, filename in enumerate(os.listdir(self.dir_name)):
            f = os.path.join(self.dir_name, filename)
            if os.path.isfile(f):

                if (i % 2) == 0:
                    if i == 0:
                        messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                    else:
                        message = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        messages = pd.concat([messages, message])

                else:
                    if i == 1:
                        orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                    else:
                        orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                        orderbooks = pd.concat([orderbooks, orderbook])

            else:
                raise ValueError("File {} is not a file".format(f))

        # use only executed orders::
        # OrderEvent.EXECUTION,
        # OrderEvent.HIDDEN_EXECUTION
        #take the indexes of the messages that are not executed
        indexes = messages[(messages["event_type"].isin([1, 2, 3, 4]))].index

        #drop the orderbooks with index like indexes
        orderbooks = orderbooks.drop(indexes)
        messages = messages.drop(indexes)

        # drop the orderbooks with missing values and save the indexes
        indexes = orderbooks[orderbooks.isnull().any(axis=1)].index
        orderbooks = orderbooks.drop(indexes)
        messages = messages.drop(indexes)
