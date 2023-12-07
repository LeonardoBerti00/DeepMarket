import os
from copy import deepcopy

from agent.Agent import Agent
from agent.ExchangeAgent import ExchangeAgent
from agent.TradingAgent import TradingAgent
from util.order.LimitOrder import LimitOrder
from util.util import log_print
from message.Message import Message

from math import sqrt
import numpy as np
import pandas as pd
import datetime

from ABIDES.util.order.MarketOrder import MarketOrder
from utils.utils_data import preprocess_dataframes, from_event_exec_to_order, reset_indexes


class WorldAgent(Agent):
    # the objective of this world agent is to replicate the market for the first 30mins and then
    # generated new orderr with the help of a diffusion model for the rest of the day,
    # the diffusion model takes in input the last orders or the last snapshot of the order book
    # and generates new orders for the next time step


    def __init__(self, id, name, type, symbol, date, date_trading_days, diffusion_model, data_dir,
                 log_orders=False, random_state=None):
        super().__init__(id, name, type, random_state=random_state)
        self.next_historical_orders_index = 0
        self.lob_snapshots = []
        self.symbol = symbol
        self.date = date
        self.log_orders = log_orders
        self.executed_trades = dict()
        self.state = 'AWAITING_WAKEUP'
        self.diffusion_model = diffusion_model
        self.historical_orders, self.historical_lob = self._load_orders_lob(self.symbol, data_dir, self.date, date_trading_days)
        self.historical_order_ids = self.historical_orders[2]
        self.unused_order_ids = np.setdiff1d(np.arange(0, 99999999), self.historical_order_ids)
        self.next_order = None
        self.subscription_requested = False
        self.date_trading_days = date_trading_days
        self.first_wakeup = True
        self.active_limit_orders = {}
        self.placed_orders = []

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle
        self.exchangeID = self.kernel.findAgentByType(ExchangeAgent)
        self.mkt_open = startTime

    def requestDataSubscription(self, symbol, levels):
        self.sendMessage(recipientID=self.exchangeID,
                         msg=Message({"msg": "MARKET_DATA_SUBSCRIPTION_REQUEST",
                                      "sender": self.id,
                                      "symbol": symbol,
                                      "levels": levels,
                                      "freq": 0})  # if freq is 0 all the LOB updates will be provided
                         )

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        # at the opening of the market we reconstruct the order book sending 10 limit orders for each side taken from self.lob
        if self.first_wakeup:
            self.state = 'PRE_GENERATING'
            offset = datetime.timedelta(seconds=self.historical_orders[0, 0])
            time_next_wakeup = currentTime + offset
            self.setWakeup(time_next_wakeup)
            self.requestDataSubscription(self.symbol, levels=10)
            self.first_wakeup = False

        #if current time is between 09:30 and 10:00, then we are in the pre-open phase
        if currentTime > self.mkt_open and currentTime <= self.mkt_open + pd.Timedelta('30min'):
            self.placeOrder(currentTime, self.historical_orders[self.next_historical_orders_index])
            self.next_historical_orders_index += 1
            offset = datetime.timedelta(seconds=self.historical_orders[self.next_historical_orders_index, 0])
            self.setWakeup(currentTime + offset)

        elif currentTime > self.mkt_open + pd.Timedelta('30min'):
            self.state = 'GENERATING'
            #firstly we place the last order generated and next we generate the next order
            if self.next_order is not None:
                self.placeOrder(currentTime, self.next_order)
                self.next_order = None
            """ 
            PSEUDOCODE:
            if cond.type == 'snapshot': 
            cond = self.oracle.observeLOB(self.symbol, currentTime)
            ready_cond = transform_cond(cond)
            x = torch.zeros(1, 1, cst.LEN_EVENT)
            generated = self.diffusion_model(ready_cond, x)
            offset_time = generated[0]
            self.setWakeup(currentTime + offset_time)
            self.next_order = generated
            self.placeOrder(currentTime, generated)
            """

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body['msg'] == 'MARKET_DATA':
            self._update_lob_snapshot(msg)
        elif msg.body['msg'] == 'ORDER_EXECUTED' and msg.body['order'].tag != 'market_order' and msg.body['order'].order_id == 10044395.0:
            #check if the quantoty is the same of the order placed
            if msg.body['order'].quantity == self.active_limit_orders[msg.body['order'].order_id].quantity:
                #if the quantity is the same then we can delete the order from the active limit orders
                del self.active_limit_orders[msg.body['order'].order_id]
            else:
                #if the quantity is not the same then we need to modify the order in the active limit orders
                new_quantity = self.active_limit_orders[msg.body['order'].order_id].quantity - msg.body['order'].quantity
                self.active_limit_orders[msg.body['order'].order_id].quantiy = new_quantity





    def _update_lob_snapshot(self, msg):
        last_lob_snapshot = []
        min_actual_lob_level = min(len(msg.body['asks']), len(msg.body['bids']))
        # we take the first 10 levels of the lob and update the list of lob snapshots
        # to use for the conditioning of the diffusion model
        for i in range(0, 10):
            if i < min_actual_lob_level:
                last_lob_snapshot.append(msg.body['asks'][i][0])
                last_lob_snapshot.append(msg.body['asks'][i][1])
                last_lob_snapshot.append(msg.body['bids'][i][0])
                last_lob_snapshot.append(msg.body['bids'][i][1])
            #we need the else in case the actual lob has less than 10 levels
            else:
                if len(msg.body['asks']) > len(msg.body['bids']) and i < len(msg.body['asks']):
                    last_lob_snapshot.append(msg.body['asks'][i][0])
                    last_lob_snapshot.append(msg.body['asks'][i][1])
                    last_lob_snapshot.append(0)
                    last_lob_snapshot.append(0)
                elif len(msg.body['bids']) > len(msg.body['asks']) and i < len(msg.body['bids']):
                    last_lob_snapshot.append(0)
                    last_lob_snapshot.append(0)
                    last_lob_snapshot.append(msg.body['bids'][i][0])
                    last_lob_snapshot.append(msg.body['bids'][i][1])
                else:
                    for _ in range(4): last_lob_snapshot.append(0)
        self.lob_snapshots.append(last_lob_snapshot)

    def placeOrder(self, currentTime, order):
        self.placed_orders.append(order)
        if self.state == "PRE_GENERATING":
            order_id = order[2]
            type = order[1]
            quantity = order[3]
            price = int(order[4])
            direction = order[5]
            if quantity > 0:
                direction = False if direction == -1 else True
                if type == 1:
                    self.placeLimitOrder(self.symbol, quantity, is_buy_order=direction, limit_price=price, order_id=order_id)
                elif type == 2 or type == 3:
                    if order_id in self.active_limit_orders:
                        old_order = self.active_limit_orders[order_id]
                        del self.active_limit_orders[order_id]
                    else:
                        raise Exception("trying to cancel an order that doesn't exist")

                    if type == 3:
                        # total deletion of a limit order
                        self.cancelOrder(old_order)
                    elif type == 2:
                        # partial deletion of a limit order
                        new_order = LimitOrder(self.id, self.currentTime, self.symbol, old_order.quantity-quantity, old_order.is_buy_order, old_order.limit_price, old_order.order_id, None)
                        self.placed_orders.append(new_order)
                        self.active_limit_orders[new_order.order_id] = order
                        self.modifyOrder(old_order, new_order)

                elif type == 4:
                    # if type == 4 it means that it is an execution order, so if it is an execution order of a sell limit order
                    # we place a market order of the same quantity and viceversa
                    is_buy_order = False if direction else True
                    # the curren order_id is the order_id of the sell (buy) limit order filled, so we need to assign to
                    # the market order another order_id
                    order_id = self.unused_order_ids[0]
                    self.unused_order_ids = self.unused_order_ids[1:]
                    self.placeMarketOrder(self.symbol, quantity, is_buy_order=is_buy_order, order_id=order_id)
            else:
                log_print("Agent ignored order of quantity zero: {}", order)

        elif self.state == "GENERATING":
            pass


    def placeLimitOrder(self, symbol, quantity, is_buy_order, limit_price, order_id=None, ignore_risk=True, tag=None):
        order = LimitOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, limit_price, order_id, tag)
        self.active_limit_orders[order.order_id] = order
        self.sendMessage(self.exchangeID, Message({"msg": "LIMIT_ORDER", "sender": self.id, "order": order}))
        # Log this activity.
        if self.log_orders: self.logEvent('ORDER_SUBMITTED', order.to_dict())

    def placeMarketOrder(self, symbol, quantity, is_buy_order, order_id=None, ignore_risk=True, tag=None):
        """
          The market order is created as multiple limit orders crossing the spread walking the book until all the quantities are matched.
        """
        order = MarketOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, order_id)
        self.sendMessage(self.exchangeID, Message({"msg": "MARKET_ORDER", "sender": self.id, "order": order}))
        if self.log_orders: self.logEvent('ORDER_SUBMITTED', order.to_dict())


    def cancelOrder(self, order):
        """Used by any Trading Agent subclass to cancel any order.
        The order must currently appear in the agent's open orders list."""
        if isinstance(order, LimitOrder):
            self.sendMessage(self.exchangeID, Message({"msg": "CANCEL_ORDER", "sender": self.id,
                                                       "order": order}))
            # Log this activity.
            if self.log_orders: self.logEvent('CANCEL_SUBMITTED', order.to_dict())
        else:
            log_print("order {} of type, {} cannot be cancelled", order, type(order))

    def modifyOrder(self, order, newOrder):
        """ Used by any Trading Agent subclass to modify any existing limit order.  The order must currently
            appear in the agent's open orders list.  Some additional tests might be useful here
            to ensure the old and new orders are the same in some way."""
        self.sendMessage(self.exchangeID, Message({"msg": "MODIFY_ORDER", "sender": self.id,
                                                   "order": order, "new_order": newOrder}))

        # Log this activity.
        if self.log_orders: self.logEvent('MODIFY_ORDER', order.to_dict())

    def _load_orders_lob(self, symbol, data_dir, date, date_trading_days):
        path = "{}/{}/{}_{}_{}".format(
            data_dir,
            symbol,
            symbol,
            date_trading_days[0],
            date_trading_days[1],
        )
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
        for i, filename in enumerate(os.listdir(path)):
            f = os.path.join(path, filename)
            filename_splitted = filename.split('_')
            file_date = filename_splitted[1]
            if os.path.isfile(f) and file_date == date:
                if filename_splitted[4] == "message":
                    events = pd.read_csv(f, header=None, names=COLUMNS_NAMES["message"])
                elif filename_splitted[4] == "orderbook":
                    lob = pd.read_csv(f, header=None, names=COLUMNS_NAMES["orderbook"])
                else:
                    raise ValueError("File name not recognized")

        dataframes = [[events, lob]]
        dataframes = self.preprocess_events(dataframes)
        events = dataframes[0][0]
        lob = dataframes[0][1]
        # transform to numpy
        lob = lob.to_numpy()
        events = events.to_numpy()
        return events, lob


    def preprocess_events(self, dataframes):

        for i in range(len(dataframes)):
            indexes = dataframes[i][0][dataframes[i][0]["event_type"].isin([5, 6, 7])].index
            dataframes[i][0] = dataframes[i][0].drop(indexes)
            dataframes[i][1] = dataframes[i][1].drop(indexes)

        # do the difference of time row per row in messages and subsitute the values with the differences
        for i in range(len(dataframes)):
            first = dataframes[i][0]["time"].iloc[0]
            dataframes[i][0]["time"] = dataframes[i][0]["time"].diff()
            dataframes[i][0]["time"].iloc[0] = first - 34200
        dataframes = reset_indexes(dataframes)
        events = dataframes[0][0]
        lob = dataframes[0][1]
        # get the order ids of the rows with order_type=1
        order_ids = events.loc[events['event_type'] == 1, 'order_id']

        # filter out the rows that have order_type != 1 and have an order id that is not in order_ids
        filtered_df = events.loc[((events['event_type'] != 1) & ~(events['order_id'].isin(order_ids)))].index

        dataframes[0][0] = events.drop(filtered_df)
        dataframes[0][1] = lob.drop(filtered_df)
        dataframes = reset_indexes(dataframes)
        return dataframes















