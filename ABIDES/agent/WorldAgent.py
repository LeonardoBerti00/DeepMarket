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
from utils.utils_data import preprocess_dataframes, from_event_exec_to_order


class WorldAgent(Agent):
    # the objective of this world agent is to replicate the market for the first 30mins and then
    # generated new orderr with the help of a diffusion model for the rest of the day,
    # the diffusion model takes in input the last orders or the last snapshot of the order book
    # and generates new orders for the next time step


    def __init__(self, id, name, type, symbol, date, date_trading_days, diffusion_model, data_dir,
                 log_orders=False, random_state=None):
        super().__init__(id, name, type, random_state=random_state)
        self.historical_orders_index = 0
        self.lob_snapshots = []
        self.symbol = symbol
        self.date = date
        self.log_orders = log_orders
        self.executed_trades = dict()
        self.state = 'AWAITING_WAKEUP'
        self.diffusion_model = diffusion_model
        self.historical_orders, self.historical_lob = self._load_orders_lob(self.symbol, data_dir, self.date, date_trading_days)
        self.next_order = None
        self.subscription_requested = False
        self.date_trading_days = date_trading_days
        self.first_wakeup = True

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
            for i in range(0, 40, 4):
                self.placeLimitOrder(self.symbol, self.historical_lob[0][i+1], False, int(self.historical_lob[0][i]))
                self.placeLimitOrder(self.symbol, self.historical_lob[0][i+3], True, int(self.historical_lob[0][i+2]))
            offset = datetime.timedelta(seconds=self.historical_orders[0, 0])
            time_next_wakeup = currentTime + offset
            self.setWakeup(time_next_wakeup)
            self.next_order = self.historical_orders[0]

        if self.subscription_requested is False and currentTime > self.mkt_open:
            self.requestDataSubscription(self.symbol, levels=10)
            self.subscription_requested = True

        #if current time is between 09:30 and 10:00, then we are in the pre-open phase
        if currentTime > self.mkt_open and currentTime <= self.mkt_open + pd.Timedelta('30min'):
            self.state = 'PRE_GENERATION'
            self.placeOrder(currentTime, self.historical_orders[self.historical_orders_index])
            self.historical_orders_index += 1
            self.setWakeup(currentTime + self.historical_orders[self.historical_orders_index, 0])

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
        if msg.body['msg'] == 'ORDER_EXECUTED':
            order = msg.body['order']
            self.executed_trades[currentTime] = [order.fill_price, order.quantity]
            self.last_trade[self.symbol] = order.fill_price

        elif msg.body['msg'] == 'MARKET_DATA':
            last_lob_snapshot = []
            min_actual_lob_level = min(len(msg.body['asks']), len(msg.body['bids']))
            #we take the first 10 levels of the lob and update the list of lob snapshots to use for the conditioning of the diffusion model
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
        order = order[0]
        order_id = order['ORDER_ID']
        existing_order = self.orders.get(order_id)
        if not existing_order and order['SIZE'] > 0:
            self.placeLimitOrder(self.symbol, order['SIZE'], order['BUY_SELL_FLAG'] == 'BUY', order['PRICE'],
                                 order_id=order_id)
        elif existing_order and order['SIZE'] == 0:
            self.cancelOrder(existing_order)
        elif existing_order:
            self.modifyOrder(existing_order, LimitOrder(self.id, currentTime, self.symbol, order['SIZE'],
                                                        order['BUY_SELL_FLAG'] == 'BUY', order['PRICE'],
                                                        order_id=order_id))


    def placeLimitOrder(self, symbol, quantity, is_buy_order, limit_price, order_id=None, ignore_risk=True, tag=None):
        order = LimitOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, limit_price, order_id, tag)

        if quantity > 0:

            q = order.quantity if order.is_buy_order else -order.quantity
            # Copy the intended order for logging, so any changes made to it elsewhere
            # don't retroactively alter our "as placed" log of the order.

            self.sendMessage(self.exchangeID, Message({"msg": "LIMIT_ORDER", "sender": self.id,
                                                       "order": order}))

            # Log this activity.
            if self.log_orders: self.logEvent('ORDER_SUBMITTED', order.to_dict())

        else:
            log_print("TradingAgent ignored limit order of quantity zero: {}", order)


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
        dataframes = preprocess_dataframes(dataframes, n_lob_levels=10)
        events = dataframes[0][0]
        lob = dataframes[0][1]
        events["event_type"] = events["event_type"] - 1.0
        # transform to numpy
        lob = lob.to_numpy()
        events = events.to_numpy()
        orders = from_event_exec_to_order(events)
        return orders, lob

