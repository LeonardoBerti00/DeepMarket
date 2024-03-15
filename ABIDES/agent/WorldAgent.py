import os
from copy import deepcopy

import torch
from agent.Agent import Agent
from agent.ExchangeAgent import ExchangeAgent
from agent.TradingAgent import TradingAgent
from torch import nn
from util.order.LimitOrder import LimitOrder
from util.util import log_print
from message.Message import Message

from math import sqrt
import numpy as np
import pandas as pd
import datetime

from ABIDES.util.order.MarketOrder import MarketOrder
from utils.utils_data import reset_indexes, normalize_messages, one_hot_encode_type, to_sparse_representation
import constants as cst

class WorldAgent(Agent):
    # the objective of this world agent is to replicate the market for the first 30mins and then
    # generated new orderr with the help of a diffusion model for the rest of the day,
    # the diffusion model takes in input the last orders or the last snapshot of the order book
    # and generates new orders for the next time step


    def __init__(self, id, name, type, symbol, date, date_trading_days, diffusion_model, data_dir, cond_type,
                 cond_seq_size, log_orders=True, random_state=None, normalization_terms=None, using_diffusion=False):

        super().__init__(id, name, type, random_state=random_state, log_to_file=log_orders)
        self.count_neg_size = 0
        self.next_historical_orders_index = 0
        self.lob_snapshots = []
        self.sparse_lob_snapshots = []
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
        self.count_diff_placed_orders = 0
        self.count_modify = 0
        self.cond_type = cond_type
        self.cond_seq_size = cond_seq_size
        self.first_generation = True
        self.normalization_terms = normalization_terms
        self.ignored_cancel = 0
        self.generated_orders_out_of_depth = 0
        self.generated_cancel_orders_empty_depth = 0
        self.depth_rounding = 0
        self.last_bid_price = 0
        self.last_ask_price = 0
        if using_diffusion:
            self.starting_time_diffusion = '15min'
        else:
            self.starting_time_diffusion = '10000min'

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle
        self.exchangeID = self.kernel.findAgentByType(ExchangeAgent)
        self.mkt_open = startTime

    def kernelTerminating(self):
        # self.kernel is set in Agent.kernelInitializing()
        super().kernelTerminating()
        print("World Agent terminating.")
        print("World Agent ignored {} cancel orders".format(self.ignored_cancel))

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
        #make a print every 10 minutes
        if currentTime.minute % 2 == 0 and currentTime.second % 30 == 0:
            print("Current time: {}".format(currentTime))
            print("Number of generated orders out of depth: {}".format(self.generated_orders_out_of_depth))
            print("Number of generated cancel orders unmatched: {}".format(self.generated_cancel_orders_empty_depth))
            print("Number of negative size: {}".format(self.count_neg_size))
            print("Number of generated placed orders: {}".format(self.count_diff_placed_orders))
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)

        if self.first_wakeup:
            self.state = 'PRE_GENERATING'
            offset = datetime.timedelta(seconds=self.historical_orders[0, 0])
            time_next_wakeup = currentTime + offset
            self.setWakeup(time_next_wakeup)
            self.requestDataSubscription(self.symbol, levels=10)
            self.first_wakeup = False

        # if current time is between 09:30 and 10:00, then we are in the pre-open phase
        elif self.mkt_open < currentTime <= self.mkt_open + pd.Timedelta(self.starting_time_diffusion):
            next_order = self.historical_orders[self.next_historical_orders_index]
            self.last_offset_time = next_order[0]
            self.placeOrder(currentTime, next_order)
            self.next_historical_orders_index += 1
            if self.next_historical_orders_index < len(self.historical_orders):
                offset = datetime.timedelta(seconds=self.historical_orders[self.next_historical_orders_index, 0])
                self.setWakeup(currentTime + offset + datetime.timedelta(microseconds=1))
            else:
                return

        elif currentTime > self.mkt_open + pd.Timedelta(self.starting_time_diffusion):
            self.state = 'GENERATING'

            # we generate the first order then the others will be generated everytime we receive the update of the lob
            if self.first_generation:
                generated = self._generate_order(currentTime)
                self.next_order = generated
                self.first_generation = False
                offset_time = datetime.timedelta(seconds=generated[0])
                self.setWakeup(currentTime + offset_time + datetime.timedelta(microseconds=1))
                return

            # firstly we place the last order generated and next we generate the next order
            if self.next_order is not None and len(self.kernel.messages.queue) == 0:
                self.placeOrder(currentTime, self.next_order)
                self.next_order = self._generate_order(currentTime)
                offset_time = datetime.timedelta(seconds=self.next_order[0])
                self.setWakeup(currentTime + offset_time + datetime.timedelta(microseconds=1))

            elif (len(self.kernel.messages.queue) > 0):
                self.setWakeup(currentTime + datetime.timedelta(microseconds=1))


    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body['msg'] == 'MARKET_DATA':
            self._update_lob_snapshot(msg)
            self._update_active_limit_orders()

        # if we had placed a market order and it is executed we receive the message of the limit order filled, so if it was a buy we receive a sell
        elif msg.body['msg'] == 'ORDER_EXECUTED':
            direction = 1 if msg.body['order'].is_buy_order else -1
            self.placed_orders.append(np.array([self.last_offset_time, 4, msg.body['order'].order_id, msg.body['order'].quantity, msg.body['order'].limit_price, direction]))
            self.logEvent('ORDER_EXECUTED', msg.body['order'].to_dict())

        elif msg.body['msg'] == 'ORDER_ACCEPTED':
            direction = 1 if msg.body['order'].is_buy_order else -1
            self.placed_orders.append(np.array([self.last_offset_time, 1, msg.body['order'].order_id, msg.body['order'].quantity, msg.body['order'].limit_price, direction]))
            self.logEvent('ORDER_ACCEPTED', msg.body['order'].to_dict())

        elif msg.body['msg'] == 'ORDER_CANCELLED':
            direction = 1 if msg.body['order'].is_buy_order else -1
            self.placed_orders.append(np.array([self.last_offset_time, 3, msg.body['order'].order_id, msg.body['order'].quantity, msg.body['order'].limit_price, direction]))
            self.logEvent('ORDER_CANCELLED', msg.body['order'].to_dict())

    def placeOrder(self, currentTime, order):
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
                    self.ignored_cancel += 1
                    return
                    # raise Exception("trying to cancel an order that doesn't exist")
                if type == 3:
                    # total deletion of a limit order
                    self.cancelOrder(old_order)
                elif type == 2:
                    # partial deletion of a limit order
                    new_order = LimitOrder(
                        agent_id=self.id, 
                        time_placed=self.currentTime, 
                        symbol=self.symbol, 
                        quantity=old_order.quantity-quantity, 
                        is_buy_order=old_order.is_buy_order, 
                        limit_price=old_order.limit_price, 
                        order_id=old_order.order_id, 
                        tag=None
                    )
                    self.modifyOrder(old_order, new_order)
                    self.placed_orders.append(np.array([order[0], 2, new_order.order_id, quantity, new_order.limit_price, direction]))

            elif type == 4:
                # if type == 4 it means that it is an execution order, so if it is an execution order of a sell limit order
                # we place a buy market order of the same quantity and viceversa
                is_buy_order = False if direction else True
                # the current order_id is the order_id of the sell (buy) limit order filled, 
                # so we need to assign to the market order another order_id
                order_id = self.unused_order_ids[0]
                self.unused_order_ids = self.unused_order_ids[1:]
                self.placeMarketOrder(self.symbol, quantity, is_buy_order=is_buy_order, order_id=order_id)
        else:
            log_print("Agent ignored order of quantity zero: {}", order)


    def _generate_order(self, currentTime):
        generated = None
        while generated is None:
            if self.cond_type == 'only_lob':
                lob_snapshots = np.array(self.lob_snapshots[-self.cond_seq_size:])
                cond = torch.from_numpy(self._z_score_orderbook(lob_snapshots)).to(cst.DEVICE, torch.float32)
            elif self.cond_type == 'only_event':
                orders = np.array(self.placed_orders[-self.cond_seq_size:])
                cond = self._preprocess_orders_for_diff_cond(orders,
                                                                            np.array(self.lob_snapshots[-self.cond_seq_size - 1:]))
                #cond = one_hot_encode_type(orders_preprocessed).to(cst.DEVICE)
            else:
                raise ValueError("cond_type not recognized")
            cond = cond.unsqueeze(0)
            x = torch.zeros(1, 1, cst.LEN_EVENT, device=cst.DEVICE, dtype=torch.float32)
            generated = self.diffusion_model.sampling(cond, x)
            generated = generated[0, 0, :]
            generated = self._postprocess_generated(generated)
        return generated


    def placeLimitOrder(self, symbol, quantity, is_buy_order, limit_price, order_id=None, ignore_risk=True, tag=None):
        order = LimitOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, limit_price, order_id, tag)
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


    def _postprocess_generated(self, generated):
        ''' we need to go from the output of the diffusion model to an actual order '''
        direction = generated[8]
        if direction < 0:
            direction = -1
        else:
            direction = 1
        
        order_type = torch.argmin(torch.sum(torch.abs(self.diffusion_model.type_embedder.weight.data - generated[1:6]), dim=1)).item()+1
        #print(order_type)
        #print(self.diffusion_model.type_embedder.weight.data)
        #print(generated[1:4])
        if order_type == 3 or order_type == 2:
            order_type += 1
        # order type == 1 -> limit order
        # order type == 3 -> cancel order
        # order type == 4 -> market order

        # we return the size and the time to the original scale
        size = round(generated[7].item() * self.normalization_terms["event"][1] + self.normalization_terms["event"][0], ndigits=0)
        depth = round(generated[-1].item() * self.normalization_terms["event"][7] + self.normalization_terms["event"][6], ndigits=0)
        time = generated[0].item() * self.normalization_terms["event"][5] + self.normalization_terms["event"][4]

        # if the price or the size are negative we return None and we generate another order
        if size < 0:
            self.count_neg_size += 1
            return None
        
        # if the time is negative we approximate to 1 microsecond
        if time <= 0:
            time = 0.0000001

        if order_type == 1:
            order_id = self.unused_order_ids[0]
            self.unused_order_ids = self.unused_order_ids[1:]
            if direction == 1:
                bid_side = self.lob_snapshots[-1][2::4]
                bid_price = bid_side[0]
                if bid_price == 0:
                    bid_price = self.last_bid_price
                else:
                    self.last_bid_price = bid_price
                last_price = bid_side[-1] 
                price = bid_price - depth*100
                # if the first 10 levels are full and the price is less than the last price we generate another order
                # because we consider only the first 10 levels
                if price < last_price and last_price > 0:
                    self.generated_orders_out_of_depth += 1
                    return None
            else:
                ask_side = self.lob_snapshots[-1][0::4]
                ask_price = ask_side[0]
                if ask_price == 0:
                    ask_price = self.last_ask_price
                else:
                    self.last_ask_price = ask_price
                last_price = ask_side[-1]
                price = ask_price + depth*100
                if price > last_price and last_price > 0:
                    self.generated_orders_out_of_depth += 1
                    return None

        elif order_type == 3:
            if direction == 1:
                bid_side = self.lob_snapshots[-1][2::4]
                bid_price = bid_side[0]
                if bid_price == 0:
                    bid_price = self.last_bid_price
                else:
                    self.last_bid_price = bid_price
                price = bid_price - depth*100
                # search all the active limit orders with the same price
                orders_with_same_price = [order for order in self.active_limit_orders.values() if order.limit_price == price]
                # if there are no orders with the same price then we generate another order
                if len(orders_with_same_price) == 0:
                    self.generated_cancel_orders_empty_depth += 1
                    return None
                # we select the order with the quantity closer to the quantity generated
                order_id = min(orders_with_same_price, key=lambda x: abs(x.quantity - size)).order_id

            else:
                ask_side = self.lob_snapshots[-1][0::4]
                ask_price = ask_side[0]
                if ask_price == 0:
                    ask_price = self.last_ask_price
                else:
                    self.last_ask_price = ask_price
                last_price = ask_side[-1]
                price = ask_price + depth*100
                # search all the active limit orders in the same level
                orders_with_same_price = [order for order in self.active_limit_orders.values() if order.limit_price == price]
                # if there are no orders with the same price then we generate another order
                if len(orders_with_same_price) == 0:
                    self.generated_cancel_orders_empty_depth += 1
                    return None
                # we select the order with the quantity near to the quantity generated
                order_id = min(orders_with_same_price, key=lambda x: abs(x.quantity - size)).order_id

        elif order_type == 4:
            if direction == 1:
                price = self.lob_snapshots[-1][0]
            else:
                price = self.lob_snapshots[-1][2]
            order_id = 0
            # the diffusion gives in output market order and not execution of limit order,
            # so we transform market orders in execution orders of the opposite side as the original message files
            direction = -direction
        self.count_diff_placed_orders += 1
        return np.array([time, order_type, order_id, size, price, direction])


    def _update_active_limit_orders(self):
        asks = self.kernel.agents[0].order_books[self.symbol].asks
        bids = self.kernel.agents[0].order_books[self.symbol].bids
        self.active_limit_orders = {}
        for level in asks:
            for order in level:
                self.active_limit_orders[order.order_id] = order
        for level in bids:
            for order in level:
                self.active_limit_orders[order.order_id] = order


    def _z_score_orderbook(self, orderbook):
        orderbook[:, 0::2] = orderbook[:, 0::2] / 100
        orderbook[:, 0::2] = (orderbook[:, 0::2] - self.normalization_terms["lob"][2]) / self.normalization_terms["lob"][3]
        orderbook[:, 1::2] = (orderbook[:, 1::2] - self.normalization_terms["lob"][0]) / self.normalization_terms["lob"][1]
        return orderbook


    def _preprocess_orders_for_diff_cond(self, orders, lob_snapshots):
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
        orders_dataframe = pd.DataFrame(orders, columns=COLUMNS_NAMES["message"])
        lob_dataframe = pd.DataFrame(lob_snapshots, columns=COLUMNS_NAMES["orderbook"])
        dataframes = [[orders_dataframe, lob_dataframe]]

        # add depth column to messages
        for i in range(len(dataframes)):
            dataframes[i][0]["depth"] = 0

        # we compute the depth of the orders with respect to the orderbook
        for i in range(0, len(dataframes)):
            for j in range(0, dataframes[i][0].shape[0]):
                order_price = dataframes[i][0]["price"].iloc[j]
                direction = dataframes[i][0]["direction"].iloc[j]
                type = dataframes[i][0]["event_type"].iloc[j]
                if type == 1:
                    index = j + 1
                else:
                    index = j
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

        # if order type is 4, then we transform the execution of a sell limit order in a buy market order
        for i in range(len(dataframes)):
            dataframes[i][0]["direction"] = dataframes[i][0]["direction"] * dataframes[i][0]["event_type"].apply(
                lambda x: -1 if x == 4 else 1)

        # drop the order_id column
        for i in range(len(dataframes)):
            dataframes[i][0] = dataframes[i][0].drop(columns=["order_id"])

        # divide all the price, both of lob and messages, by 100
        for i in range(len(dataframes)):
            dataframes[i][0]["price"] = dataframes[i][0]["price"] / 100

        # apply z score to orders
        for i in range(len(dataframes)):
            dataframes[i][0], _, _, _, _, _, _, _, _ = normalize_messages(dataframes[i][0],
                                                                    mean_size=self.normalization_terms["event"][0],
                                                                    mean_prices=self.normalization_terms["event"][2],
                                                                    std_size=self.normalization_terms["event"][1],
                                                                    std_prices=self.normalization_terms["event"][3],
                                                                    mean_time=self.normalization_terms["event"][4],
                                                                    std_time=self.normalization_terms["event"][5],
                                                                    mean_depth=self.normalization_terms["event"][6],
                                                                    std_depth=self.normalization_terms["event"][7]
                                                                    )
                                                                    


        return torch.from_numpy(dataframes[0][0].to_numpy()).to(cst.DEVICE, torch.float32)


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
        dataframes = self._preprocess_events_for_market_replay(dataframes)
        events = dataframes[0][0]
        lob = dataframes[0][1]
        # transform to numpy
        lob = lob.to_numpy()
        events = events.to_numpy()
        return events, lob


    def _preprocess_events_for_market_replay(self, dataframes):

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
        self.last_lob_snapshot = last_lob_snapshot
        self.lob_snapshots.append(last_lob_snapshot)
        self.sparse_lob_snapshots.append(to_sparse_representation(last_lob_snapshot, 100))












