import argparse
from datetime import datetime
import time

import numpy as np
import pandas as pd
import sys
import datetime as dt

import torch
from dateutil.parser import parse
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

import constants as cst
from Kernel import Kernel
from agent.WorldAgent import WorldAgent
from util.order import LimitOrder
from util import util
from agent.ExchangeAgent import ExchangeAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent
from pathlib import Path

import configuration
from models.NNEngine import NNEngine

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for RMSC03 config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-date',
                    '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('-st',
                    '--start-time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('-et',
                    '--end-time',
                    default='11:00:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# Execution agent config
parser.add_argument('-e',
                    '--execution-agents',
                    type=bool,
                    default=False,
                    help='Flag to allow the execution agent to trade.')
parser.add_argument('-m',
                    '--chosen-model',
                    type=str,
                    default='CDT')
parser.add_argument('-p',
                    '--execution-pov',
                    type=float,
                    default=0.1,
                    help='Participation of Volume level for execution agent')
parser.add_argument('-d',
                    '--diffusion',
                    type=bool,
                    default=False,
                    help='Using diffusion')
#add a parser argument that takes in nput a float value for the proportion of volume
# that the agent will trade
parser.add_argument('-id',
                    '--id',
                    type=float,
                    default=None,
                    help='diffusion-id-which-is-best-val-loss')

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

seed = cst.SEED  # Random seed specification on the command line.

exchange_log_orders = True
log_orders = True

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
if symbol == "TSLA":
    normalization_terms = {
        "lob": [
            cst.TSLA_LOB_MEAN_SIZE_10,
            cst.TSLA_LOB_STD_SIZE_10,
            cst.TSLA_LOB_MEAN_PRICE_10,
            cst.TSLA_LOB_STD_PRICE_10,
        ],
        "event": [
            cst.TSLA_EVENT_MEAN_SIZE,
            cst.TSLA_EVENT_STD_SIZE,
            cst.TSLA_EVENT_MEAN_PRICE,
            cst.TSLA_EVENT_STD_PRICE,
            cst.TSLA_EVENT_MEAN_TIME,
            cst.TSLA_EVENT_STD_TIME,
            cst.TSLA_EVENT_MEAN_DEPTH,
            cst.TSLA_EVENT_STD_DEPTH,
        ]
    }

elif symbol == "INTC":
    normalization_terms = {
        "lob": [
            cst.INTC_LOB_MEAN_SIZE_10, 
            cst.INTC_LOB_STD_SIZE_10, 
            cst.INTC_LOB_MEAN_PRICE_10,
            cst.INTC_LOB_STD_PRICE_10
            ],
        "event": [
            cst.INTC_EVENT_MEAN_SIZE, 
            cst.INTC_EVENT_STD_SIZE, 
            cst.INTC_EVENT_MEAN_PRICE, 
            cst.INTC_EVENT_STD_PRICE, 
            cst.INTC_EVENT_MEAN_TIME, 
            cst.INTC_EVENT_STD_TIME,
            cst.INTC_EVENT_MEAN_DEPTH,
            cst.INTC_EVENT_STD_DEPTH
        ]
    }

starting_cash = 100000000000  # Cash in this simulator is always in CENTS.

# 1) Exchange Agent

#  How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
stream_history_length = 2500000

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=exchange_log_orders,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=stream_history_length,
                             book_freq=0,
                             wide_book=True,
                             random_state=np.random.RandomState(
                                 seed=seed))
               ])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) World Agent


if args.diffusion:
    chosen_model = args.chosen_model
    dir_path = Path(cst.DIR_SAVED_MODEL + "/" + str(chosen_model))
    best_val_loss = 1000000
    if args.id is None:
        for file in dir_path.iterdir():
            try:
                val_loss = float(file.name.split("=")[1].split("_")[0])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_reference = file
            except:
                continue
    else:
        for file in dir_path.iterdir():
            try:
                val_loss = float(file.name.split("=")[1].split("_")[0])
                if val_loss == args.id:
                    checkpoint_reference = file
            except:
                continue
    print("checkpoint used: ", checkpoint_reference)
    checkpoint = torch.load(checkpoint_reference, map_location=cst.DEVICE)
    config = checkpoint["hyper_parameters"]["config"]
    config.IS_WANDB = False
    # load checkpoint
    model = NNEngine.load_from_checkpoint(checkpoint_reference, config=config, map_location=cst.DEVICE)
    # we freeze the model
    for param in model.parameters():
        param.requires_grad = False
else:
    model = None


agents.extend([WorldAgent(id=1,
                          name="WORLD_AGENT",
                          type="WorldAgent",
                          symbol=symbol,
                          date=str(historical_date.date()),
                          date_trading_days=cst.DATE_TRADING_DAYS,
                          diffusion_model=model,
                          data_dir=cst.DATA_DIR,
                          cond_type=config.COND_TYPE if args.diffusion else None,
                          cond_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE] - config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE] if args.diffusion else None,
                          size_type_emb=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB] if args.diffusion else None,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(
                              seed=cst.SEED),
                          normalization_terms=normalization_terms,
                          using_diffusion=args.diffusion
                          )
               ])
agent_types.extend("WorldAgent")
agent_count += 1

# 3) Execution Agent
trade_pov = True if args.execution_agents else False

#### Participation of Volume Agent parameters
# POV agent start one hour after market open and ends 30 minutes after 
pov_agent_start_time = mkt_open + pd.to_timedelta('0:15:00')
pov_agent_end_time = mkt_open + pd.to_timedelta('01:00:00')
pov_proportion_of_volume = args.execution_pov
pov_quantity = 1e5
pov_frequency = '1min'
pov_direction = "BUY"

pov_agent = POVExecutionAgent(id=agent_count,
                              name='POV_EXECUTION_AGENT',
                              type='ExecutionAgent',
                              symbol=symbol,
                              starting_cash=starting_cash,
                              start_time=pov_agent_start_time,
                              end_time=pov_agent_end_time,
                              freq=pov_frequency,
                              lookback_period=pov_frequency,
                              pov=pov_proportion_of_volume,
                              direction=pov_direction,
                              quantity=pov_quantity,
                              trade=trade_pov,
                              log_orders=True,  # needed for plots so conflicts with others
                              random_state=np.random.RandomState(seed=seed))
if trade_pov:
    execution_agents = [pov_agent]
    agents.extend(execution_agents)
    agent_types.extend("ExecutionAgent")
    agent_count += 1

########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("World Agent Kernel", random_state=np.random.RandomState(seed=seed))
kernelStartTime = mkt_open
kernelStopTime = mkt_close + pd.to_timedelta('00:00:01')

# parse the string into a datetime object
tmp = datetime.strptime(str(mkt_close), "%Y-%m-%d %H:%M:%S")

# extract the date and time components
date = tmp.date()
time_mkt_close = str(tmp.time()).replace(':', '-')

if trade_pov:
    if args.diffusion:
        log_dir = "world_agent_{}_{}_{}_pov_{}_{}_".format(symbol, date, time_mkt_close, pov_proportion_of_volume, cst.SEED) + checkpoint_reference.name[:-5] 
    else:
        log_dir = "market_replay_{}_{}_{}_pov_{}_{}".format(symbol, date, time_mkt_close, pov_proportion_of_volume, cst.SEED)
else:
    if args.diffusion:
        log_dir = "world_agent_{}_{}_{}_{}_".format(symbol, date, time_mkt_close, cst.SEED) + checkpoint_reference.name[:-5]
    else:
        log_dir = "market_replay_{}_{}_{}_{}".format(symbol, date, time_mkt_close, cst.SEED)

defaultComputationDelay = 0  # 50 nanoseconds
# time.sleep(3)
kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              defaultComputationDelay=defaultComputationDelay,
              log_dir=log_dir)

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
