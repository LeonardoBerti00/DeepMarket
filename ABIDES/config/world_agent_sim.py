import argparse
import time

import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

import constants as cst
from Kernel import Kernel
from agent.WorldAgent import WorldAgent
from util.order import LimitOrder
from util import util
from agent.ExchangeAgent import ExchangeAgent


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
parser.add_argument('-d', '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('--start-time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('--end-time',
                    default='16:00:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# Execution agent config
parser.add_argument('-e',
                    '--execution-agents',
                    action='store_true',
                    help='Flag to allow the execution agent to trade.')
#parser.add_argument('-p',
#                    '--execution-pov',
#                    type=float,
#                    default=0.1,
#                    help='Participation of Volume level for execution agent')

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

exchange_log_orders = False
log_orders = None

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
starting_cash = 1000000000  # Cash in this simulator is always in CENTS.

# 1) Exchange Agent

#  How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=exchange_log_orders,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=0,
                             book_freq=0,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 16, dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1


# 2) World Agent
agents.extend([WorldAgent(id=1,
                            name="WORLD_AGENT",
                            type="WorldAgent",
                            symbol=symbol,
                            date=str(historical_date.date()),
                            date_trading_days=cst.DATE_TRADING_DAYS,
                            diffusion_model=None,
                            data_dir="C:/Users/leona/PycharmProjects/Diffusion-Models-for-Time-Series/data",
                            log_orders=log_orders,
                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 16, dtype='uint64'))
                          )
               ])

agent_types.extend("WorldAgent")
agent_count += 1



########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("World Agent Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 16,
                                                                                                  dtype='uint64')))
kernelStartTime = mkt_open
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

defaultComputationDelay = 0  # 50 nanoseconds
#time.sleep(3)
kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              defaultComputationDelay=defaultComputationDelay,
              log_dir=args.log_dir)


simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))