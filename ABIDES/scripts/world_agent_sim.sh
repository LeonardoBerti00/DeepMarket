#!/bin/bash
NUM_JOBS=32
# market replay simulation
python -u abides/abides.py -c world_agent_sim -t TSLA -d 20150130 -s 1234 -pt ema
# market replay with pov agent
python -u abides/abides.py -c world_agent_sim -t TSLA -d 20150130 -s 1234 -p 0.1 -e True
# world agent with diffusion simulation
python -u abides/abides.py -c world_agent_sim -t TSLA -d 20150130 -s 1234 -pt ema -d True -m CDT
# world agent with diffusion with POV agent
python -u abides/abides.py -c world_agent_sim -t TSLA -d 20150130 -s 1234 -p 0.1 -d True -m CDT -e True
LOG_NAME=market_replay_TSLA_2015-01-30_12-00-00
CONFIG_NAME=plot_09.30_11.30.json
# you may need to change the name of the log directory that now is market_replay_sim_11_30
python -u abides/util/plotting/liquidity_telemetry.py ../../log/${LOG_NAME}/EXCHANGE_AGENT.bz2 ../../log/${LOG_NAME}/ORDERBOOK_TSLA_FULL.bz2 -o ../../log/${LOG_NAME}/world_agent_sim.png -c configs/${CONFIG_NAME}

# you may need to change the json config file with the correct log directory
python -u abides/realism/impact_single_day_pov.py plot_configs/plot_configs/single_day/world_agent_sim_single_day.json


# Multiple seeds for execution experiment
#for seed in $(seq 100 120); do
#  sem -j${NUM_JOBS} --line-buffer python -u abides.py -c rmsc03 -t ABM -d 20200605 -s ${seed} -l rmsc03_demo_no_${seed}_20200605
#  for pov in  0.01 0.05 0.1 0.5; do
#      sem -j${NUM_JOBS} --line-buffer python -u abides.py -c rmsc03 -t ABM -d 20200605 -s ${seed} -l rmsc03_demo_yes_${seed}_pov_${pov}_20200605 -e -p ${pov}
#  done
#done
#sem --wait

# Plot multiple seed experiment
#cd realism && python -u impact_multiday_pov.py plot_configs/plot_configs/multiday/rmsc03_demo_multiday.json -n ${NUM_JOBS} && cd ..