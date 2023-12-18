#!/bin/bash
NUM_JOBS=32
#python printf "Hello World"
python -u abides.py -c world_agent_sim -t TSLA -d 20150130 -s 1234 -pt ema

cd util/plotting
# you may need to change the name of the log directory that now is market_replay_sim_11_30
python -u liquidity_telemetry.py ../../log/{insert_log_dir}/EXCHANGE_AGENT.bz2 ../../log/{insert_log_dir}/ORDERBOOK_TSLA_FULL.bz2 -o ../../log/{insert_log_dir}/world_agent_sim.png -c configs/plot_09.30_11.30.json
cd ../../

#python -u abides.py -c market_replay_sim_11_30 -t TSLA -d 20150130 -s 1234 -l world_agent_sim_pov_0.01 -e -p 0.01
#python -u abides.py -c market_replay_sim_11_30 -t TSLA -d 20150130 -s 1234 -l market_replay_sim_11_30_pov_0.05 -e -p 0.05
python -u abides.py -c world_agent_sim -t TSLA -d 20150130 -s 1234 -e -p 0.1
#python -u abides.py -c market_replay_sim_11_30 -t TSLA -d 20150130 -s 1234 -l world_agent_sim_pov_0.5 -e -p 0.5

cd realism
#you may need to change the json config file
python -u impact_single_day_pov.py plot_configs/plot_configs/single_day/world_agent_sim_single_day.json
cd ..

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