#!/bin/bash
NUM_JOBS=16
#python printf "Hello World"
python -u abides.py -c world_agent_sim -t TSLA -d 20150130 -s 1234 -l world_agent_sim

cd util/plotting && python -u liquidity_telemetry.py ../../log/world_agent_sim/EXCHANGE_AGENT.bz2 ../../log/world_agent_sim/ORDERBOOK_TSLA_FULL.bz2 \
-o world_agent_sim.png -c configs/plot_09.30_11.30.json && cd ../../