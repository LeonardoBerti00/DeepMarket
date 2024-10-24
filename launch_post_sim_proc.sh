#!/bin/bash -l
#SBATCH -s
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J LSTM 
#SBATCH -p cuda
#SBATCH -c 8
#SBATCH --gres=gpu:fast
#SBATCH --output=/dev/null

cd ABIDES/util/plotting
LOG_NAME=world_agent_TSLA_2015-01-30_11-00-00_val_ema=0.832_epoch=9_TSLA_only_event_MLP_seq_size_64_augment_dim_64_TRADES_depth_4_aug_True_diffsteps_100_size_type_3
CONFIG_NAME=plot_09.30_11.30.json
srun python -u liquidity_telemetry.py ../../log/${LOG_NAME}/EXCHANGE_AGENT.bz2 ../../log/${LOG_NAME}/ORDERBOOK_TSLA_FULL.bz2 -o ../../log/${LOG_NAME}/world_agent_sim.png -c configs/${CONFIG_NAME} -stream ../../log/${LOG_NAME} > ../../../output/std/output_${SLURM_JOB_ID}.txt 2> ../../../output/err/err${SLURM_JOB_ID}.txt


