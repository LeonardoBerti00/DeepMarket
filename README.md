# DeepMarket: Limit Order Book (LOB) simulation with Deep Learning.
DeepMarket is a Python-based open-source framework developed for Limit Order Book (LOB) simulation with Deep Learning.
This is also the official repository for the paper [TRADES: Generating Realistic Market Simulations with Diffusion Models](https://arxiv.org/abs/2502.07071).

## Introduction 
DeepMarket offers the following features: 
1. Pre-processing for high-frequency market data.
2. Training environment implemented with PyTorch Lightning. 
3. Hyperparameter search facilitated with WANDB. 
4. Implementations and checkpoints for TRADES and CGAN to directly generate market simulations without training.
5. Comprehensive qualitative (via the plots in the paper) and quantitative (via the predictive score) evaluation. 
6. TRADES-LOB: a synthetic LOB dataset in data/TRADES-LOB. 

To perform the simulation with our world agent and historical data, we extend ABIDES, an open-source agent-based interactive Python tool.

## TRADES-LOB: A synthetic LOB dataset 
To foster collaboration and help the research community we release a synthetic LOB dataset: TRADES-LOB. TRADES-LOB comprises simulated TRADES market data for Tesla and Intel, for 29/01 and 30/01. Specifically, the dataset is structured into four CSV files, each containing 50 columns. The initial six columns delineate the order features, followed by 40 columns that represent a snapshot of the LOB across the top 10 levels. The concluding four columns provide key financial metrics: mid-price, spread, order volume imbalance, and Volume-Weighted Average Price (VWAP), which can be useful for downstream financial tasks, such as stock price prediction. In total, the dataset is composed of 265,986 rows and 13,299,300 cells, which is similar in size to the benchmark FI-2010 dataset.

# Getting Started 
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisities
This project requires Python and pip. If you don't have them installed, please do so first.   

## Installing
To set up the environment for this project, follow these steps:

1. Clone the repository:
```sh
git clone https://github.com/LeonardoBerti00/DeepMarket.git
```
2. Navigate to the project directory
3. Create a virtual environment:
```sh
python -m venv env
```
4. Activate the new Pip environment:
```sh
env\Scripts\activate
```
5. Download the necessary packages:
```sh
pip install -r requirements.txt
```

# Market Simulation
If your objective is to execute a market simulation, this is the section for you.

## Generate a Market Simulation with TRADES checkpoint
![TRADES's Architecture](https://github.com/LeonardoBerti00/DeepMarket/blob/main/data/architecture.jpg)
To execute a market simulation with a TRADES checkpoint, there are two options:
1. If you do not have LOBSTER data, you can run the following command:
```sh
python -u ABIDES/abides.py -c world_agent_sim -t INTC -date 2012-06-21 -d True -m TRADES -st '09:30:00' -et '12:00:00' -id 2.317
```
The data that you are going to use is from [LOBSTER](https://lobsterdata.com/info/DataSamples.php). Since the model was not trained with this data, we cannot guarantee good performance. 

2. If you have LOBSTER data, you need to save the data in f"data/{stock_name}/{stock_name}_{year}-{start_month}-{start_day}_{year}-{end_month}-{end_day}". The format of the data should be the same as LOBSTER: f"{year}-{month}-{day}_34200000_57600000_{type}". You can see an example with INTC. Then you need to simply change cst.DATE_TRAING_DAYS setting the start day and end day, run the following command, inserting the stock symbol and the date that you want to simulate:
```sh
python -u ABIDES/abides.py -c world_agent_sim -t ${stock_symbol} -date ${date} -d True -m TRADES -st '09:30:00' -et '12:00:00' 
```

When the simulation ends, a log will be saved in ABIDES/log, where you can find the processed orders of the simulation and all the plots used in the paper to evaluate the stylized facts. At the end of the simulation also the predictive score will also be computed. 
If you want to perform a simulation with CGAN, you simply need to change the -m option to CGAN.
To reproduce the results of the paper, you need exactly the same data, so TSLA or INTC of 29/01/2015 or 30/01/2015.

## Running a Market Simulation with IABS configuration
If you want to run the IABS configuration:
```sh
python -u ABIDES/abides.py -c rsmc_03 -date 20150130 -st '09:30:00' -et '12:00:00' 
```

# Training
If you aim to train a TRADES model or implement your model, you should follow those steps.

## Data 
1. Firstly, you need to have some LOBSTER data, otherwise, it would be impossible to train a new model. The format of the data should be the same as LOBSTER: f"{year}-{month}-{day}_34200000_57600000_{type}" and the data should be saved in f"data/{stock_name}/{stock_name}_{year}-{start_month}-{start_day}_{year}-{end_month}-{end_day}". The type can be "message" or "orderbook".
2. You need to add the new stock to the constants and to the config file.
3. You need to change cst.DATE_TRAINING_DAYS setting the start day and end day
4. You need to start the preprocessing setting. To do so, set config.IS_DATA_PREPROCESSED to False and run python main.py

## Implementing and Training a new model 
To train a new model, follow these steps:
1. Implement your model class in the models/ directory. Your model class should inherit from the NNEngine class and should be a PyTorch Lightning engine. 
2. Update the HP_DICT_MODEL dictionary in run.py to include your model and its hyperparameters.
3. Create a file {model_name}_hparam and write the hyperparameters that you want to use for your model. You can also specify hyperparameters for a hyperparameter search. Use the TRADES model as an example.
4. Choose a configuration by modifying the `configuration.py` file.
5. Run the training script:
```sh
python main.py
```
6. A checkpoint will be saved in data/checkpoints/ that you can later use to perform a market simulation

## Training a TRADES Model 
To train a TRADES model, you need to follow these steps:
1. Set the CHOSEN_MODEL in configuration.py to cst.Models.TRADES
2. Optionally, adjust the simulation parameters in `configuration.py`.
3. Now you can run the main.py with:
```sh
python main.py
```

# Citing
If you use the framework in a research project, please cite:
```sh
@misc{berti2025tradesgeneratingrealisticmarket,
      title={TRADES: Generating Realistic Market Simulations with Diffusion Models}, 
      author={Leonardo Berti and Bardh Prenkaj and Paola Velardi},
      year={2025},
      eprint={2502.07071},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2502.07071}, 
}
```

