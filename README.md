# DeepMarket: Conditioned Diffusion Models for Realistic Market Simulation
DeepMarket is a Python-based open-source framework developed for Limit Order Book (LOB) simulation with Deep Learning.
This is the official repository for the paper ...


## Introduction 
DeepMarket offers the following features: 
1. Pre-processing for high-frequency market data.
2. Training environment implemented with PyTorch Lightning. 
3. Hyperparameter search facilitated with WANDB. 
4. Implementations and checkpoints for TRADES and CGAN to directly generate market simulations without training.
5. comprehensive qualitative (via the plots in the paper) and quantitative (via the predictive score) evaluation. 
To perform the simulation with our world agent and historical data, we extend ABIDES, an open-source agent-based interactive Python tool.

# Getting Started 
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisities
This project requires Python and Conda. If you don't have them installed, please do so first. It is possible to do it using pip, but in that case you are on you own.   

## Installing
To set up the environment for this project, follow these steps:

1. Clone the repository:
```sh
git clone <repository_url>
```
2. Navigate to the project directory
3. Create a new Conda environment using the environment.yml file:
```sh
conda env create -f environment.yml
```
4. Activate the new Conda environment:
```sh
conda activate deepmarket
```

# Training
If your objective is to train a TRADES model or implement your model you should follow those steps. If your objective is to simply execute a market simulation skip this section.

## Data 
1. Firstly you need to have some LOBSTER data otherwise it would be impossible to train a new model. The format of the data should be the same of LOBSTER: f"{year}-{month}-{day}_34200000_57600000_{type}" and the data should be saved in f"data/{stock_name}/{stock_name}_{year}-{start_month}-{start_day}_{year}-{end_month}-{end_day}". Type can be or message or orderbook.
2. You need to add the new stock to the constants and to the config file.
3. You need to start the preprocessing setting, to do so set config.IS_DATA_PREPROCESSED to False and run python main.py

## Implementing and Training a new model 
To train a new model, follow these steps:
1. Implement your model class in the models/ directory. Your model class should inherit from the NNEngine class and should be a Pytorch Lightning engine. 
2. Update the HP_DICT_MODEL dictionary in run.py to include your model and its hyperparameters.
3. Create a file {model_name}_hparam and write the hyperparameters that you want to use for your model. You can also specify hyperparameters for a hyperparameter search. Use the TRADES model as an example.
4. Choose a configuration by modifying the `configuration.py` file.
5. Run the training script:
```sh
python main.py
```
6. A checkpoint will be saved in data/checkpoints/ that later you can use to perform a market simulation

## Training a TRADES Model 
To train a TRADES model, you need to follow these steps:
1. Set the CHOSEN_MODEL in configuration.py to cst.Models.TRADES
2. Optionally, adjust the simulation parameters in `configuration.py`.
2. Now you can run the main.py with:
```sh
python main.py
```

# Generate a Market Simulation with TRADES checkpoint
To execute a market simulation with a TRADES checkpoint, there are two options:
1. If you have LOBSTER data you need to save the data in f"data/{stock_name}/{stock_name}_{year}-{start_month}-{start_day}_{year}-{end_month}-{end_day}". The format of the data should be the same of LOBSTER: f"{year}-{month}-{day}_34200000_57600000_{type}". You can see an example with INTC in the following point. Then you need to simply run the following command, inserting the stock symbol and the date that you want to simulate:
```sh
python -u ABIDES/abides.py -c world_agent_sim -t ${stock_symbol} -date ${date} -d True -m TRADES -st '09:30:00' -et '12:00:00' 
```
2. The second possibility is that you do not have LOBSTER data. In this case you can go to [LOBSTER](https://lobsterdata.com/info/DataSamples.php), download one of the available stock with 10 levels, unzip the dir and places the two XLS file in data/{stock_name}/{stock_name}_2012-06-21_2012-06-21. Finally you can run the following command:
```sh
python -u ABIDES/abides.py -c world_agent_sim -t ${stock_name} -date 2012-06-21 -d True -m TRADES -st '09:30:00' -et '12:00:00' 
```
Since the model was not trained with this data we cannot guarantee good performance. 

If you want to perform a simulation with CGAN you need simply to change the -m option to CGAN.

# Running a Market Simulation with IABS configuration
If you want to run the IABS configuration:
```sh
python -u ABIDES/abides.py -c rsmc_03 -date 20150130 -st '09:30:00' -et '12:00:00' 
```

When the simulation ends a log dir will be saved in ABIDES/log, here you can find the processed orders of the simulation, and all the plots used to in the paper to evaluate the stylized facts. At the end of the simulation also the predictive score will be computed. 
