# DeepMarket: Conditioned Diffusion Models for Realistic Market Simulation
DeepMarket is a Python-based open-source framework developed for Limit Order Book (LOB) simulation with Deep Learning.
This is the official repository for the paper Conditioned Diffusion Models for Realistic Market Simulation.


## Introduction 
We present DeepMarket, an open-source Python framework  developed for LOB market simulation with deep learning. DeepMarket offers the following features: (1) pre-processing for high-frequency market data; (2) a training environment implemented with PyTorch Lightning; (3) hyperparameter search facilitated with WANDB; (4) CDT and CGAN implementations and checkpoints to directly generate a market simulation without training; (5) a comprehensive qualitative (via the plots in this paper) and quantitative (via the predictive score) evaluation. 
To perform the simulation with our world agent and historical data, we extend ABIDES, an open-source agent-based interactive Python tool.

# Getting Started 
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisities
This project requires Python and Conda. If you don't have them installed, please do so first. It is possible to do it using only pip but in that case you are on you own.   

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
If your objective is to train a CDT model or implement your model you should follow those steps. If your objective is to simply execute a market simulation skip this section.

## Data 
1. Firstly you need to have some LOBSTER data otherwise would be impossible to train a new model. The format of the data should be the same of LOBSTER: f"{year}-{month}-{day}_34200000_57600000_{type}" and the data should be saved in f"data/{stock_name}/{stock_name}_{year}-{start_month}-{start_day}_{year}-{end_month}-{end_day}". Type can be or message or orderbook.
2. You need to add the new stock to the constants and to the config file.
3. You need to start the preprocessing setting the config.IS_DATA_PREPROCESSED = False

## Implementing and Training a new model 
To train a new model, you need to have some data follow these steps:
 
### Model
1. Implement your model class in the models/ directory. Your model class should inherit from the NNEngine class and should be a Pytorch Lightning engine. 
2. Update the HP_DICT_MODEL dictionary in run.py to include your model and its hyperparameters.
3. Create a file {model_name}_hparam and write the hyperparameters that you want to use for your model. You can also choose the ones for an hp search. Take inspiration form CDT. 
4. Now choose a configuration, modying hte file configuration.py.
5. Now you can run the main.py with:
```sh
python main.py
```

## Training a CDT Model 
To train a CDT model, you need to follow these steps:
1. Set the CHOSEN_MODEL in configuration.py to cst.Models.CDT
2. Choose different parameters simulation if you want in configuraion.py
2. Now you can run the main.py with:
```sh
python main.py
```

# Running a Market Simulation
If your objective is to execute a market simulation you need to run this command:
```sh
python -u ABIDES/abides.py -c world_agent_sim -t TSLA -date 20150130 -d True -m CDT -st '09:30:00' -et '12:00:00' 
```


If you want to run the same IABS configuration:
```sh
python -u ABIDES/abides.py -c rsmc_03 -date 20150130 -st '09:30:00' -et '12:00:00' 
```

Checkpoints will be released upon acceptance.