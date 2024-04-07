from pathlib import Path
import random
import numpy as np
import wandb
from lightning.pytorch.loggers import WandbLogger
from run import run_wandb, run, sweep_init
import torch
from utils.utils import noise_scheduler
import constants as cst
import configuration
from preprocessing.LOBSTERDataBuilder import LOBSTERDataBuilder
from models.NNEngine import NNEngine
import evaluation.quantitative_eval.predictive_lstm as predictive_lstm
import evaluation.quantitative_eval.discriminative_lstm as discriminative_lstm
import evaluation.visualizations.comparison_distribution_order_type as comparison_distribution_order_type
import evaluation.visualizations.comparison_distribution_volume_price as comparison_distribution_volume_price
import evaluation.visualizations.comparison_distribution_market_spread as comparison_distribution_market_spread
import evaluation.visualizations.PCA_plots as PCA_plots
import evaluation.visualizations.comparison_midprice as comparison_midprice
import evaluation.visualizations.comparison_multiple_days_midprice as comparison_multiple_days_midprice
import evaluation.visualizations.TSNE_plots as TSNE_plots
import evaluation.visualizations.comparison_volume_distribution as comparison_volume_distribution
import evaluation.visualizations.comparison_distribution_log_interarrival_times as comparison_distribution_log_interarrival_times
import evaluation.visualizations.TSNE_density_plot as TSNE_density_plot
import evaluation.visualizations.PCA_density_plot as PCA_density_plot



def set_torch():
    torch.manual_seed(cst.SEED)
    np.random.seed(cst.SEED)
    random.seed(cst.SEED)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    #print("REMEMBER TO PUT DETECT ANOMALY TO FALSE")
    torch.set_float32_matmul_precision('high')

def plot_graphs(real_data_path, gen_data_path):
    comparison_distribution_order_type.main(real_data_path, gen_data_path)
    comparison_distribution_volume_price.main(real_data_path, gen_data_path)
    comparison_distribution_market_spread.main(real_data_path, gen_data_path, IS_REAL=True)
    comparison_distribution_market_spread.main(real_data_path, gen_data_path, IS_REAL=False)
    PCA_plots.main(real_data_path, gen_data_path)
    comparison_midprice.main(real_data_path, gen_data_path)
    comparison_multiple_days_midprice.main(days_paths=[real_data_path, gen_data_path])
    comparison_volume_distribution.main(real_data_path, gen_data_path, IS_REAL=True)
    comparison_volume_distribution.main(real_data_path, gen_data_path, IS_REAL=False)
    comparison_distribution_log_interarrival_times.main(real_data_path, gen_data_path)
    TSNE_density_plot.main(real_data_path, gen_data_path)
    PCA_density_plot.main(real_data_path, gen_data_path)
    TSNE_plots.main(real_data_path, gen_data_path)
    

def predictive_discriminative_scores(real_data_path, gen_data_path):
    predictive_lstm.main(real_data_path, gen_data_path)
    discriminative_lstm.main(real_data_path, gen_data_path)


if __name__ == "__main__":
    set_torch()
    '''
    print("Device:", cst.DEVICE)
    num_gpus = torch.cuda.device_count()
    print(f'Number of GPUs Available: {num_gpus}')
    for i in range(num_gpus):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    exit()
    '''
    config = configuration.Configuration()
    if (cst.DEVICE == "cpu"):
        accelerator = "cpu"
    else:
        accelerator = "gpu"

    if (not config.IS_DATA_PREPROCESSED):
        data_builder = LOBSTERDataBuilder(
            stock_name=config.CHOSEN_STOCK.name,
            data_dir=cst.DATA_DIR,
            date_trading_days=cst.DATE_TRADING_DAYS,
            split_rates=config.SPLIT_RATES,
        )
        data_builder.prepare_save_datasets()
        exit()
    if config.IS_WANDB:
        if config.IS_SWEEP:
            sweep_config = sweep_init(config)
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME, entity="leonardo-berti07")
            wandb.agent(sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"])
        else:
            start_wandb = run_wandb(config, accelerator)
            start_wandb()

    # training without using wandb
    elif config.IS_TRAINING:
        run(config, accelerator)

    elif config.QUANT_METRICS:
        predictive_discriminative_scores(config.REAL_DATA_PATH, config.GEN_DATA_PATH)

    elif config.PLOT_GRAPHS:
        plot_graphs(config.REAL_DATA_PATH, config.GEN_DATA_PATH)



