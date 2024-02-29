from pathlib import Path
import wandb
from lightning.pytorch.loggers import WandbLogger
from run import run_wandb, run, sweep_init
import torch
from utils.utils import noise_scheduler
import constants as cst
import configuration
from preprocessing.LOBSTERDataBuilder import LOBSTERDataBuilder
from models.NNEngine import NNEngine
import evaluation.predictive_discriminative.predictive_lstm as predictive_lstm
import evaluation.predictive_discriminative.discriminative_lstm as discriminative_lstm
import evaluation.visualizations.comparison_distribution_order_type as comparison_distribution_order_type
import evaluation.visualizations.comparison_distribution_volume_price as comparison_distribution_volume_price
import evaluation.visualizations.comparison_distribution_market_spread as comparison_distribution_market_spread
import evaluation.visualizations.PCA_plots as PCA_plots
import evaluation.visualizations.comparison_midprice as comparison_midprice
import evaluation.visualizations.comparison_multiple_days_midprice as comparison_multiple_days_midprice
import evaluation.visualizations.TSNE_plots as TSNE_plots
import evaluation.visualizations.comparison_volume_distribution as comparison_volume_distribution

def set_torch():
    #torch.manual_seed(cst.SEED)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('high')

def plot_graphs():
    if config.IS_COMPARISON_DISTRIBUTION_ORDER_TYPE:
        comparison_distribution_order_type.main()

    elif config.IS_COMPARISON_DISTRIBUTION_VOLUME_PRICE:
        comparison_distribution_volume_price.main()

    elif config.IS_COMPARISON_DISTRIBUTION_MARKET_SPREAD:
        comparison_distribution_market_spread.main()

    elif config.IS_PCA:
        PCA_plots.main()

    elif config.IS_TSNE:
        TSNE_plots.main()

    elif config.IS_COMPARISON_MIDPRICE:
        comparison_midprice.main()

    elif config.IS_COMPARISON_MULTIPLE_DAYS_MIDPRICE:
        comparison_multiple_days_midprice.main()

    elif config.IS_COMPARISON_VOLUME_DISTRIBUTION:
        comparison_volume_distribution.main()

def pred_discrim():
    if config.IS_PREDICTIVE:
        predictive_lstm.main()
    elif config.IS_DISCRIMINATIVE:
        discriminative_lstm.main()


if __name__ == "__main__":
    set_torch()
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
        
    if config.IS_WANDB:
        if config.IS_SWEEP:
            sweep_config = sweep_init(config)
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME)
            wandb.agent(sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"])
        else:
            start_wandb = run_wandb(config, accelerator)
            start_wandb()

    # training without using wandb
    elif config.IS_TRAINING:
        run(config, accelerator)

    elif config.PRED_DISC:
        pred_discrim()

    elif config.PLOT_GRAPHS:
        plot_graphs()



