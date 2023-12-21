from pathlib import Path
import wandb
from lightning.pytorch.loggers import WandbLogger
from run import run_wandb, run, sweep_init
import torch
from utils.utils import noise_scheduler
import constants as cst
import configuration
from preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder
from models.NNEngine import NNEngine

def set_torch():
    #torch.manual_seed(cst.SEED)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('high')


if __name__ == "__main__":
    set_torch()
    torch.autograd.set_detect_anomaly(True)
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
            sweep_config.update({"name":
                                     f"model_{config.CHOSEN_MODEL.name}_"
                                     f"stock_{config.CHOSEN_STOCK.name}_"
                                     f"ct_{config.COND_TYPE}_"
                                     f"cm_{config.COND_METHOD}_"
                                     f"aug_{config.IS_AUGMENTATION}_"
                                     f"ndiffstep_{config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS]}"
                                 })
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME)
            wandb.agent(sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"])
        else:
            start_wandb = run_wandb(config, accelerator)
            start_wandb()

    # training without using wandb
    elif config.IS_TRAINING:
        run(config, accelerator)

    elif config.IS_DISCRIMINATIVE:
        pass

    elif config.IS_PREDICTIVE:
        pass

