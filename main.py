import wandb
from lightning.pytorch.loggers import WandbLogger
from run import run_wandb, run, sweep_init
import torch
from utils.utils import noise_scheduler
import constants as cst
from config import Configuration
from preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder


def set_torch():
    #torch.manual_seed(cst.SEED)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')



if __name__ == "__main__":
    set_torch()
    config = Configuration()
    config.ALPHAS_CUMPROD, config.BETAS = noise_scheduler(
        diffusion_steps=config.HYPER_PARAMETERS[cst.LearningHyperParameter.DIFFUSION_STEPS],
        s=config.HYPER_PARAMETERS[cst.LearningHyperParameter.S]
    )
    if (cst.DEVICE == "cpu"):
        accelerator = "cpu"
    else:
        accelerator = "gpu"

    if (not config.IS_DATA_PREPROCESSED):
        data_builder = LOBSTERDataBuilder(
            stock_name=config.CHOSEN_STOCK.name,
            data_dir=cst.DATA_DIR,
            date_trading_days=config.DATE_TRADING_DAYS,
            split_rates=config.SPLIT_RATES,
        )
        data_builder.prepare_save_datasets()

    if (config.IS_WANDB):
        config.wandb_config_setup()
        if (config.IS_SWEEP):
            wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model=True, save_dir=cst.WANDB_DIR)
            sweep_config = sweep_init(config)
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME)
            wandb.agent(sweep_id, run_wandb(config, sweep_config, accelerator, wandb_logger), count=sweep_config["run_cap"])
        else:
            wandb_logger = WandbLogger(project="MMLM", log_model=True, save_dir=cst.WANDB_DIR)
            run_wandb(config, None, accelerator, wandb_logger)

    elif(config.IS_TESTING):
        #TODO load pre trained model and test
        pass

    # training without using wandb
    elif(config.IS_TRAINING):
        run(config, accelerator)

