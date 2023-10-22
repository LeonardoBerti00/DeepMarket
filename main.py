import wandb
from lightning.pytorch.loggers import WandbLogger
from run import run_wandb, run
import torch
from utils.utils import noise_scheduler, wandb_init
import constants as cst
from config import Configuration
from preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder


def set_torch():
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    set_torch()
    config = Configuration()
    config.ALPHAS_DASH, config.BETAS = noise_scheduler(
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

    if (config.IS_SWEEP):
        wandb_logger = WandbLogger(project="MMLM", log_model=True, save_dir=cst.WANDB_DIR)
        wandb_config = wandb_init()
        sweep_id = wandb.sweep(wandb_config, project="MMLM")
        wandb.agent(sweep_id, run_wandb, count=wandb_config["run_cap"])

    else:
        run(config, accelerator)

