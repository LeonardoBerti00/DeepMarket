from pathlib import Path
import wandb
from lightning.pytorch.loggers import WandbLogger
from run import run_wandb, run, sweep_init
import torch
from utils.utils import noise_scheduler
import constants as cst
from config import Configuration
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
    config = Configuration()
    config.BETAS = noise_scheduler(
        num_diffusion_timesteps=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS],
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

    if config.IS_WANDB:
        wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model="all", save_dir=cst.DIR_SAVED_MODEL)
        if config.IS_SWEEP:
            sweep_config = sweep_init(config)
            sweep_config.update({"name": f"model_{config.CHOSEN_MODEL.name}_stock_{config.CHOSEN_STOCK.name}_cond_type_{config.COND_TYPE}_cond_method_{config.COND_METHOD}_is_augmentation_{config.IS_AUGMENTATION}"})
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME)
            wandb.agent(sweep_id, run_wandb(config, accelerator, wandb_logger), count=sweep_config["run_cap"])
        else:
            start_wandb = run_wandb(config, accelerator, wandb_logger)
            start_wandb()

    elif config.IS_TESTING:
        dir = Path(cst.DIR_SAVED_MODEL+"/"+str(config.CHOSEN_MODEL.name))
        best_val_loss = 100000
        for file in dir.iterdir():
            val_loss = float(file.name.split("=")[1].split("_")[0])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_reference = file

        # load checkpoint
        model = NNEngine.load_from_checkpoint(checkpoint_reference, config=config, test_num_steps=0)
        run(config, accelerator, model)

    # training without using wandb
    elif config.IS_TRAINING:
        run(config, accelerator)

