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

    if (config.IS_WANDB):
        config.wandb_config_setup()
        if (config.IS_SWEEP):
            wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model="all", save_dir=cst.DIR_SAVED_MODEL)
            sweep_config = sweep_init(config)
            sweep_config.update({"name": f"model_{config.CHOSEN_MODEL.name}_stock_{config.CHOSEN_STOCK.name}_cond_type_{config.COND_TYPE}_cond_method_{config.COND_METHOD}_is_augmentation_{config.IS_AUGMENTATION}"})
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME)
            wandb.agent(sweep_id, run_wandb(config, accelerator, wandb_logger), count=sweep_config["run_cap"])
        else:
            wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model="all", save_dir=cst.DIR_SAVED_MODEL)
            run_wandb(config, accelerator, wandb_logger)

    elif(config.IS_TESTING):
        # reference can be retrieved in artifacts panel
        # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
        checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"

        # download checkpoint locally (if not already cached)
        wandb_run = wandb.init(project=cst.PROJECT_NAME)
        artifact = wandb_run.use_artifact(checkpoint_reference, type="model")
        artifact_dir = artifact.download()

        # load checkpoint
        model = NNEngine.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
        run(config, accelerator, model)

    # training without using wandb
    elif(config.IS_TRAINING):
        run(config, accelerator)

