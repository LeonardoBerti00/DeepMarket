import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger
import constants as cst
from config import Configuration
from data_preprocessing.DataModule import DataModule
from data_preprocessing.LOB.LOBDataset import LOBDataset
from data_preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch

def run():
    config = Configuration()

    if (not config.IS_DATA_PREPROCESSED):
        data_builder = LOBSTERDataBuilder(
            stock_name=config.CHOSEN_STOCK.name,
            data_dir=cst.DATA_DIR,
            n_lob_levels=config.N_LOB_LEVELS,
            date_trading_days=config.DATE_TRADING_DAYS,
            split_rates=config.SPLIT_RATES,
        )
        data_builder.prepare_save_datasets()

    if (config.IS_SWEEP):

        wandb_logger = WandbLogger(project="MMLM", log_model=True, save_dir=cst.WANDB_DIR)
        wandb_config = wandb_init()
        checkpoint_callback = wandb.ModelCheckpoint(monitor="val_loss", mode="min")
        with wandb.init(config=wandb_config):
            trainer = L.Trainer(
                accelerator="cpu",
                precision="32",
                max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
                profiler="advanced",
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True), checkpoint_callback],
                num_sanity_val_steps=0,
                logger=wandb_logger,
            )

    else:
        trainer = L.Trainer(
            accelerator="gpu",
            precision="32",
            max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
            profiler="advanced",
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True)],
            num_sanity_val_steps=0,
        )

    train_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/train.npy",
        T = config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW_SIZE],
    )

    val_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/val.npy",
        T = config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW_SIZE],
    )

    test_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/test.npy",
        T = config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW_SIZE],
    )

    data_module = DataModule(train_set, val_set, test_set, batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE], num_workers=16)
    train_dataloader, val_dataloader, test_dataloader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
    model = ...
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
    wandb.finish()


def wandb_init():
    #wandb.login("d29d51017f4231b5149d36ad242526b374c9c60a")
    sweep_config = {
        'method': 'random',
        'metric': {
            'goal': 'minimize',
            'name': 'validation_loss'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 15,
            'eta': 2
        },
        'run_cap': 4
    }

    parameters_dict = {
        'epochs': {
            'value': 2
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'dropout': {
            'values': [0.3, 0.4, 0.5]
        },
        'lr': {
            'distribution': 'uniform',
            'max': 0.01,
            'min': 0.0001,
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'eps': {
            'value': 1e-08
        },
        'weight_decay': {
            'value': 0
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="MMLM")
    wandb.agent(sweep_id, run, count=sweep_config["run_cap"])
    return sweep_config
    '''