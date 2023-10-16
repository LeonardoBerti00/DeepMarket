import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger
import constants as cst
from config import Configuration
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch

def run():
    config = Configuration()
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
            accelerator="cpu",
            precision="32",
            max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
            profiler="advanced",
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True)],
            num_sanity_val_steps=0,
        )

    train_set = MultiSourceDataset(
        sr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_RATE],
        channels=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CHANNEL_SIZE],
        min_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MIN_DURATION],
        max_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MAX_DURATION],
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_LENGTH],
        audio_files_dir=cst.AUDIO_FILES_DIR_TRAIN,
        stems=cst.STEMS,
    )
    val_set = MultiSourceDataset(
        sr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_RATE],
        channels=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CHANNEL_SIZE],
        min_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MIN_DURATION],
        max_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MAX_DURATION],
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_LENGTH],
        audio_files_dir=cst.AUDIO_FILES_DIR_VAL,
        stems=cst.STEMS,
    )
    test_set = MultiSourceDataset(
        sr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_RATE],
        channels=config.HYPER_PARAMETERS[cst.LearningHyperParameter.CHANNEL_SIZE],
        min_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MIN_DURATION],
        max_duration=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MAX_DURATION],
        aug_shift=config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUG_SHIFT],
        sample_length=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SAMPLE_LENGTH],
        audio_files_dir=cst.AUDIO_FILES_DIR_TEST,
        stems=cst.STEMS,
    )
    data_module = Datamodule(train_set, val_set, test_set, batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE], num_workers=16)
    train_dataloader, val_dataloader, test_dataloader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
    #model = VAE_MLP(config).to(cst.DEVICE)
    #model = VAE_CNN(config).to(cst.DEVICE)
    model = VQVAE_CNN(config).to(cst.DEVICE)
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