import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint

import constants as cst
from preprocessing.DataModule import DataModule
from preprocessing.LOB.LOBDataset import LOBDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from models.NNEngine import NNEngine
from collections import namedtuple

from models.diffusers.DiT.DiT_hparam import HP_DiT


HP_SEARCH_TYPES = namedtuple('HPSearchTypes', ("sweep", "fixed"))
HP_DICT_MODEL = {
    cst.Models.DiT: HP_SEARCH_TYPES(HP_DiT, HP_DiT)
}


def run(config, accelerator):
    trainer = L.Trainer(
        accelerator=accelerator,
        precision=cst.PRECISION,
        max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
        profiler="advanced",
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True)],
        num_sanity_val_steps=0,
    )
    train(config, trainer)

def run_wandb(config, wandb_config, accelerator, wandb_logger):

    run_name = None
    if not config.IS_SWEEP:
        run_name = config.WANDB_SWEEP_NAME
    with wandb.init(config=wandb_config, project=cst.PROJECT_NAME, name=run_name) as wandb_instance:
        # log simulation details in WANDB console
        wandb_instance.log({"model": config.CHOSEN_MODEL.name})
        wandb_instance.log({"stock_train": config.CHOSEN_STOCK.name})
        wandb_instance.log({"stock_test": config.CHOSEN_STOCK.name})
        wandb_instance.log({"seq_size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]})
        wandb_instance.log({"masked_seq_size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE]})
        wandb_instance.log({"cond_type": config.HYPER_PARAMETERS[cst.LearningHyperParameter.COND_TYPE]})

        config.WANDB_RUN_NAME = wandb_instance.name
        config.WANDB_INSTANCE = wandb_instance

        params_dict = wandb_config
        if not config.IS_SWEEP:
            params_dict = None
            model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed

        for param in cst.LearningHyperParameter:
            if param.value in model_params:
                config.HYPER_PARAMETERS[param] = model_params[param.value]

        config.wandb_config_setup()
        print("model_"+str(config.CHOSEN_MODEL)+"-epoch_{epoch}-val_loss_{val_loss:.2f}-"+str(config.HYPER_PARAMETERS[:]))
        checkpoint_callback = ModelCheckpoint(
            dirpath=cst.DIR_SAVED_MODEL,
            monitor="val_loss",
            mode="max",
            filename="model_"+str(config.CHOSEN_MODEL)+"-epoch_{epoch}-val_loss_{val_loss:.2f}-"+str(config.HYPER_PARAMETERS[:]),
        )

        trainer = L.Trainer(
            accelerator=accelerator,
            precision=cst.PRECISION,
            max_epochs=wandb_config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
            profiler="advanced",
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True),
                checkpoint_callback,
            ],
            num_sanity_val_steps=0,
            logger=wandb_logger,
        )

        train(wandb_config, trainer)
    wandb.finish()


def sweep_init(config):
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
        'run_cap': 20,
        'parameters': {**HP_DICT_MODEL[config.CHOSEN_MODEL].sweep}
    }
    return sweep_config


def train(config, trainer):
    print_setup(config)
    train_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/train.npy",
        seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE],
        cond_type=config.HYPER_PARAMETERS[cst.LearningHyperParameter.COND_TYPE],
        x_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
    )

    val_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/val.npy",
        seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE],
        cond_type=config.HYPER_PARAMETERS[cst.LearningHyperParameter.COND_TYPE],
        x_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
    )

    test_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/test.npy",
        seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE],
        cond_type=config.HYPER_PARAMETERS[cst.LearningHyperParameter.COND_TYPE],
        x_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
    )

    data_module = DataModule(train_set, val_set, test_set, batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE], num_workers=16)

    train_dataloader, val_dataloader, test_dataloader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()

    model = NNEngine(
        config=config,
        val_num_steps=val_set.__len__(),
        test_num_steps=test_set.__len__(),
        val_data=val_set.data,
        test_data=test_set.data,
        trainer=trainer
    ).to(cst.DEVICE, torch.float32)

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)


def print_setup(config):
    print("Is augmented: ", config.IS_AUGMENTATION)
    print("Conditioning type: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.COND_TYPE])
    if config.CHOSEN_MODEL.name == "DiT":
        print("Conditioning DiT type: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.DiT_TYPE])
        print("Masked sequence size: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE])

