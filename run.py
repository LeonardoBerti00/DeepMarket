import lightning as L
import torch
import wandb
import constants as cst
from preprocessing.DataModule import DataModule
from preprocessing.LOB.LOBDataset import LOBDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from models.NNEngine import NNEngine

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

def run_wandb(wandb_config, accelerator, wandb_logger):
    checkpoint_callback = wandb.ModelCheckpoint(monitor="val_loss", mode="min")
    with wandb.init(config=wandb_config):
        trainer = L.Trainer(
            accelerator=accelerator,
            precision=cst.PRECISION,
            max_epochs=wandb_config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
            profiler="advanced",
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True), checkpoint_callback],
            num_sanity_val_steps=0,
            logger=wandb_logger,
        )
        train(wandb_config, trainer)
    wandb.finish()


def train(config, trainer):
    train_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/train.npy",
        L=config.HYPER_PARAMETERS[cst.LearningHyperParameter.WINDOW_SIZE],
    )

    val_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/val.npy",
        L=config.HYPER_PARAMETERS[cst.LearningHyperParameter.WINDOW_SIZE],
    )

    test_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/test.npy",
        L=config.HYPER_PARAMETERS[cst.LearningHyperParameter.WINDOW_SIZE],
    )

    data_module = DataModule(train_set, val_set, test_set, batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE], num_workers=16)

    train_dataloader, val_dataloader, test_dataloader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()

    model = NNEngine(
        config=config,
    ).to(cst.DEVICE, torch.float32)

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)




