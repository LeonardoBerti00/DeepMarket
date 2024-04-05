import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
import constants as cst
from preprocessing.DataModule import DataModule
from preprocessing.LOBDataset import LOBDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import TQDMProgressBar
from models.NNEngine import NNEngine
from collections import namedtuple
from models.diffusers.CDT.CDT_hparam import HP_CDT, HP_CDT_FIXED
from models.diffusers.CSDI.CSDI_hparam import HP_CSDI, HP_CSDI_FIXED


HP_SEARCH_TYPES = namedtuple('HPSearchTypes', ("sweep", "fixed"))
HP_DICT_MODEL = {
    cst.Models.CDT: HP_SEARCH_TYPES(HP_CDT, HP_CDT_FIXED),
    cst.Models.CSDI: HP_SEARCH_TYPES(HP_CSDI, HP_CSDI_FIXED)
}

def train(config, trainer):
    print_setup(config)
    train_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/train.npy",
        seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE],
        one_hot_encoding_type=config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE],
        x_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
    )

    val_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/val.npy",
        seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE],
        one_hot_encoding_type=config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE],
        x_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
    )

    #print("size of train set: ", train_set.data.size())
    #print("size of val set: ", val_set.data.size())
    #print("size of test set: ", test_set.data.size())

    if config.IS_DEBUG:
        train_set.data = train_set.data[:256]
        val_set.data = val_set.data[:256]
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS] = 5
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_DEPTH] = 1

    data_module = DataModule(
        train_set=train_set,
        val_set=val_set,
        batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
        test_batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.TEST_BATCH_SIZE],
        num_workers=4
    )

    train_dataloader, val_dataloader = data_module.train_dataloader(), data_module.val_dataloader()
    model = NNEngine(config=config).to(cst.DEVICE, torch.float32, non_blocking=True)
    trainer.fit(model, train_dataloader, val_dataloader)


def run(config, accelerator, model=None):
    wandb_instance_name = ""
    model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed
    for param in cst.LearningHyperParameter:
        if param.value in model_params:
            config.HYPER_PARAMETERS[param] = model_params[param.value]
            wandb_instance_name += str(param.value[:2]) + "_" + str(model_params[param.value]) + "_"

    cond_type = config.COND_TYPE
    is_augmentation = config.IS_AUGMENTATION
    stock_name = config.CHOSEN_STOCK.name
    diffsteps = config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS]
    augmenter = config.CHOSEN_AUGMENTER
    aug_dim = config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM]
    config.FILENAME_CKPT = str(stock_name) + "_" +  str(cond_type) + "_" + str(augmenter) + "_" + wandb_instance_name + "aug_" + str(is_augmentation) + "_" + str(aug_dim) + "_diffsteps_" + str(diffsteps)
    wandb_instance_name = config.FILENAME_CKPT

    trainer = L.Trainer(
        accelerator=accelerator,
        precision=cst.PRECISION,
        max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
        callbacks=[
            EarlyStopping(monitor="val_ema_loss", mode="min", patience=1, verbose=True, min_delta=0.005),
            TQDMProgressBar(refresh_rate=1000)
            ],
        num_sanity_val_steps=0,
        detect_anomaly=False,
        profiler="simple",
        check_val_every_n_epoch=2
    )
    train(config, trainer)

def run_wandb(config, accelerator):
    def wandb_sweep_callback():
        wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model=False, save_dir=cst.DIR_SAVED_MODEL)
        run_name = None
        if not config.IS_SWEEP:
            run_name = ""
            model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed
            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    run_name += str(param.value[:3]) + "_" + str(model_params[param.value]) + "_"

        with wandb.init(project=cst.PROJECT_NAME, name=run_name, entity="leonardo-berti07") as wandb_instance:
            config.WANDB_INSTANCE = wandb_instance

            if config.IS_SWEEP:
                model_params = wandb.config
            wandb_instance_name = ""

            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    config.HYPER_PARAMETERS[param] = model_params[param.value]
                    wandb_instance_name += str(param.value[:2]) + "_" + str(model_params[param.value]) + "_"

            aug_dim = config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM]
            if aug_dim == 64:
                config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_NUM_HEADS] = 1
            elif aug_dim == 256:
                config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_NUM_HEADS] = 4
            elif aug_dim == 512:
                config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_NUM_HEADS] = 8

            cond_type = config.COND_TYPE
            is_augmentation = config.IS_AUGMENTATION
            stock_name = config.CHOSEN_STOCK.name
            diffsteps = config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS]
            augmenter = config.CHOSEN_AUGMENTER
            seq_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]
            size_type = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB]
            config.FILENAME_CKPT = str(stock_name) + "_" +  str(cond_type) + "_" + str(augmenter) + "_" + wandb_instance_name + "aug_" + str(is_augmentation) + "_diffsteps_" + str(diffsteps) + "_size_type_" + str(size_type) + "_seq_size_" + str(seq_size)
            wandb_instance_name = config.FILENAME_CKPT
            trainer = L.Trainer(
                accelerator=accelerator,
                precision=cst.PRECISION,
                max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
                callbacks=[
                    EarlyStopping(monitor="val_ema_loss", mode="min", patience=1, verbose=True, min_delta=0.005),
                    TQDMProgressBar(refresh_rate=10000)
                ],
                num_sanity_val_steps=0,
                logger=wandb_logger,
                detect_anomaly=False,
                check_val_every_n_epoch=2,
            )

            # log simulation details in WANDB console
            wandb_instance.log({"model": config.CHOSEN_MODEL.name}, commit=False)
            wandb_instance.log({"stock train": config.CHOSEN_STOCK.name}, commit=False)
            wandb_instance.log({"stock test": config.CHOSEN_STOCK.name}, commit=False)
            wandb_instance.log({"cond type": config.COND_TYPE}, commit=False)
            wandb_instance.log({"cond method": config.COND_METHOD}, commit=False)
            wandb_instance.log({"num diff steps": config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS]}, commit=False)
            wandb_instance.log({"is augmentation": config.IS_AUGMENTATION}, commit=False)
            wandb_instance.log({"seq size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]}, commit=False)
            wandb_instance.log({"augmentation dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM]}, commit=False)
            wandb_instance.log({"cdt depth": config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_DEPTH]}, commit=False)
            wandb_instance.log({"cdt num heads": config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_NUM_HEADS]}, commit=False)
            wandb_instance.log({"learning rate": config.HYPER_PARAMETERS[cst.LearningHyperParameter.LEARNING_RATE]}, commit=False)
            wandb_instance.log({"optimizer": config.HYPER_PARAMETERS[cst.LearningHyperParameter.OPTIMIZER]}, commit=False)
            wandb_instance.log({"batch size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE]}, commit=False)
            wandb_instance.log({"augmenter": config.CHOSEN_AUGMENTER}, commit=False)

            train(config, trainer)

    return wandb_sweep_callback

def sweep_init(config):
    #wandb.login("d29d51017f4231b5149d36ad242526b374c9c60a")
    sweep_config = {
        'method': 'grid',
        'metric': {
            'goal': 'minimize',
            'name': 'val_ema_loss'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            'eta': 1.5
        },
        'run_cap': 100,
        'parameters': {**HP_DICT_MODEL[config.CHOSEN_MODEL].sweep}
    }
    return sweep_config

def print_setup(config):
    print("Chosen model is: ", config.CHOSEN_MODEL.name)
    print("Is augmented: ", config.IS_AUGMENTATION)
    if config.IS_AUGMENTATION:
        print("Augmentation dim: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM])
        print("Augmenter: ", config.CHOSEN_AUGMENTER)
        print("CDT depth: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_DEPTH])
        print("CDT num heads: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_NUM_HEADS])
    print("Conditioning type: ", config.COND_TYPE)
    print("Conditioning method: ", config.COND_METHOD)
    print("Number of diffusion steps: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS])
    print("Sequence size: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE])
    print("Batch size: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE])
    print("Learning rate: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.LEARNING_RATE])
    print("Optimizer: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.OPTIMIZER])
    print("One hot encoding type: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE])
    if not config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE]:
        print("Size order embedding: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB])  