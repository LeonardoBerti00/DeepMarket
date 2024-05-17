import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from configuration import Configuration
import constants as cst
from models.gans.CGAN_hparam import HP_CGAN, HP_CGAN_FIXED
from models.gans.GANEngine import GANEngine
from preprocessing.DataModule import DataModule
from preprocessing.LOBDataset import LOBDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import TQDMProgressBar
from models.NNEngine import NNEngine
from collections import namedtuple
from models.diffusers.CDT.CDT_hparam import HP_CDT, HP_CDT_FIXED
from models.diffusers.CSDI.CSDI_hparam import HP_CSDI, HP_CSDI_FIXED
from models.gans.CGAN_hparam import HP_CGAN, HP_CGAN_FIXED

HP_SEARCH_TYPES = namedtuple('HPSearchTypes', ("sweep", "fixed"))
HP_DICT_MODEL = {
    cst.Models.CDT: HP_SEARCH_TYPES(HP_CDT, HP_CDT_FIXED),
    cst.Models.CSDI: HP_SEARCH_TYPES(HP_CSDI, HP_CSDI_FIXED),
    cst.Models.CGAN: HP_SEARCH_TYPES(HP_CGAN, HP_CGAN_FIXED)
}

def train(config: Configuration, trainer: L.Trainer):
    print_setup(config)
    train_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/train.npy",
        seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE],
        one_hot_encoding_type=config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE],
        x_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
        chosen_model=config.CHOSEN_MODEL,
        chosen_stock=config.CHOSEN_STOCK,
    )

    val_set = LOBDataset(
        path=cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/val.npy",
        seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE],
        one_hot_encoding_type=config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE],
        x_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
        chosen_model=config.CHOSEN_MODEL,
        chosen_stock=config.CHOSEN_STOCK,
    )
    #print("size of train set: ", train_set.data.size())
    #print("size of val set: ", val_set.data.size())
    #print("size of test set: ", test_set.data.size())
    if config.IS_DEBUG:
        train_set.data = train_set.data[:256]
        val_set.data = val_set.data[:256]
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_DEPTH] = 1
        
    val_set.data = val_set.data[:51200]
    data_module = DataModule(
        train_set=train_set,
        val_set=val_set,
        batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
        test_batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.TEST_BATCH_SIZE],
        num_workers=4
    )
    
    if config.USE_ENGINE == cst.Engine.GAN_ENGINE:
        model = GANEngine(config=config).to(cst.DEVICE, torch.float32, non_blocking=True)
    elif config.USE_ENGINE == cst.Engine.NN_ENGINE:
        model = NNEngine(config=config).to(cst.DEVICE, torch.float32, non_blocking=True)
    else:
        raise ValueError("Specify a valid Engine")

    train_dataloader, val_dataloader = data_module.train_dataloader(), data_module.val_dataloader()
    trainer.fit(model, train_dataloader, val_dataloader)


def run(config: Configuration, accelerator, model=None):
    print("BELLA")
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
    if is_augmentation:
        config.FILENAME_CKPT = str(stock_name) + "_" +  str(cond_type) + "_" + str(augmenter) + "_" + str(aug_dim) + "_" + wandb_instance_name + "_diffsteps_" + str(diffsteps)
    else:
        config.FILENAME_CKPT = str(stock_name) + "_" +  str(cond_type) + "_" + wandb_instance_name + "_diffsteps_" + str(diffsteps)
    wandb_instance_name = config.FILENAME_CKPT

    trainer = L.Trainer(
        accelerator=accelerator,
        precision=cst.PRECISION,
        max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
        callbacks=[
            EarlyStopping(monitor="val_ema_loss", mode="min", patience=2, verbose=True, min_delta=0.005),
            TQDMProgressBar(refresh_rate=1000)
            ],
        num_sanity_val_steps=0,
        detect_anomaly=False,
        profiler="simple",
        check_val_every_n_epoch=1
    )
    train(config, trainer)

def run_wandb(config: Configuration, accelerator):
    def wandb_sweep_callback():
        wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model=False, save_dir=cst.DIR_SAVED_MODEL)
        run_name = None
        if not config.IS_SWEEP:
            model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed
            run_name = ""
            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    run_name += str(param.value[:3]) + "_" + str(model_params[param.value]) + "_"

        run = wandb.init(project=cst.PROJECT_NAME, name=run_name, entity="leonardo-berti07")
        if config.IS_SWEEP:
            model_params = run.config
                       
        wandb_instance_name = ""
        for param in cst.LearningHyperParameter:
            if param.value in model_params:
                config.HYPER_PARAMETERS[param] = model_params[param.value]
                wandb_instance_name += str(param.value) + "_" + str(model_params[param.value]) + "_"
        
        run.name = wandb_instance_name
        aug_dim = config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM]
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_NUM_HEADS] = aug_dim // 64
        cond_type = config.COND_TYPE
        stock_name = config.CHOSEN_STOCK.name
        diffsteps = config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        augmenter = config.CHOSEN_AUGMENTER
        cond_augmenter = config.CHOSEN_COND_AUGMENTER
        config.FILENAME_CKPT = str(stock_name) + "_" +  str(cond_type) + "_cond_aug_" + str(cond_augmenter) + "_" + config.COND_METHOD + "_" + str(augmenter) + "_" + wandb_instance_name + "_diffsteps_" + str(diffsteps)
        wandb_instance_name = config.FILENAME_CKPT
        trainer = L.Trainer(
            accelerator=accelerator,
            precision=cst.PRECISION,
            max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
            callbacks=[
                EarlyStopping(monitor="val_ema_loss", mode="min", patience=2, verbose=True, min_delta=0.005),
                TQDMProgressBar(refresh_rate=1000)
            ],
            num_sanity_val_steps=0,
            logger=wandb_logger,
            detect_anomaly=False,
            check_val_every_n_epoch=1,
        )

        # log simulation details in WANDB console
        run.log({"model": config.CHOSEN_MODEL.name}, commit=False)
        run.log({"stock train": config.CHOSEN_STOCK.name}, commit=False)
        run.log({"stock test": config.CHOSEN_STOCK.name}, commit=False)
        run.log({"cond type": config.COND_TYPE}, commit=False)
        run.log({"num diff steps": config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS]}, commit=False)
        run.log({"is augmentation": config.IS_AUGMENTATION}, commit=False)
        run.log({"seq size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]}, commit=False)
        run.log({"augmentation dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM]}, commit=False)
        run.log({"cdt depth": config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_DEPTH]}, commit=False)
        run.log({"cdt num heads": config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_NUM_HEADS]}, commit=False)
        run.log({"learning rate": config.HYPER_PARAMETERS[cst.LearningHyperParameter.LEARNING_RATE]}, commit=False)
        run.log({"optimizer": config.HYPER_PARAMETERS[cst.LearningHyperParameter.OPTIMIZER]}, commit=False)
        run.log({"batch size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE]}, commit=False)
        run.log({"augmenter": config.CHOSEN_AUGMENTER}, commit=False)
        run.log({"size type emb": config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB]}, commit=False)
        run.log({"cond augmenter": config.CHOSEN_COND_AUGMENTER}, commit=False)
        run.log({"cond method": config.COND_METHOD}, commit=False)
        run.log({"seed": cst.SEED}, commit=False)
        run.log({"lambda": config.HYPER_PARAMETERS[cst.LearningHyperParameter.LAMBDA]}, commit=False)        
        train(config, trainer)
        run.finish()

    return wandb_sweep_callback

def sweep_init(config: Configuration):
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

def print_setup(config: Configuration):
    print("Chosen model is: ", config.CHOSEN_MODEL.name)
    print("Is augmented: ", config.IS_AUGMENTATION)
    if config.IS_AUGMENTATION:
        print("Augmentation dim: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM])
        print("Augmenter: ", config.CHOSEN_AUGMENTER)
        print("CDT depth: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_DEPTH])
        print("CDT num heads: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.CDT_NUM_HEADS])
    print("Conditioning type: ", config.COND_TYPE)
    print("Number of diffusion steps: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS])
    print("Sequence size: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE])
    print("Batch size: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE])
    print("Learning rate: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.LEARNING_RATE])
    print("Optimizer: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.OPTIMIZER])
    print("One hot encoding type: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE])
    if not config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE]:
        print("Size order embedding: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB])  