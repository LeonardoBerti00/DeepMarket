import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from configuration import Configuration
import constants as cst
from models.gan.CGAN_hparam import HP_CGAN, HP_CGAN_FIXED
from preprocessing.DataModule import DataModule
from preprocessing.LOBDataset import LOBDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import TQDMProgressBar
from collections import namedtuple
from models.diffusers.TRADES.TRADES_hparam import HP_TRADES, HP_TRADES_FIXED
from models.gan.CGAN_hparam import HP_CGAN, HP_CGAN_FIXED
from models.diffusers.diffusion_engine import DiffusionEngine
from models.gan.gan_engine import GANEngine

HP_SEARCH_TYPES = namedtuple('HPSearchTypes', ("sweep", "fixed"))
HP_DICT_MODEL = {
    cst.Models.TRADES: HP_SEARCH_TYPES(HP_TRADES, HP_TRADES_FIXED),
    cst.Models.CGAN: HP_SEARCH_TYPES(HP_CGAN, HP_CGAN_FIXED)
}

def train(config: Configuration, trainer: L.Trainer):
    print_setup(config)
    
    # due to the fact that CGAN uses a different dataset, we need to create different numpy files
    if config.CHOSEN_MODEL == cst.Models.CGAN:
        train_data_path = cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/train_cgan.npy"
        val_data_path = cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/val_cgan.npy" 
    else:
        train_data_path = cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/train.npy"
        val_data_path = cst.DATA_DIR + "/" + config.CHOSEN_STOCK.name + "/val.npy"
        
    train_set = LOBDataset(
        path=train_data_path,
        seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE],
        one_hot_encoding_type=config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE],
        x_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
        chosen_model=config.CHOSEN_MODEL,
        chosen_stock=config.CHOSEN_STOCK,
    )

    val_set = LOBDataset(
        path=val_data_path,
        seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE],
        one_hot_encoding_type=config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE],
        x_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
        chosen_model=config.CHOSEN_MODEL,
        chosen_stock=config.CHOSEN_STOCK,
    )
    
    #print("size of train set: ", train_set.data.size())
    #print("size of val set: ", val_set.data.size())
    
    if config.IS_DEBUG:
        train_set.data = train_set.data[:256]
        val_set.data = val_set.data[:256]
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.TRADES_DEPTH] = 1
        
    data_module = DataModule(
        train_set=train_set,
        val_set=val_set,
        batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
        test_batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.TEST_BATCH_SIZE],
        num_workers=4
    )
    if config.CHOSEN_MODEL == cst.Models.CGAN:
        model = GANEngine(config)
    elif config.CHOSEN_MODEL == cst.Models.TRADES:
        model = DiffusionEngine(config)
    train_dataloader, val_dataloader = data_module.train_dataloader(), data_module.val_dataloader()
    trainer.fit(model, train_dataloader, val_dataloader)


def run(config: Configuration, accelerator, model=None):
    wandb_instance_name = ""
    model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed
    for param in cst.LearningHyperParameter:
        if param.value in model_params:
            config.HYPER_PARAMETERS[param] = model_params[param.value]
            wandb_instance_name += str(param.value[:2]) + "_" + str(model_params[param.value]) + "_"
    wandb_instance_name += f"seed_{cst.SEED}"
    
    if config.CHOSEN_MODEL == cst.Models.CGAN:
        config.FILENAME_CKPT = "CGAN_" + wandb_instance_name
        wandb_instance_name = config.FILENAME_CKPT
    else:
        cond_type = config.COND_TYPE
        is_augmentation = config.IS_AUGMENTATION
        stock_name = config.CHOSEN_STOCK.name
        diffsteps = config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        augmenter = config.CHOSEN_AUGMENTER
        aug_dim = config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM]
        config.FILENAME_CKPT = f"{stock_name}_{wandb_instance_name}"
        wandb_instance_name = config.FILENAME_CKPT

    trainer = L.Trainer(
        accelerator=accelerator,
        precision=cst.PRECISION,
        max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS],
        callbacks=[
            EarlyStopping(monitor="val_ema_loss", mode="min", patience=1, verbose=True, min_delta=0.005),
            TQDMProgressBar(refresh_rate=100)
            ],
        num_sanity_val_steps=0,
        detect_anomaly=False,
        profiler="advanced",
        check_val_every_n_epoch=1,
        limit_val_batches=50
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
                
        wandb_instance_name += f"seed_{cst.SEED}"
        
        if config.CHOSEN_MODEL == cst.Models.CGAN:
            config.FILENAME_CKPT = "CGAN_" + wandb_instance_name
            wandb_instance_name = config.FILENAME_CKPT
        elif config.CHOSEN_MODEL == cst.Models.TRADES:
            run.name = wandb_instance_name
            aug_dim = config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM]
            config.HYPER_PARAMETERS[cst.LearningHyperParameter.TRADES_NUM_HEADS] = aug_dim // 64
            stock_name = config.CHOSEN_STOCK.name
            config.FILENAME_CKPT = str(stock_name) + "_" + wandb_instance_name 
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
            logger=wandb_logger,
            detect_anomaly=False,
            profiler="advanced",
            check_val_every_n_epoch=1,
            limit_val_batches=50
        )

        # log simulation details in WANDB console
        run.log({"model": config.CHOSEN_MODEL.name}, commit=False)
        run.log({"stock train": config.CHOSEN_STOCK.name}, commit=False)
        run.log({"stock test": config.CHOSEN_STOCK.name}, commit=False)
        if config.CHOSEN_MODEL == cst.Models.TRADES:
            run.log({"cond type": config.COND_TYPE}, commit=False)
            run.log({"num diff steps": config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS]}, commit=False)
            run.log({"is augmentation": config.IS_AUGMENTATION}, commit=False)
            run.log({"seq size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]}, commit=False)
            run.log({"augmentation dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM]}, commit=False)
            run.log({"TRADES depth": config.HYPER_PARAMETERS[cst.LearningHyperParameter.TRADES_DEPTH]}, commit=False)
            run.log({"TRADES num heads": config.HYPER_PARAMETERS[cst.LearningHyperParameter.TRADES_NUM_HEADS]}, commit=False)
            run.log({"learning rate": config.HYPER_PARAMETERS[cst.LearningHyperParameter.LEARNING_RATE]}, commit=False)
            run.log({"optimizer": config.HYPER_PARAMETERS[cst.LearningHyperParameter.OPTIMIZER]}, commit=False)
            run.log({"batch size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE]}, commit=False)
            run.log({"augmenter": config.CHOSEN_AUGMENTER}, commit=False)
            run.log({"size type emb": config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB]}, commit=False)
            run.log({"cond augmenter": config.CHOSEN_COND_AUGMENTER}, commit=False)
            run.log({"cond method": config.COND_METHOD}, commit=False)
            run.log({"seed": cst.SEED}, commit=False)
            run.log({"lambda": config.HYPER_PARAMETERS[cst.LearningHyperParameter.LAMBDA]}, commit=False)        
        elif config.CHOSEN_MODEL == cst.Models.CGAN:
            run.log({"seq size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]}, commit=False)
            run.log({"market features dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.MARKET_FEATURES_DIM]}, commit=False)
            run.log({"order features dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.ORDER_FEATURES_DIM]}, commit=False)
            run.log({"generator LSTM hidden state dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_LSTM_HIDDEN_STATE_DIM]}, commit=False)
            run.log({"generator FC hidden dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_FC_HIDDEN_DIM]}, commit=False)
            run.log({"generator kernel size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_KERNEL_SIZE]}, commit=False)
            run.log({"generator num FC layers": config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_NUM_FC_LAYERS]}, commit=False)
            run.log({"generator num conv layers": config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_NUM_CONV_LAYERS]}, commit=False)
            run.log({"generator stride": config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_STRIDE]}, commit=False)
            run.log({"discriminator LSTM hidden state dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_LSTM_HIDDEN_STATE_DIM]}, commit=False)
            run.log({"discriminator FC hidden dim": config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_FC_HIDDEN_DIM]}, commit=False)
            run.log({"discriminator num FC layers": config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_NUM_FC_LAYERS]}, commit=False)
            run.log({"discriminator num conv layers": config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_NUM_CONV_LAYERS]}, commit=False)
            run.log({"discriminator kernel size": config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_KERNEL_SIZE]}, commit=False)
            run.log({"discriminator stride": config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_STRIDE]}, commit=False)
            run.log({"seed": cst.SEED}, commit=False)
        train(config, trainer)
        run.finish()

    return wandb_sweep_callback

def sweep_init(config: Configuration):
    # put your wandb key here
    wandb.login()
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
    if config.CHOSEN_MODEL == cst.Models.TRADES:
        print("Is augmented: ", config.IS_AUGMENTATION)
        if config.IS_AUGMENTATION:
            print("Augmentation dim: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.AUGMENT_DIM])
            print("Augmenter: ", config.CHOSEN_AUGMENTER)
            print("TRADES depth: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.TRADES_DEPTH])
            print("TRADES num heads: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.TRADES_NUM_HEADS])
        print("Conditioning type: ", config.COND_TYPE)
        print("Number of diffusion steps: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS])
        print("Sequence size: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE])
        print("Batch size: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE])
        print("Learning rate: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.LEARNING_RATE])
        print("Optimizer: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.OPTIMIZER])
        print("One hot encoding type: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE])
        if not config.HYPER_PARAMETERS[cst.LearningHyperParameter.ONE_HOT_ENCODING_TYPE]:
            print("Size order embedding: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB])  
    elif config.CHOSEN_MODEL == cst.Models.CGAN:
        #self.HYPER_PARAMETERS[LearningHyperParameter.SEQ_LEN] = 256
        print("Sequence size: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE])
        print("Market features dim: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.MARKET_FEATURES_DIM])
        print("Order features dim: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.ORDER_FEATURES_DIM])
        print("Generator LSTM hidden state dim: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_LSTM_HIDDEN_STATE_DIM])
        print("Generator FC hidden dim: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_FC_HIDDEN_DIM])
        print("Generator kernel size: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_KERNEL_SIZE])
        print("Generator num FC layers: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_NUM_FC_LAYERS])
        print("Generator num conv layers: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_NUM_CONV_LAYERS])
        print("Generator stride: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.GENERATOR_STRIDE])
        print("Discriminator LSTM hidden state dim: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_LSTM_HIDDEN_STATE_DIM]) 
        print("Discriminator FC hidden dim: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_FC_HIDDEN_DIM])
        print("Discriminator num FC layers: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_NUM_FC_LAYERS])
        print("Discriminator num conv layers: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_NUM_CONV_LAYERS])
        print("Discriminator kernel size: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_KERNEL_SIZE])
        print("Discriminator stride: ", config.HYPER_PARAMETERS[cst.LearningHyperParameter.DISCRIMINATOR_STRIDE])        