from constants import GANHyperParameters, LearningHyperParameter
import constants as cst
from utils.utils import noise_scheduler


class Configuration:

    def __init__(self):

        self.IS_WANDB = False
        self.IS_SWEEP = False
        self.IS_TRAINING = False
        self.IS_DEBUG = False
        self.IS_EVALUATION = True

        self.VALIDATE_EVERY = 1

        self.IS_AUGMENTATION = True

        self.IS_DATA_PREPROCESSED = True
        self.SPLIT_RATES = (.85, .05, .10)

        self.CHOSEN_MODEL = cst.Models.CDT
        self.CHOSEN_AUGMENTER = "MLP"
        self.CHOSEN_COND_AUGMENTER = "MLP"
        self.USE_ENGINE = cst.Engine.GAN_ENGINE
        
        if self.CHOSEN_MODEL == cst.Models.CDT:
            cst.PROJECT_NAME = "CDTS"
        elif self.CHOSEN_MODEL == cst.Models.CSDI:
            cst.PROJECT_NAME = "CSDI"
        elif self.CHOSEN_MODEL == cst.Models.CGAN:
            cst.PROJECT_NAME = "CGAN"

        self.CHOSEN_STOCK = cst.Stocks.TSLA

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.WANDB_SWEEP_NAME = None

        self.IS_SHUFFLE_TRAIN_SET = True

        # insert the path of the generated and real orders with a relative path
        self.REAL_DATA_PATH = "ABIDES/log/paper/market_replay_TSLA_2015-01-29_12-00-00/processed_orders.csv"
        self.CDT_DATA_PATH = "ABIDES/log/paper/world_agent_TSLA_2015-01-29_12-00-00_val_ema=0.811_epoch=3_seed_40/processed_orders.csv"
        self.IABS_DATA_PATH = "ABIDES/log/paper/IABS_TSLA_20150129_120000/processed_orders.csv"

        self.HYPER_PARAMETERS = {lp: None for lp in LearningHyperParameter}

        self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 256
        self.HYPER_PARAMETERS[LearningHyperParameter.TEST_BATCH_SIZE] = 512
        self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.001
        self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS] = 50
        self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.ADAM.value

        self.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE] = 256        #it's the sequencce length
        self.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE] = 1      #it's the number of elements to be masked, so the events that we generate at a time

        self.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT] = 0.0
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS] = 100
        self.HYPER_PARAMETERS[LearningHyperParameter.SIZE_TYPE_EMB] = 3
        self.HYPER_PARAMETERS[LearningHyperParameter.ONE_HOT_ENCODING_TYPE] = False
        if not self.HYPER_PARAMETERS[LearningHyperParameter.ONE_HOT_ENCODING_TYPE]:
            self.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB] = cst.LEN_EVENT + self.HYPER_PARAMETERS[LearningHyperParameter.SIZE_TYPE_EMB] - 1
        else:
            self.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB] = cst.LEN_EVENT
        
        self.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA] = 0.01       #its the parameter used in the loss function to prevent L_vlb from overwhleming L_simple
        self.HYPER_PARAMETERS[LearningHyperParameter.P_NORM] = 2 if self.CHOSEN_STOCK == cst.Stocks.INTC else 5
        self.HYPER_PARAMETERS[LearningHyperParameter.REG_TERM_WEIGHT] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.CDT_DEPTH] = 8
        self.HYPER_PARAMETERS[LearningHyperParameter.CDT_MLP_RATIO] = 4
        self.HYPER_PARAMETERS[LearningHyperParameter.CDT_NUM_HEADS] = 2

        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_SIDE_DIM] = 10
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_CHANNELS] = 2
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_DIFFUSION_STEP_EMB_DIM] = 64
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_EMBEDDING_TIME_DIM] = 64
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_EMBEDDING_FEATURE_DIM] = 16
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_LAYERS] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_N_HEADS] = 2
        self.BETAS = noise_scheduler(num_diffusion_timesteps=self.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS])

        self.COND_METHOD = "concatenation"
        self.COND_TYPE = "only_event"  # it can be full or only_event or only_lob
        if self.COND_TYPE == "full":
            self.COND_SIZE = cst.LEN_LEVEL * cst.N_LOB_LEVELS
        elif self.COND_TYPE == "only_event":
            self.COND_SIZE = self.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB]

        self.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM] = 64
        
        # generator's hyperparameters    
        self.HYPER_PARAMETERS[GANHyperParameters.SEQ_LEN] = 256
        self.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_LSTM_INPUT_DIM] = 9
        self.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_LSTM_HIDDEN_STATE_DIM] = 100
        self.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_FC_HIDDEN_DIM] = 50
        self.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_KERNEL_SIZE] = 3
        self.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_NUM_FC_LAYERS] = 2
        self.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_NUM_CONV_LAYERS] = 2
        self.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_STRIDE] = 1
        self.HYPER_PARAMETERS[GANHyperParameters.ORDER_FEATURES_DIM] = 7
        # discriminator's hyperparameters
        self.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_LSTM_INPUT_DIM] = self.HYPER_PARAMETERS[GANHyperParameters.GENERATOR_LSTM_INPUT_DIM] + self.HYPER_PARAMETERS[GANHyperParameters.ORDER_FEATURES_DIM]
        self.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_LSTM_HIDDEN_STATE_DIM] = 100
        self.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_FC_HIDDEN_DIM] = 50
        self.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_NUM_FC_LAYERS] = 2
        self.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_NUM_CONV_LAYERS] = 2
        self.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_KERNEL_SIZE] = 3
        self.HYPER_PARAMETERS[GANHyperParameters.DISCRIMINATOR_STRIDE] = 1