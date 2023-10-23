from constants import CSDIParameters, LearningHyperParameter
import constants as cst

class Configuration:

    def __init__(self):

        self.IS_DEBUG = False
        self.IS_TEST_ONLY = False

        self.SEED = 500
        self.VALIDATE_EVERY = 1

        self.IS_AUGMENTATION_X = True
        self.IS_AUGMENTATION_COND = False
        self.IS_TRAINING = True

        self.IS_DATA_PREPROCESSED = True

        self.SPLIT_RATES = (.65, .05, .3)

        self.CHOSEN_MODEL = cst.Models.DDPM

        self.CHOSEN_STOCK = cst.Stocks.TSLA
        self.DATE_TRADING_DAYS = ["2015-01-02", "2015-01-30"]

        self.IS_SWEEP = False

        self.HP_SEARCH_METHOD = 'bayes'  # 'bayes'

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.WANDB_SWEEP_NAME = None

        self.JSON_DIRECTORY = ""
        self.NUM_WORKERS = 4

        self.EARLY_STOPPING_METRIC = None

        self.HYPER_PARAMETERS = {lp: None for lp in LearningHyperParameter}

        self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.01
        self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS] = 100
        self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.ADAM.value
        self.HYPER_PARAMETERS[LearningHyperParameter.WEIGHT_DECAY] = 0.0
        self.HYPER_PARAMETERS[LearningHyperParameter.EPS] = 1e-08  # default value for ADAM
        self.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM] = 0.9

        self.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE] = 50        #it's the sequencce length
        self.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE] = 1      #it's the number of elements to be masked, so the events that we generate at a time
        self.HYPER_PARAMETERS[LearningHyperParameter.IS_SHUFFLE_TRAIN_SET] = True

        self.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM] = 32
        self.HYPER_PARAMETERS[LearningHyperParameter.DIFFUSION_STEPS] = 100
        self.HYPER_PARAMETERS[LearningHyperParameter.S] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA] = 0.0001       #its the parameter used in the loss function to prevent L_vlb from overwhleming L_simple
        self.HYPER_PARAMETERS[LearningHyperParameter.EMB_T_DIM] = 2
        self.HYPER_PARAMETERS[LearningHyperParameter.DiT_DEPTH] = 12
        self.HYPER_PARAMETERS[LearningHyperParameter.DiT_MLP_RATIO] = 4
        self.HYPER_PARAMETERS[LearningHyperParameter.DiT_NUM_HEADS] = 8
        self.HYPER_PARAMETERS[LearningHyperParameter.DiT_HIDDEN_SIZE] = 64
        self.HYPER_PARAMETERS[LearningHyperParameter.DiT_TYPE] = "adaln_zero"
        self.HYPER_PARAMETERS[LearningHyperParameter.COND_TYPE] = "full"    #it can be full or 'only_event'

        self.ALPHAS_DASH, self.BETAS = None, None

        self.CSDI_HYPERPARAMETERS = {lp: None for lp in CSDIParameters}
        
        self.CSDI_HYPERPARAMETERS[CSDIParameters.N_HEADS] = 2
        self.CSDI_HYPERPARAMETERS[CSDIParameters.SIDE_DIM] = 10
        self.CSDI_HYPERPARAMETERS[CSDIParameters.CHANNELS] = 2
        self.CSDI_HYPERPARAMETERS[CSDIParameters.DIFFUSION_STEP_EMB_DIM] = 128







