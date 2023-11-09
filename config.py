from constants import CSDIParameters, LearningHyperParameter
import constants as cst

class Configuration:

    def __init__(self):

        self.IS_WANDB = False
        self.IS_SWEEP = False
        self.IS_TESTING = False
        self.IS_TRAINING = True

        self.VALIDATE_EVERY = 1

        self.IS_AUGMENTATION = True


        self.IS_DATA_PREPROCESSED = False

        self.SPLIT_RATES = (.65, .05, .3)

        self.CHOSEN_MODEL = cst.Models.CSDI

        self.CHOSEN_STOCK = cst.Stocks.TSLA
        self.DATE_TRADING_DAYS = ["2015-01-02", "2015-01-30"]

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

        self.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE] = 50        #it's the sequencce length
        self.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE] = 1      #it's the number of elements to be masked, so the events that we generate at a time
        self.COND_SEQ_SIZE = self.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE] - self.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE]
        self.HYPER_PARAMETERS[LearningHyperParameter.IS_SHUFFLE_TRAIN_SET] = True

        self.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM] = 32
        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_TIMESTEPS] = 1000
        self.HYPER_PARAMETERS[LearningHyperParameter.S] = 0.008           #value taken from the papre IDDPM
        self.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA] = 0.0001       #its the parameter used in the loss function to prevent L_vlb from overwhleming L_simple
        self.HYPER_PARAMETERS[LearningHyperParameter.DiT_DEPTH] = 12
        self.HYPER_PARAMETERS[LearningHyperParameter.DiT_MLP_RATIO] = 4
        self.HYPER_PARAMETERS[LearningHyperParameter.DiT_NUM_HEADS] = 8
        self.HYPER_PARAMETERS[LearningHyperParameter.DiT_TYPE] = "adaln_zero"
        self.HYPER_PARAMETERS[LearningHyperParameter.COND_TYPE] = "only_event"    #it can be full or 'only_event'
        self.ALPHAS_CUMPROD, self.BETAS = None, None
        self.COND_SIZE = cst.LEN_LEVEL * cst.N_LOB_LEVELS + cst.LEN_EVENT if self.HYPER_PARAMETERS[LearningHyperParameter.COND_TYPE] == 'full' else cst.LEN_EVENT

        #da cmbiare come per DiT
        self.CSDI_HYPERPARAMETERS = {lp: None for lp in CSDIParameters}
        
        self.CSDI_HYPERPARAMETERS[CSDIParameters.N_HEADS] = 2
        self.CSDI_HYPERPARAMETERS[CSDIParameters.SIDE_DIM] = 10
        self.CSDI_HYPERPARAMETERS[CSDIParameters.CHANNELS] = 2
        self.CSDI_HYPERPARAMETERS[CSDIParameters.DIFFUSION_STEP_EMB_DIM] = 128
        self.CSDI_HYPERPARAMETERS[CSDIParameters.EMBEDDING_TIME_DIM] = 128
        self.CSDI_HYPERPARAMETERS[CSDIParameters.EMBEDDING_FEATURE_DIM] = 16
        self.CSDI_HYPERPARAMETERS[CSDIParameters.LAYERS] = 1

    def wandb_config_setup(self):
        self.WANDB_SWEEP_NAME = self.cf_name_format().format(
            self.CHOSEN_MODEL.name,
            self.CHOSEN_STOCK.name,
            self.HYPER_PARAMETERS[cst.LearningHyperParameter.COND_TYPE],
        )




