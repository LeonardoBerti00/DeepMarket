from constants import CSDIParameters, LearningHyperParameter
import constants as cst

class Configuration:

    def __init__(self):

        self.IS_WANDB = False
        self.IS_SWEEP = False
        self.IS_TESTING = False
        self.IS_TRAINING = True
        self.IS_DEBUG = True

        assert (self.IS_WANDB + self.IS_TESTING + self.IS_TRAINING) == 1

        self.VALIDATE_EVERY = 1

        self.IS_AUGMENTATION = True

        self.IS_DATA_PREPROCESSED = False

        self.SPLIT_RATES = (.65, .05, .3)

        self.CHOSEN_MODEL = cst.Models.CDT

        self.CHOSEN_STOCK = cst.Stocks.TSLA

        self.HP_SEARCH_METHOD = 'bayes'  # 'bayes'

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.WANDB_SWEEP_NAME = None

        self.NUM_WORKERS = 4

        self.EARLY_STOPPING_METRIC = None
        self.IS_SHUFFLE_TRAIN_SET = True

        self.HYPER_PARAMETERS = {lp: None for lp in LearningHyperParameter}

        self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.TEST_BATCH_SIZE] = 1024
        self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.01
        self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS] = 30
        self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.ADAM.value

        self.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE] = 50        #it's the sequencce length
        self.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE] = 1      #it's the number of elements to be masked, so the events that we generate at a time
        self.COND_SEQ_SIZE = self.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE] - self.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE]

        self.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM] = 40
        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS] = 50
        self.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA] = 0.0001       #its the parameter used in the loss function to prevent L_vlb from overwhleming L_simple
        self.HYPER_PARAMETERS[LearningHyperParameter.CDT_DEPTH] = 12
        self.HYPER_PARAMETERS[LearningHyperParameter.CDT_MLP_RATIO] = 4
        self.HYPER_PARAMETERS[LearningHyperParameter.CDT_NUM_HEADS] = 8
        self.COND_METHOD = "concatenation"

        self.COND_TYPE = "only_event"  # it can be full or only_event or only_lob
        self.BETAS = None
        if self.COND_TYPE == "full":
            self.COND_SIZE = cst.LEN_LEVEL * cst.N_LOB_LEVELS + cst.LEN_EVENT_ONE_HOT
            self.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM] = 44
        elif self.COND_TYPE == "only_event":
            self.COND_SIZE = cst.LEN_EVENT_ONE_HOT
        elif self.COND_TYPE == "only_lob":
            self.COND_SIZE = cst.LEN_LEVEL * cst.N_LOB_LEVELS
            self.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM] = 40

        #da cmbiare come per CDT
        self.CSDI_HYPERPARAMETERS = {lp: None for lp in CSDIParameters}
        
        self.CSDI_HYPERPARAMETERS[CSDIParameters.N_HEADS] = 2
        self.CSDI_HYPERPARAMETERS[CSDIParameters.SIDE_DIM] = 10
        self.CSDI_HYPERPARAMETERS[CSDIParameters.CHANNELS] = 2
        self.CSDI_HYPERPARAMETERS[CSDIParameters.DIFFUSION_STEP_EMB_DIM] = 128
        self.CSDI_HYPERPARAMETERS[CSDIParameters.EMBEDDING_TIME_DIM] = 128
        self.CSDI_HYPERPARAMETERS[CSDIParameters.EMBEDDING_FEATURE_DIM] = 16
        self.CSDI_HYPERPARAMETERS[CSDIParameters.LAYERS] = 1







