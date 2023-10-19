from constants import LearningHyperParameter
import constants as cst

class Configuration:

    def __init__(self):

        self.IS_DEBUG = False
        self.IS_TEST_ONLY = False

        self.SEED = 500
        self.VALIDATE_EVERY = 1

        self.IS_AUGMENTATION = True
        self.IS_TRAINING = True

        self.IS_DATA_PREPROCESSED = False

        self.SPLIT_RATES = (.65, .05, .3)
        self.N_LOB_LEVELS = 3
        self.CHOSEN_MODEL = cst.Models.crea_un_modello_stronzo

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

        self.HYPER_PARAMETERS[LearningHyperParameter.WINDOW_SIZE] = 50        #it's the sequencce length
        self.HYPER_PARAMETERS[LearningHyperParameter.MASKED_WINDOW_SIZE] = 1      #it's the number of elements to be masked, so the events that we generate at a time
        self.HYPER_PARAMETERS[LearningHyperParameter.IS_SHUFFLE_TRAIN_SET] = True

        self.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.LATENT_DIM] = 32
        self.HYPER_PARAMETERS[LearningHyperParameter.DIFFUSION_STEPS] = 100
        self.HYPER_PARAMETERS[LearningHyperParameter.S] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA] = 0.0001       #its the parameter used in the loss function to prevent L_vlb from overwhleming L_simple








