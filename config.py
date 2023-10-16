from constants import LearningHyperParameter
import constants as cst

class Configuration:

    def __init__(self):

        self.IS_DEBUG = False
        self.IS_TEST_ONLY = False

        self.SEED = 500
        self.VALIDATE_EVERY = 1

        self.IS_DATA_PRELOAD = True

        self.TRAIN_VAL_TEST_SPLIT = (.8, .1, .1)  # META Only

        self.CHOSEN_DATASET = cst.DatasetFamily.FI
        self.CHOSEN_MODEL = cst.Models.METALOB

        self.CHOSEN_STOCKS = {
            cst.STK_OPEN.TRAIN: cst.Stocks.FI,
            cst.STK_OPEN.TEST: cst.Stocks.FI
        }

        self.IS_WANDB = True
        self.IS_TUNE_H_PARAMS = False

        self.HP_SEARCH_METHOD = 'bayes'  # 'bayes'

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.WANDB_SWEEP_NAME = None

        self.SWEEP_METRIC = {
            'goal': 'maximize',
            'name': None
        }

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

        self.HYPER_PARAMETERS[LearningHyperParameter.BACKWARD_WINDOW_SIZE] = 10
        self.HYPER_PARAMETERS[LearningHyperParameter.IS_SHUFFLE_TRAIN_SET] = True

        self.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.N_LOB_LEVELS] = 0
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.1









