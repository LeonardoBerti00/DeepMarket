from constants import LearningHyperParameter
import constants as cst
from utils.utils import noise_scheduler


class Configuration:

    def __init__(self):

        self.IS_WANDB = False
        self.IS_SWEEP = False
        self.IS_TRAINING = True
        self.IS_DEBUG = False

        # evaluation
        self.QUANT_METRICS = False # to activate the predictive and discriminative evaluation

        self.PLOT_GRAPHS = False #to activate the plot of the graphs

        assert (self.IS_WANDB + self.IS_TRAINING) == 1

        self.VALIDATE_EVERY = 1

        self.IS_AUGMENTATION = True

        self.IS_DATA_PREPROCESSED = True

        self.SPLIT_RATES = (.75, .05, .2)

        self.CHOSEN_MODEL = cst.Models.CDT
        self.CHOSEN_AUGMENTER = "MLP"
        
        if self.CHOSEN_MODEL == cst.Models.CDT:
            cst.PROJECT_NAME = "CDTS"
        elif self.CHOSEN_MODEL == cst.Models.CSDI:
            cst.PROJECT_NAME = "CSDI"

        self.CHOSEN_STOCK = cst.Stocks.TSLA

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.WANDB_SWEEP_NAME = None

        self.IS_SHUFFLE_TRAIN_SET = True

        # insert the path of the generated and real orders with a relative path
        self.REAL_DATA_PATH = ""
        self.GEN_DATA_PATH = ""

        self.HYPER_PARAMETERS = {lp: None for lp in LearningHyperParameter}

        self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 256
        self.HYPER_PARAMETERS[LearningHyperParameter.TEST_BATCH_SIZE] = 512
        self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.001
        self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS] = 20
        self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.ADAM.value

        self.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE] = 128        #it's the sequencce length
        self.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE] = 1      #it's the number of elements to be masked, so the events that we generate at a time
        self.COND_SEQ_SIZE = self.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE] - self.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE]

        self.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT] = 0.1
        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS] = 100
        self.HYPER_PARAMETERS[LearningHyperParameter.SIZE_TYPE_EMB] = 1    # try higher values
        self.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB] = cst.LEN_EVENT + self.HYPER_PARAMETERS[LearningHyperParameter.SIZE_TYPE_EMB] - 1
        self.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA] = 0.01       #its the parameter used in the loss function to prevent L_vlb from overwhleming L_simple
        self.HYPER_PARAMETERS[LearningHyperParameter.REG_TERM_WEIGHT] = 1

        self.HYPER_PARAMETERS[LearningHyperParameter.CDT_DEPTH] = 8
        self.HYPER_PARAMETERS[LearningHyperParameter.CDT_MLP_RATIO] = 4
        self.HYPER_PARAMETERS[LearningHyperParameter.CDT_NUM_HEADS] = 2
        self.COND_METHOD = "concatenation"

        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_SIDE_DIM] = 10
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_CHANNELS] = 2
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_DIFFUSION_STEP_EMB_DIM] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_EMBEDDING_TIME_DIM] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_EMBEDDING_FEATURE_DIM] = 16
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_LAYERS] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.CSDI_N_HEADS] = 2
        self.BETAS = noise_scheduler(num_diffusion_timesteps=self.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_DIFFUSIONSTEPS])

        self.COND_TYPE = "only_event"  # it can be full or only_event or only_lob
        if self.COND_TYPE == "full":
            self.COND_SIZE = cst.LEN_LEVEL * cst.N_LOB_LEVELS + cst.LEN_EVENT + self.HYPER_PARAMETERS[LearningHyperParameter.SIZE_TYPE_EMB] - 1
        elif self.COND_TYPE == "only_event":
            self.COND_SIZE = cst.LEN_EVENT + self.HYPER_PARAMETERS[LearningHyperParameter.SIZE_TYPE_EMB] - 1
        elif self.COND_TYPE == "only_lob":
            self.COND_SIZE = cst.LEN_LEVEL * cst.N_LOB_LEVELS

        self.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM] = 32









