from enum import Enum
import torch


class LearningHyperParameter(str, Enum):
    NUM_DIFFUSIONSTEPS = "num_diffusionsteps"
    OPTIMIZER = "optimizer_name"
    LEARNING_RATE = "lr"
    EPOCHS = "epochs"
    BATCH_SIZE = "batch_size"
    CONDITIONAL_DROPOUT = "conditional_dropout"
    DROPOUT = "dropout"
    SEQ_SIZE = "seq_size"          #it's the sequence length
    MASKED_SEQ_SIZE = "masked_seq_size"
    AUGMENT_DIM = "augment_dim"
    LAMBDA = "lambda"
    CDT_DEPTH = "CDT_depth"
    CDT_MLP_RATIO = "CDT_mlp_ratio"
    CDT_NUM_HEADS = "CDT_num_heads"
    TEST_BATCH_SIZE = "test_batch_size"


class Optimizers(Enum):
    ADAM = "Adam"
    RMSPROP = "RMSprop"
    SGD = "SGD"
    LION = "LION"


class Metrics(Enum):      #Quantitative evaluation
    test_loss = 'test_loss'
    pred_score = 'pred_score'
    disc_score = 'disc_score'
    js_shannon = 'js_shannon'
    kolmogorov_smirnov = 'kolmogorov_smirnov'

class Models(str, Enum):
    CDT = "CDT"
    CSDI = "CSDI"

class LOB_Charts(Enum):      #Qualitative evaluation

    #real vs generated distribution
    t_sne = 't_sne'
    density_volume = 'density_volume'
    density_price = 'density_price'
    histogram_direction = 'density_direction'
    density_interarrival = 'density_interarrival'
    histogram_type = 'density_type'
    volume_first_time = 'volume_first_time'
    in_volume_min_time = 'in_volume_min_time'
    depth_time = 'depth_time'
    spread_time = 'spread_time'

    #market_experiment charts
    market_experiment_mid_price_time = 'market_experiment_mid_price_time'
    market_experiment_mid_price_difference_time = 'market_experiment_mid_price_difference_time'

    #stylized facts
    minutely_log_returns = 'minutely_log_returns'
    volume_correlation = 'volume_correlation'
    autocorrelation =  'autocorrelation'
    volatility_clustering = 'volatility_clustering'
    agregation_normality = 'agregation_normality'
    order_volume = 'order_volume'
    quoote_interarrival_time = 'quoote_interarrival_time'
    time_to_first_fill = 'time_to_first_fill'
    num_lim_orders_time_SEQ = 'num_lim_orders_time_SEQ'


class Stocks(Enum):
    APPL = "AAPL"
    INTC = "INTC"
    TSLA = "TSLA"
    AVXL = "AVXL"


class OrderEvent(Enum):
    """ The possible kind of orders in the lob """
    SUBMISSION = 1
    CANCELLATION = 2
    DELETION = 3
    EXECUTION = 4


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"
    
    
class CSDIParameters(Enum):
    CHANNELS = 4
    SIDE_DIM = 10
    N_HEADS = 2
    DIFFUSION_STEP_EMB_DIM = 128
    EMBEDDING_TIME_DIM = 128
    EMBEDDING_FEATURE_DIM = 16
    LAYERS = 1


LOB_MEAN_SIZE_10 = 166.30953649243466
LOB_STD_SIZE_10 = 504.06824347178383
LOB_MEAN_PRICE_10 = 2019441.963735062
LOB_STD_PRICE_10 = 86751.57157762564

EVENT_MEAN_SIZE = 7.223659859967962
EVENT_STD_SIZE = 124.45374263781343
EVENT_MEAN_PRICE = 2019249.3560175144
EVENT_STD_PRICE = 86641.41068006848
EVENT_MEAN_TIME = 0.006584290788104395
EVENT_STD_TIME = 42.99986459166484

SEED = 0

PRECISION = 32
N_LOB_LEVELS = 10
LEN_LEVEL = 4
LEN_EVENT_ONE_HOT = 7
LEN_EVENT = 4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIR_EXPERIMENTS = "data/experiments"
DIR_SAVED_MODEL = "data/checkpoints"
DATA_DIR = "data"
RECON_DIR = "data/reconstructions"
PROJECT_NAME = "CDTS"

