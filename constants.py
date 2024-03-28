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
    SIZE_TYPE_EMB = "size_type_emb"
    SIZE_ORDER_EMB = "size_order_emb"
    LAMBDA = "lambda"
    CDT_DEPTH = "CDT_depth"
    CDT_MLP_RATIO = "CDT_mlp_ratio"
    CDT_NUM_HEADS = "CDT_num_heads"
    TEST_BATCH_SIZE = "test_batch_size"
    CSDI_SIDE_DIM = "CSDI_side_dim"
    CSDI_CHANNELS = "CSDI_channels"
    CSDI_DIFFUSION_STEP_EMB_DIM = "CSDI_diffusion_step_emb_dim"
    CSDI_EMBEDDING_TIME_DIM = "CSDI_embedding_time_dim"
    CSDI_EMBEDDING_FEATURE_DIM = "CSDI_embedding_feature_dim"
    CSDI_LAYERS = "CSDI_layers"
    CSDI_N_HEADS = "CSDI_n_heads"
    REG_TERM_WEIGHT = "reg_term_weight"
    ONE_HOT_ENCODING_TYPE = "one_hot_encoding_type"


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

    

# for 15 days of TSLA
TSLA_LOB_MEAN_SIZE_10 = 165.44670902537212
TSLA_LOB_STD_SIZE_10 = 481.7127061897184
TSLA_LOB_MEAN_PRICE_10 = 20180.439318660694
TSLA_LOB_STD_PRICE_10 = 814.8782058033195

TSLA_EVENT_MEAN_SIZE = 88.09459295373463
TSLA_EVENT_STD_SIZE = 86.55913199110894
TSLA_EVENT_MEAN_PRICE = 20178.610720500274
TSLA_EVENT_STD_PRICE = 813.8188032145645
TSLA_EVENT_MEAN_TIME = 0.08644932804905886
TSLA_EVENT_STD_TIME = 0.3512181506722207
TSLA_EVENT_MEAN_DEPTH = 7.365325300819055
TSLA_EVENT_STD_DEPTH = 8.59342838063813

SEED = 0

PRECISION = 32
N_LOB_LEVELS = 10
LEN_LEVEL = 4
LEN_EVENT = 6

DATE_TRADING_DAYS = ["2015-01-02", "2015-01-30"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIR_EXPERIMENTS = "data/experiments"
DIR_SAVED_MODEL = "data/checkpoints"
DATA_DIR = "data"
RECON_DIR = "data/reconstructions"
PROJECT_NAME = "CDTS"


