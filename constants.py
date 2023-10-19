from enum import Enum
import torch


class LearningHyperParameter(str, Enum):
    DIFFUSION_STEPS = "diffusion_steps"
    BACKWARD_WINDOW_SIZE = "backward_window_size"
    OPTIMIZER = "optimizer_name"
    LEARNING_RATE = "lr"
    WEIGHT_DECAY = "weight_decay"
    EPS = "eps"
    MOMENTUM = "momentum"
    EPOCHS = "epochs"
    IS_SHUFFLE_TRAIN_SET = "is_shuffle"
    BATCH_SIZE = "batch_size"
    CONDITIONAL_DROPOUT = "conditional_dropout"
    DROPOUT = "dropout"
    WINDOW_SIZE = "window_size"          #it's the sequence length
    MASKED_WINDOW_SIZE = "masked_window_size"
    LATENT_DIM = "latent_dim"
    S = "s"
    LAMBDA = "lambda"


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
    crea_un_modello_stronzo = "crea_un_modello_stronzo"

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
    num_lim_orders_time_window = 'num_lim_orders_time_window'


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

PRECISION = 32

LEN_LEVEL = 4
LEN_EVENT = 5

DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

DIR_EXPERIMENTS = "data/experiments"
DIR_SAVED_MODEL = "data/checkpoints"

DATA_DIR = "data"



