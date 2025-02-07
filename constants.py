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
    REG_TERM_WEIGHT = "reg_term_weight"
    P_NORM = "p_norm"
    DDIM_ETA = "ddim_eta"
    DDIM_NSTEPS = "ddim_nsteps"
    ONE_HOT_ENCODING_TYPE = "one_hot_encoding_type"
    CSDI_SIDE_DIM = "CSDI_side_dim"
    CSDI_CHANNELS = "CSDI_channels"
    CSDI_DIFFUSION_STEP_EMB_DIM = "CSDI_diffusion_step_emb_dim"
    CSDI_EMBEDDING_TIME_DIM = "CSDI_embedding_time_dim"
    CSDI_EMBEDDING_FEATURE_DIM = "CSDI_embedding_feature_dim"
    CSDI_LAYERS = "CSDI_layers"
    CSDI_N_HEADS = "CSDI_n_heads"
    MARKET_FEATURES_DIM = "market_features_dim"
    ORDER_FEATURES_DIM = "order_features_dim"
    GENERATOR_CHANNELS = "gen_channels"
    GENERATOR_LSTM_INPUT_DIM = "gen_LSTM_input_dimensions"
    GENERATOR_LSTM_HIDDEN_STATE_DIM = "gen_LSTM_hidden_state_dim"
    GENERATOR_NUM_FC_LAYERS = "gen_num_fc_layers"
    GENERATOR_FC_HIDDEN_DIM = "gen_fc_hidden_dim"
    GENERATOR_NUM_CONV_LAYERS = "gen_num_conv_layers"
    GENERATOR_KERNEL_SIZE = "gen_kernel_size"
    GENERATOR_STRIDE = "gen_stride"
    DISCRIMINATOR_LSTM_INPUT_DIM = "disc_LSTM_input_dimensions"
    DISCRIMINATOR_LSTM_HIDDEN_STATE_DIM = "disc_LSTM_hidden_state_dim"
    DISCRIMINATOR_NUM_FC_LAYERS = "disc_num_fc_layers"
    DISCRIMINATOR_FC_HIDDEN_DIM = "disc_fc_hidden_dim"
    DISCRIMINATOR_NUM_CONV_LAYERS = "disc_num_conv_layers"
    DISCRIMINATOR_KERNEL_SIZE = "disc_kernel_size"
    DISCRIMINATOR_STRIDE = "disc_stride"
    DISCRIMINATOR_CHANNELS = "disc_channels"
    
    


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
    TRADES = "TRADES"
    CGAN = "CGAN"
    CDT = "CDT"

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
    GOOG = "GOOG"


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
    

class Engine(str, Enum):    
    """NN_ENGINE = "NNEngine"
    GAN_ENGINE = "GANEngine"""
    DIFFUSION_ENGINE = "models.diffusers.DiffusionEngine"
    GAN_ENGINE = "models.gan.GANEngine"

    

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

# these are the values for the market features used by CGAN
TSLA_MEAN_SPREAD = 1628.1331238445746
TSLA_STD_SPREAD = 823.685980941235
TSLA_MEAN_RETURN = 2.471099866467089e-07
TSLA_STD_RETURN = 0.00020927952921847475
TSLA_MEAN_VOL_IMB = 0.5036961437566201
TSLA_STD_VOL_IMB = 0.18250211511475767
TSLA_MEAN_ABS_VOL = 965.1632447653776
TSLA_STD_ABS_VOL = 1285.16124777206
TSLA_MEAN_CANCEL_DEPTH = 1.2893666222896607
TSLA_STD_CANCEL_DEPTH = 2.1555155776464994
TSLA_MEAN_SIZE_100 = 0.6347363685292531
TSLA_STD_SIZE_100 = 0.8520664436360541


# for 15 days of INTC
INTC_LOB_MEAN_SIZE_10 = 6222.424274871972
INTC_LOB_STD_SIZE_10 = 7538.341086370264
INTC_LOB_MEAN_PRICE_10 = 3635.766219937785
INTC_LOB_STD_PRICE_10 = 44.15649995373795

INTC_EVENT_MEAN_SIZE = 324.6800802006092
INTC_EVENT_STD_SIZE = 574.5781447696605
INTC_EVENT_MEAN_PRICE = 3635.78165265669
INTC_EVENT_STD_PRICE = 43.872407609651184
INTC_EVENT_MEAN_TIME = 0.025201754040915927
INTC_EVENT_STD_TIME = 0.11013627432323592
INTC_EVENT_MEAN_DEPTH = 1.3685517399834501
INTC_EVENT_STD_DEPTH = 2.333747222206966

INTC_MEAN_SPREAD = 116.59695974561068
INTC_STD_SPREAD = 39.33230591185948
INTC_MEAN_RETURN = -5.2581575805820944e-08
INTC_STD_RETURN = 8.171171316578973e-05
INTC_MEAN_VOL_IMB = 0.5005629676888042
INTC_STD_VOL_IMB = 0.1838374952729647
INTC_MEAN_ABS_VOL = 40100.45630062603
INTC_STD_ABS_VOL = 43213.292109848255
INTC_MEAN_CANCEL_DEPTH = 0.649548691430768
INTC_STD_CANCEL_DEPTH = 1.6964303084449814
INTC_MEAN_SIZE_100 = 3.040093999614961
INTC_STD_SIZE_100 = 5.5826348200688924



SEED = 30

PRECISION = 32
N_LOB_LEVELS = 10
LEN_LEVEL = 4
LEN_ORDER = 6
LEN_ORDER_CGAN = 7

DATE_TRADING_DAYS = ["2012-06-21", "2012-06-21"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIR_EXPERIMENTS = "data/experiments"
DIR_SAVED_MODEL = "data/checkpoints"
DATA_DIR = "data"
RECON_DIR = "data/reconstructions"
PROJECT_NAME = ""
SPLIT_RATES = (.85, .05, .10)



