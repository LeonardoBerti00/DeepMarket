import os
from einops import rearrange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_compute_log_returns(file_path):
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df['minute'] = df['time'].dt.floor('T')
    df = df.query("ask_price_1 < 9999999")
    df = df.query("bid_price_1 < 9999999")
    df = df.query("ask_price_1 > -9999999")
    df = df.query("bid_price_1 > -9999999")
    df = df.groupby('minute')['MID_PRICE'].first().reset_index()
    df['log_return'] = np.log(df['MID_PRICE'].shift(1) / df['MID_PRICE'])
    df.dropna(inplace=True)
    return df['log_return']

def load_and_compute_volatility(df, i):
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df['minute'] = df['time'].dt.floor('T')
    df['second'] = df['time'].dt.second
    df = df.query("ask_price_1 < 9999999")
    df = df.query("bid_price_1 < 9999999")
    df = df.query("ask_price_1 > -9999999")
    df = df.query("bid_price_1 > -9999999")
    df = df.groupby(['minute', 'second'])['MID_PRICE'].first().reset_index()
    df['return'] = np.log(df['MID_PRICE'].shift(i)) - np.log(df['MID_PRICE'])
    #take the indexes of the nan values
    std_dev = df['return'].rolling(window=100).std().reset_index(drop=True)
    nan_indexes = std_dev[std_dev.isna()].index
    return std_dev, nan_indexes

def load_and_compute_volume(df, i):
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df['minute'] = df['time'].dt.floor('T')
    df['second'] = df['time'].dt.second
    df = df.query("ask_price_1 < 9999999")
    df = df.query("bid_price_1 < 9999999")
    df = df.query("ask_price_1 > -9999999")
    df = df.query("bid_price_1 > -9999999")
    df = df.groupby(['minute', 'second'])['SIZE'].sum().reset_index()
    volume_sum = df['SIZE']
    nan_indexes = volume_sum[volume_sum.isna()].index
    return volume_sum, nan_indexes

def load_and_compute_returns(df, i):
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df['minute'] = df['time'].dt.floor('T')
    df['second'] = df['time'].dt.second
    df = df.query("ask_price_1 < 9999999")
    df = df.query("bid_price_1 < 9999999")
    df = df.query("ask_price_1 > -9999999")
    df = df.query("bid_price_1 > -9999999")
    df = df.groupby(['minute', 'second'])['MID_PRICE'].first().reset_index()
    df['log_return'] = np.log(df['MID_PRICE'] / df['MID_PRICE'].shift(i))
    returns = df['log_return'].rolling(window=100).sum().reset_index(drop=True)
    nan_indexes = returns[returns.isna()].index
    return returns, nan_indexes

def compute_correlation_by_lag(log_returns, max_lag):
    correlations = []
    for lag in range(1, max_lag + 1, 2):
        corr = log_returns.corr(log_returns.shift(lag))
        correlations.append(corr)
    return correlations

def main(real_path, TRADES_path, iabs_path, cgan_path):
    
    log_returns_real = load_and_compute_log_returns(real_path)
    log_returns_TRADES = load_and_compute_log_returns(TRADES_path) 
    log_returns_iabs = load_and_compute_log_returns(iabs_path)
    log_returns_cgan = load_and_compute_log_returns(cgan_path)

    correlations_real = compute_correlation_by_lag(log_returns_real, 30)
    correlations_TRADES = compute_correlation_by_lag(log_returns_TRADES, 30)
    correlations_iabs = compute_correlation_by_lag(log_returns_iabs, 30)
    correlations_cgan = compute_correlation_by_lag(log_returns_cgan, 30)
    
    plt.plot(range(1, 31, 2), correlations_real, marker='o', linestyle='-', label='Real')
    plt.plot(range(1, 31, 2), correlations_TRADES, marker='o', linestyle='-', label='TRADES')
    plt.plot(range(1, 31, 2), correlations_iabs, marker='o', linestyle='-', label='IABS')
    plt.plot(range(1, 31, 2), correlations_cgan, marker='o', linestyle='-', label='CGAN')

    plt.xlabel('Lag (minutes)')
    plt.ylabel('Correlation Coefficient')
    plt.title('Log Returns Autocorrelation')
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--')
    file_name = f"corr_coef_lag_join.pdf"
    dir_path = os.path.dirname(TRADES_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()
    '''
    #TODO divide the code in two functions
    
    corr_iabs_coefs = []
    corr_real_coefs = []
    corr_TRADES_coefs = []
    corr_cgan_coefs = []
    df_TRADES = pd.read_csv(TRADES_path)
    df_real = pd.read_csv(real_path)
    df_iabs = pd.read_csv(iabs_path)
    df_cgan = pd.read_csv(cgan_path)
    
    for i in range(10000, df_real.shape[0], 10000):
        volatility_real, nan_indexes_volat_real = load_and_compute_volatility(df_real[i-10000:i],360)
        volatility_TRADES, nan_indexes_volat_TRADES = load_and_compute_volatility(df_TRADES[i-10000:i],360)
        volatility_iabs, nan_indexes_volat_iabs = load_and_compute_volatility(df_iabs[i-10000:i],360)
        volatility_cgan, nan_indexes_volat_cgan = load_and_compute_volatility(df_cgan[i-10000:i],360)
        
        volume_iabs, nan_indexes_vol_iabs = load_and_compute_volume(df_iabs[i-10000:i],360)
        volume_real, nan_indexes_vol_real = load_and_compute_volume(df_real[i-10000:i],360)
        volume_TRADES, nan_indexes_vol_TRADES = load_and_compute_volume(df_TRADES[i-10000:i],360)
        volume_cgan, nan_indexes_vol_cgan = load_and_compute_volume(df_cgan[i-10000:i],360)
        #drop the first value from volume

        nan_indexes_real = np.union1d(nan_indexes_vol_real, nan_indexes_volat_real)
        nan_indexes_TRADES = np.union1d(nan_indexes_volat_TRADES, nan_indexes_vol_TRADES)
        nan_indexes_iabs = np.union1d(nan_indexes_vol_iabs, nan_indexes_volat_iabs)
        nan_indexes_cgan = np.union1d(nan_indexes_vol_cgan, nan_indexes_volat_cgan)
        
        volume_real = volume_real.drop(nan_indexes_real)  
        volume_TRADES = volume_TRADES.drop(nan_indexes_TRADES)
        volume_iabs = volume_iabs.drop(nan_indexes_iabs)
        volume_cgan = volume_cgan.drop(nan_indexes_cgan)
        
        volatility_real = volatility_real.drop(nan_indexes_real)
        volatility_TRADES = volatility_TRADES.drop(nan_indexes_TRADES)
        volatility_iabs = volatility_iabs.drop(nan_indexes_iabs)
        volatility_cgan = volatility_cgan.drop(nan_indexes_cgan)
        print(volume_real.shape)
        corr_real_coefs.append(np.corrcoef(volume_real.values, volatility_real.values)[0, 1])
        corr_TRADES_coefs.append(np.corrcoef(volume_TRADES.values, volatility_TRADES.values)[0, 1])
        corr_iabs_coefs.append(np.corrcoef(volume_iabs.values, volatility_iabs.values)[0, 1])
        corr_cgan_coefs.append(np.corrcoef(volume_cgan.values, volatility_cgan.values)[0, 1])

    
    sns.kdeplot(corr_TRADES_coefs, bw=0.1, shade=True,  color='blue', label="TRADES")
    sns.kdeplot(corr_iabs_coefs, bw=0.1, shade=True,  color='green', label="IABS")
    sns.kdeplot(corr_real_coefs, bw=0.1, shade=True,  color='orange', label='Real')
    sns.kdeplot(corr_cgan_coefs, bw=0.1, shade=True,  color='red', label='CGAN')
    plt.title("Correlation between volume and volatility")
    plt.xlabel("Correlation")
    plt.ylabel("Density")
    plt.legend()
    file_name = f"corr_vol_volatility_join.pdf"
    dir_path = os.path.dirname(TRADES_path)
    file_path = os.path.join(dir_path, file_name)
    #set limit of x to 1 and -1
    plt.xlim(-1, 1)
    plt.savefig(file_path)
    plt.close()
    '''
if __name__ == '__main__':
    main()