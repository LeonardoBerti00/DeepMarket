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
    df['log_return'] = np.log(df['MID_PRICE'] / df['MID_PRICE'].shift(1))
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
    df['log_return'] = np.log(df['MID_PRICE'] / df['MID_PRICE'].shift(i))
    #take the indexes of the nan values
    std_dev = df['log_return'].rolling(window=100).std().reset_index(drop=True)
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
    df['volume_shift'] = df['SIZE'].rolling(window=i).sum().reset_index(drop=True)
    volume_sum = df['volume_shift'].rolling(window=100).sum().reset_index(drop=True)
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

def main(real_path, cdt_path, iabs_path, cgan_path):
    '''
    log_returns_real = load_and_compute_log_returns(real_path)
    log_returns_cdt = load_and_compute_log_returns(cdt_path) 
    log_returns_iabs = load_and_compute_log_returns(iabs_path)
    log_returns_cgan = load_and_compute_log_returns(cgan_path)

    correlations_real = compute_correlation_by_lag(log_returns_real, 30)
    correlations_cdt = compute_correlation_by_lag(log_returns_cdt, 30)
    correlations_iabs = compute_correlation_by_lag(log_returns_iabs, 30)
    correlations_cgan = compute_correlation_by_lag(log_returns_cgan, 30)
    
    plt.plot(range(1, 31, 2), correlations_real, marker='o', linestyle='-', label='Real')
    plt.plot(range(1, 31, 2), correlations_cdt, marker='o', linestyle='-', label='CDT')
    plt.plot(range(1, 31, 2), correlations_iabs, marker='o', linestyle='-', label='IABS')
    plt.plot(range(1, 31, 2), correlations_cgan, marker='o', linestyle='-', label='CGAN')

    plt.xlabel('Lag (minutes)')
    plt.ylabel('Correlation Coefficient')
    plt.title('Log Returns Autocorrelation')
    plt.legend()
    file_name = f"corr_coef_lag_join.pdf"
    dir_path = os.path.dirname(cdt_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()
    '''
    corr_iabs_coefs = []
    corr_real_coefs = []
    corr_cdt_coefs = []
    corr_cgan_coefs = []
    df_cdt = pd.read_csv(cdt_path)
    df_real = pd.read_csv(real_path)
    df_iabs = pd.read_csv(iabs_path)
    df_cgan = pd.read_csv(cgan_path)
    
    for i in range(60, 7200, 30):
        volatility_real, nan_indexes_volat_real = load_and_compute_volatility(df_real, i)
        volatility_cdt, nan_indexes_volat_cdt = load_and_compute_volatility(df_cdt, i)
        volatility_iabs, nan_indexes_volat_iabs = load_and_compute_volatility(df_iabs, i)
        volatility_cgan, nan_indexes_volat_cgan = load_and_compute_volatility(df_cgan, i)
        
        volume_iabs, nan_indexes_vol_iabs = load_and_compute_volume(df_iabs, i)
        volume_real, nan_indexes_vol_real = load_and_compute_volume(df_real, i)
        volume_cdt, nan_indexes_vol_cdt = load_and_compute_volume(df_cdt, i)
        volume_cgan, nan_indexes_vol_cgan = load_and_compute_volume(df_cgan, i)
        #drop the first value from volume

        nan_indexes_real = np.union1d(nan_indexes_vol_real, nan_indexes_volat_real)
        nan_indexes_cdt = np.union1d(nan_indexes_volat_cdt, nan_indexes_vol_cdt)
        nan_indexes_iabs = np.union1d(nan_indexes_vol_iabs, nan_indexes_volat_iabs)
        nan_indexes_cgan = np.union1d(nan_indexes_vol_cgan, nan_indexes_volat_cgan)
        
        volume_real = volume_real.drop(nan_indexes_real)  
        volume_cdt = volume_cdt.drop(nan_indexes_cdt)
        volume_iabs = volume_iabs.drop(nan_indexes_iabs)
        volume_cgan = volume_cgan.drop(nan_indexes_cgan)
        
        volatility_real = volatility_real.drop(nan_indexes_real)
        volatility_cdt = volatility_cdt.drop(nan_indexes_cdt)
        volatility_iabs = volatility_iabs.drop(nan_indexes_iabs)
        volatility_cgan = volatility_cgan.drop(nan_indexes_cgan)
        
        corr_real_coefs.append(np.corrcoef(volume_real.values, volatility_real.values)[0, 1])
        corr_cdt_coefs.append(np.corrcoef(volume_cdt.values, volatility_cdt.values)[0, 1])
        corr_iabs_coefs.append(np.corrcoef(volume_iabs.values, volatility_iabs.values)[0, 1])
        corr_cgan_coefs.append(np.corrcoef(volume_cgan.values, volatility_cgan.values)[0, 1])

    
    sns.kdeplot(corr_cdt_coefs, bw=0.1, shade=True,  color='blue', label="CDT")
    sns.kdeplot(corr_iabs_coefs, bw=0.1, shade=True,  color='green', label="IABS")
    sns.kdeplot(corr_real_coefs, bw=0.1, shade=True,  color='orange', label='Real')
    sns.kdeplot(corr_cgan_coefs, bw=0.1, shade=True,  color='red', label='CGAN')
    
    plt.title("Correlation between volume and volatility")
    plt.xlabel("Correlation")
    plt.ylabel("Density")
    plt.legend()
    file_name = f"corr_vol_volatility_join.pdf"
    dir_path = os.path.dirname(cdt_path)
    file_path = os.path.join(dir_path, file_name)
    #set limit of x to 1 and -1
    plt.xlim(-1, 1)
    plt.savefig(file_path)
    plt.close()
    '''
    corr_real_coefs = []
    corr_cdt_coefs = []
    corr_iabs_coefs = []
    corr_cgan_coefs = []
    #PLOT RETURNS/VOLATILITY CORRELATION
    for i in range(1, 7200, 10):
        volatility_real, nan_indexes_volat_real = load_and_compute_volatility(df_real, i)
        volatility_cdt, nan_indexes_volat_cdt = load_and_compute_volatility(df_cdt, i)
        volatility_iabs, nan_indexes_volat_iabs = load_and_compute_volatility(df_iabs, i)
        volatility_cgan, nan_indexes_volat_cgan = load_and_compute_volatility(df_cgan, i)
        
        returns_real, nan_indexes_ret_real = load_and_compute_returns(df_real, i)
        returns_cdt, nan_indexes_ret_cdt = load_and_compute_returns(df_cdt, i)
        returns_iabs, nan_indexes_ret_iabs = load_and_compute_returns(df_iabs, i)
        returns_cgan, nan_indexes_ret_cgan = load_and_compute_returns(df_cgan, i)
        #drop the first value from volume

        nan_indexes_real = np.union1d(nan_indexes_ret_real, nan_indexes_volat_real)
        nan_indexes_cdt = np.union1d(nan_indexes_ret_cdt, nan_indexes_volat_cdt)
        nan_indexes_iabs = np.union1d(nan_indexes_ret_iabs, nan_indexes_volat_iabs)
        nan_indexes_cgan = np.union1d(nan_indexes_ret_cgan, nan_indexes_volat_cgan)
        
        returns_real = returns_real.drop(nan_indexes_real)  
        returns_cdt = returns_cdt.drop(nan_indexes_cdt)
        returns_iabs = returns_iabs.drop(nan_indexes_iabs)
        returns_cgan = returns_cgan.drop(nan_indexes_cgan)
        volatility_real = volatility_real.drop(nan_indexes_real)
        volatility_cdt = volatility_cdt.drop(nan_indexes_cdt)
        volatility_iabs = volatility_iabs.drop(nan_indexes_iabs)
        volatility_cgan = volatility_cgan.drop(nan_indexes_cgan)
        
        corr_real_coefs.append(np.corrcoef(returns_real.values, volatility_real.values)[0, 1])
        corr_cdt_coefs.append(np.corrcoef(returns_cdt.values, volatility_cdt.values)[0, 1])
        corr_iabs_coefs.append(np.corrcoef(returns_iabs.values, volatility_iabs.values)[0, 1])
        corr_cgan_coefs.append(np.corrcoef(returns_cgan.values, volatility_cgan.values)[0, 1])
    
    #print(corr_real_coefs)
    #print(corr_generated_coefs)
    sns.kdeplot(corr_cdt_coefs, bw=0.1, shade=True, color='blue', label='CDT')
    sns.kdeplot(corr_iabs_coefs, bw=0.1, shade=True, color='green', label='IABS')
    sns.kdeplot(corr_real_coefs, bw=0.1, shade=True, color='orange', label='Real')
    sns.kdeplot(corr_cgan_coefs, bw=0.1, shade=True, color='red', label='CGAN')
    plt.title("Correlation between returns and volatility")
    plt.xlabel("Correlation")
    plt.ylabel("Density")
    plt.legend()
    file_name = f"corr_returns_vol_join.pdf"
    file_path = os.path.join(dir_path, file_name)
    #set limit of x to 1 and -1
    plt.xlim(-1, 1)
    plt.savefig(file_path)
    plt.close()
    '''
if __name__ == '__main__':
    main()