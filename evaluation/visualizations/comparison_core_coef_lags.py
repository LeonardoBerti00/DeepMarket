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
    df = df.groupby('minute')['PRICE'].first().reset_index()
    df['log_return'] = np.log(df['PRICE'] / df['PRICE'].shift(1))
    df.dropna(inplace=True)
    return df['log_return']

def load_and_compute_volatility(df, i):
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df['minute'] = df['time'].dt.floor('T')
    df['second'] = df['time'].dt.second
    df = df.groupby(['minute', 'second'])['PRICE'].first().reset_index()
    df['log_return'] = np.log(df['PRICE'] / df['PRICE'].shift(i))
    #take the indexes of the nan values
    std_dev = df['log_return'].rolling(window=100).std().reset_index(drop=True)
    nan_indexes = std_dev[std_dev.isna()].index
    return std_dev, nan_indexes

def load_and_compute_volume(df, i):
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df['minute'] = df['time'].dt.floor('T')
    df['second'] = df['time'].dt.second
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
    df = df.groupby(['minute', 'second'])['PRICE'].first().reset_index()
    df['log_return'] = np.log(df['PRICE'] / df['PRICE'].shift(i))
    returns = df['log_return'].rolling(window=100).sum().reset_index(drop=True)
    nan_indexes = returns[returns.isna()].index
    return returns, nan_indexes

def compute_correlation_by_lag(log_returns, max_lag):
    correlations = []
    for lag in range(1, max_lag + 1, 2):
        corr = log_returns.corr(log_returns.shift(lag))
        correlations.append(corr)
    return correlations

def main(real_path, generated_path):

    log_returns_real = load_and_compute_log_returns(real_path)
    log_returns_generated = load_and_compute_log_returns(generated_path) 

    correlations_real = compute_correlation_by_lag(log_returns_real, 30)
    correlations_generated = compute_correlation_by_lag(log_returns_generated, 30)

    plt.plot(range(1, 31, 2), correlations_real, marker='o', linestyle='-', label='Real')
    plt.plot(range(1, 31, 2), correlations_generated, marker='o', linestyle='-', label='Generated')

    plt.xlabel('Lag')
    plt.ylabel('Correlation Coefficient')
    plt.title('Log Returns Autocorrelation')
    plt.legend()
    file_name = "corr_coef_lag.pdf"
    dir_path = os.path.dirname(generated_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()
    corr_real_coefs = []
    corr_generated_coefs = []
    df_generated = pd.read_csv(generated_path)
    df_real = pd.read_csv(real_path)
    for i in range(1, 7200, 10):
        volatility_real, nan_indexes_volat_real = load_and_compute_volatility(df_real, i)
        volatility_generated, nan_indexes_volat_gen = load_and_compute_volatility(df_generated, i)
        volume_real, nan_indexes_vol_real = load_and_compute_volume(df_real, i)
        volume_generated, nan_indexes_vol_gen = load_and_compute_volume(df_generated, i)
        #drop the first value from volume

        nan_indexes_real = np.union1d(nan_indexes_vol_real, nan_indexes_volat_real)
        nan_indexes_gen = np.union1d(nan_indexes_volat_gen, nan_indexes_vol_gen)
        
        volume_real = volume_real.drop(nan_indexes_real)  
        volume_generated = volume_generated.drop(nan_indexes_gen)
        volatility_real = volatility_real.drop(nan_indexes_real)
        volatility_generated = volatility_generated.drop(nan_indexes_gen)
        
        corr_real_coefs.append(np.corrcoef(volume_real.values, volatility_real.values)[0, 1])
        corr_generated_coefs.append(np.corrcoef(volume_generated.values, volatility_generated.values)[0, 1])
    
    #print(corr_real_coefs)
    #print(corr_generated_coefs)
    sns.kdeplot(corr_generated_coefs, bw=0.5, color='blue', label='Generated')

    sns.kdeplot(corr_real_coefs, bw=0.5, color='orange', label='Real')
    plt.title("Correlation between volume and volatility")
    plt.xlabel("Correlation")
    plt.ylabel("Density")
    plt.legend()
    file_name = "corr_vol_volatility.pdf"
    dir_path = os.path.dirname(generated_path)
    file_path = os.path.join(dir_path, file_name)
    #set limit of x to 1 and -1
    plt.xlim(-1, 1)
    plt.savefig(file_path)
    plt.close()
    corr_real_coefs = []
    corr_generated_coefs = []
    #PLOT RETURNS/VOLATILITY CORRELATION
    for i in range(1, 7200, 10):
        volatility_real, nan_indexes_volat_real = load_and_compute_volatility(df_real, i)
        volatility_generated, nan_indexes_volat_gen = load_and_compute_volatility(df_generated, i)
        returns_real, nan_indexes_ret_real = load_and_compute_returns(df_real, i)
        returns_generated, nan_indexes_ret_gen = load_and_compute_returns(df_generated, i)
        #drop the first value from volume

        nan_indexes_real = np.union1d(nan_indexes_ret_real, nan_indexes_volat_real)
        nan_indexes_gen = np.union1d(nan_indexes_ret_gen, nan_indexes_volat_gen)
        
        returns_real = returns_real.drop(nan_indexes_real)  
        returns_generated = returns_generated.drop(nan_indexes_gen)
        volatility_real = volatility_real.drop(nan_indexes_real)
        volatility_generated = volatility_generated.drop(nan_indexes_gen)
        
        corr_real_coefs.append(np.corrcoef(returns_real.values, volatility_real.values)[0, 1])
        corr_generated_coefs.append(np.corrcoef(returns_generated.values, volatility_generated.values)[0, 1])
    
    #print(corr_real_coefs)
    #print(corr_generated_coefs)
    sns.kdeplot(corr_generated_coefs, bw=0.5, color='blue', label='Generated')

    sns.kdeplot(corr_real_coefs, bw=0.5, color='orange', label='Real')
    plt.title("Correlation between returns and volatility")
    plt.xlabel("Correlation")
    plt.ylabel("Density")
    plt.legend()
    file_name = "corr_returns_vol.pdf"
    dir_path = os.path.dirname(generated_path)
    file_path = os.path.join(dir_path, file_name)
    #set limit of x to 1 and -1
    plt.xlim(-1, 1)
    plt.savefig(file_path)
    plt.close()

if __name__ == '__main__':
    main()