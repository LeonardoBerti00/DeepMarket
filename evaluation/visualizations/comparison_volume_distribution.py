import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.dates as mdates
import os

def ci(row, n, alpha):
    mean = row['asksize1_mean']
    std = row['asksize1_std']

    margin = st.t.interval(1-alpha, n-1, mean, std/np.sqrt(n))

    return pd.Series(margin, index=['LOWER', 'UPPER'])

def ci_(row, n, alpha):
    mean = row['bidsize1_mean']
    std = row['bidsize1_std']

    margin = st.t.interval(1-alpha, n-1, mean, std/np.sqrt(n))

    return pd.Series(margin, index=['LOWER_', 'UPPER_'])

def main(path):
    df = pd.read_csv(path, header=0)
    df = df.query("ask_price_1 < 9999999")
    df = df.query("bid_price_1 < 9999999")
    df = df.query("ask_price_1 > -9999999")
    df = df.query("bid_price_1 > -9999999")

    # rename 'Unnamed: 0' con TIME
    df.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)

    # new df with only SPREAD and TIME
    df_ = df[['TIME', 'ask_size_1', 'bid_size_1']]

    df_['TIME'] = pd.to_datetime(df_['TIME'])
    df_['TIME'] = df_['TIME'].dt.strftime('%d-%m-%Y %H:%M:%S')

    dividend = df.shape[0] // 100
    df_grouped = df.groupby(df.index // dividend).agg({'TIME': 'first', 'ask_size_1': ['mean','std'], 'bid_size_1': ['mean','std']})

    df_grouped.columns = ['TIME', 'asksize1_mean', 'asksize1_std', 'bidsize1_mean', 'bidsize1_std']

    alpha = 0.05

    n = len(df_grouped)

    df_grouped[['LOWER', 'UPPER']] = df_grouped.apply(ci, args=(n, alpha), axis=1)
    df_grouped[['LOWER_', 'UPPER_']] = df_grouped.apply(ci_, args=(n, alpha), axis=1)

    # create df_f with only not NaN values
    df_f = df_grouped.dropna()

    df_f['TIME'] = pd.to_datetime(df_f['TIME'])

    df_f['TIME'] = df_f['TIME'].dt.time

    df_f['TIME'] = pd.to_datetime(df_f['TIME'], format='%H:%M:%S.%f')

    df_f['TIME'] = pd.to_datetime(df_f['TIME'], format='%H:%M')

    df_f['TIME'] = mdates.date2num(df_f['TIME'])

    plt.plot(df_f['TIME'], df_f['asksize1_mean'], label='ask_size mean', color='green', marker='o', linestyle='', markersize=3)
    plt.fill_between(df_f['TIME'], df_f['LOWER'], df_f['UPPER'], color='green', alpha=0.3, label='ptc5-95 ask')

    plt.plot(df_f['TIME'], df_f['bidsize1_mean'], label='bid_size mean', color='red', marker='o', linestyle='', markersize=3)
    plt.fill_between(df_f['TIME'], df_f['LOWER_'], df_f['UPPER_'], color='red', alpha=0.3, label='ptc5-95 bid')

    plt.xlabel('Time')
    plt.ylabel('Volume at 1st level')
    
    
    
    if "replay" in path:
        plt.ylim(50, 600)
        title = "Volume Market Replay"
    if "IABS" in path:
        title = "Volume IABS simulation"
    elif "TRADES" in path:
        plt.ylim(50, 600)
        title = "Volume TRADES simulation"
    elif "GAN" in path:
        title = "Volume CGAN simulation"
    else:
        title = "Volume TRADES simulation"
        plt.ylim(50, 600)
    plt.title(title)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.legend()
    file_name = "comp_vol_distr.pdf"
    dir_path = os.path.dirname(path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()


if __name__ == '__main__':
    main()