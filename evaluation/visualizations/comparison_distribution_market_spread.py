import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.dates as mdates
import os


def ci(row, n, alpha):
    mean = row['SPREAD_mean']
    std = row['SPREAD_std']

    margin = st.t.interval(1-alpha, n-1, mean, std/np.sqrt(n))

    return pd.Series(margin, index=['LOWER', 'UPPER'])

def process_df(df):
    df = df.query("ask_price_1 < 9999999")
    df = df.query("bid_price_1 < 9999999")
    df = df.query("ask_price_1 > -9999999")
    df = df.query("bid_price_1 > -9999999")
    # rename 'Unnamed: 0' con TIME
    df.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)
        
    df_ = df[["TIME", "SPREAD"]]

    df_['TIME'] = pd.to_datetime(df_['TIME'])
    df_['TIME'] = df_['TIME'].dt.strftime('%d-%m-%Y %H:%M:%S')
    dividend = df.shape[0] // 100
    df_grouped = df.groupby(df.index // dividend).agg({'TIME': 'first', 'SPREAD': ['mean','std']})

    df_grouped.columns = ['TIME', 'SPREAD_mean', 'SPREAD_std']

    alpha = 0.05

    n = len(df_grouped)

    df_grouped[['LOWER', 'UPPER']] = df_grouped.apply(ci, args=(n, alpha), axis=1)


    # create df_f with only not NaN values
    df_f = df_grouped.dropna()

    df_f['TIME'] = pd.to_datetime(df_f['TIME'])

    df_f['TIME'] = df_f['TIME'].dt.time

    df_f['TIME'] = pd.to_datetime(df_f['TIME'], format='%H:%M:%S.%f')

    df_f['TIME'] = pd.to_datetime(df_f['TIME'], format='%H:%M')

    df_f['TIME'] = mdates.date2num(df_f['TIME'])

    
    return df_f

def main(real_path=None, TRADES_path=None, iabs_path=None, cgan_path=None):
    
    df_real = pd.read_csv(real_path, header=0)
    df_TRADES = pd.read_csv(TRADES_path, header=0)
    df_iabs = pd.read_csv(iabs_path, header=0)
    df_cgan = pd.read_csv(cgan_path, header=0)
    
    df_real = process_df(df_real)
    df_TRADES = process_df(df_TRADES)
    df_iabs = process_df(df_iabs)
    df_cgan = process_df(df_cgan)

    plt.plot(df_real['TIME'], df_real['SPREAD_mean'], label='Real', color='blue')#, marker='o', linestyle='', markersize=3)
    plt.plot(df_TRADES['TIME'], df_TRADES['SPREAD_mean'], label='TRADES', color='red')#, marker='o', linestyle='', markersize=3)
    plt.plot(df_iabs['TIME'], df_iabs['SPREAD_mean'], label='IABS', color='orange')#, marker='o', linestyle='', markersize=3)
    plt.plot(df_cgan['TIME'], df_cgan['SPREAD_mean'], label='CGAN', color='green')#, marker='o', linestyle='', markersize=3)

    plt.xlabel('Time')
    plt.ylabel('Spread')
    
    plt.title('Market spread')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.legend()
    file_name = "market_spread.pdf"
    dir_path = os.path.dirname(TRADES_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()
    
    

if __name__ == '__main__':
    main()