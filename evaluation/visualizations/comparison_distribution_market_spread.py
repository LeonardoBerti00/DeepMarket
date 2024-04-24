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

def main(real_path, generated_path, IS_REAL):
    if IS_REAL:
        df = pd.read_csv(real_path, header=0)
    else:
        df = pd.read_csv(generated_path, header=0)


    # rename 'Unnamed: 0' con TIME
    df.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)

    # new df with only SPREAD and TIME
    df_ = df[['TIME', 'SPREAD']]

    df_['TIME'] = pd.to_datetime(df_['TIME'])
    df_['TIME'] = df_['TIME'].dt.strftime('%d-%m-%Y %H:%M:%S')

    df_grouped = df.groupby(df.index // 500).agg({'TIME': 'first', 'SPREAD': ['mean','std']})

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

    plt.plot(df_f['TIME'], df_f['SPREAD_mean'], label='mean spread', color='green', marker='o', linestyle='', markersize=3)
    plt.fill_between(df_f['TIME'], df_f['LOWER'], df_f['UPPER'], color='green', alpha=0.3, label='ptc5-95 enveloppe spread')

    plt.xlabel('Time')
    plt.ylabel('Spread')
    if IS_REAL:
        plt.title('Real Data')
    else:
        plt.title('Generated Data')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.legend()
    file_name = "market_spread.pdf"
    if IS_REAL:
        dir_path = os.path.dirname(real_path)
    else:
        dir_path = os.path.dirname(generated_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()

if __name__ == '__main__':
    main()