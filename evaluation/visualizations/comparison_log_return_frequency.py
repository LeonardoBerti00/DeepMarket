import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main(real_path, TRADES_path, cgan_path):
    def load_and_compute_log_returns(file_path):
        df = pd.read_csv(file_path)
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
    
    
    log_returns_real = load_and_compute_log_returns(real_path)
    log_returns_TRADES = load_and_compute_log_returns(TRADES_path)
    log_returns_cgan = load_and_compute_log_returns(cgan_path)
    
    sns.set(style="whitegrid")

    sns.kdeplot(log_returns_real, shade=True, color="blue", label='Real', alpha=0.15)
    sns.kdeplot(log_returns_TRADES, shade=True, color="red", label='TRADES', alpha=0.15)
    sns.kdeplot(log_returns_cgan, shade=True, color="green", label='CGAN', alpha=0.15)

    plt.yscale('log')

    plt.xlabel('Log Returns')
    plt.ylabel('Log Frequency')
    plt.title('Minutely Log Returns Comparison')

    plt.legend()
    file_name = "log_return_join.pdf"
    dir_path = os.path.dirname(TRADES_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()
    #plt.show()


if __name__ == '__main__':
    main()