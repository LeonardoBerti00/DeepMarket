import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main(real_path, cdt_path, iabs_path, cgan_path):
    def load_and_compute_correlation(file_path, window=30, lag=1):
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
        df['rolling_corr'] = df['log_return'].rolling(window=window).corr(df['log_return'].shift(lag))
        return df['rolling_corr'].dropna()

    correlation_real = load_and_compute_correlation(real_path)
    correlation_cdt = load_and_compute_correlation(cdt_path)
    correlation_iabs = load_and_compute_correlation(iabs_path)
    correlation_cgan = load_and_compute_correlation(cgan_path)
    
    
    sns.set(style="whitegrid")

    sns.kdeplot(correlation_real, shade=True, color="blue", label='Real')
    sns.kdeplot(correlation_iabs, shade=True, color="green", label='IABS')
    sns.kdeplot(correlation_cdt, shade=True, color="red", label='CDT')
    sns.kdeplot(correlation_cgan, shade=True, color="purple", label='CGAN')

    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Autocorrelation Log Returns Distribution')

    plt.legend()
    file_name = f"corr_coef_join.pdf"
    dir_path = os.path.dirname(cdt_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()
    #plt.show()

if __name__ == '__main__':
    main()