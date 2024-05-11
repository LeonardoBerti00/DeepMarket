import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main(real_path, generated_path):
    def load_and_compute_correlation(file_path, window=30, lag=1):
        df = pd.read_csv(file_path)
        df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = pd.to_datetime(df['time'])
        df['minute'] = df['time'].dt.floor('T')
        df = df.query("ask_price_1 < 9999999")
        df = df.query("bid_price_1 < 9999999")
        df = df.query("ask_price_1 > -9999999")
        df = df.query("bid_price_1 > -9999999")
        df = df.groupby('minute')['mid_price'].first().reset_index()
        df['log_return'] = np.log(df['mid_price'] / df['mid_price'].shift(1))
        df['rolling_corr'] = df['log_return'].rolling(window=window).corr(df['log_return'].shift(lag))
        return df['rolling_corr'].dropna()

    correlation_real = load_and_compute_correlation(real_path)
    correlation_generated = load_and_compute_correlation(generated_path)

    sns.set(style="whitegrid")

    sns.kdeplot(correlation_real, shade=True, color="blue", label='Real')

    sns.kdeplot(correlation_generated, shade=True, color="red", label='Generated')

    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Correlation Coefficient Log Returns Distribution')

    plt.legend()
    file_name = "corr_coef.pdf"
    generated_path = os.path.dirname(generated_path)
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)
    plt.close()
    #plt.show()

if __name__ == '__main__':
    main()