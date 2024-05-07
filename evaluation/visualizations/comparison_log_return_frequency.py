import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main(real_path, generated_path):
    def load_and_compute_log_returns(file_path):
        df = pd.read_csv(file_path)
        df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])
        df['minute'] = df['time'].dt.floor('T')
        df = df.groupby('minute')['PRICE'].first().reset_index()
        df['log_return'] = np.log(df['PRICE'] / df['PRICE'].shift(1))
        df.dropna(inplace=True)
        return df['log_return']

    log_returns_real = load_and_compute_log_returns(real_path)
    log_returns_generated = load_and_compute_log_returns(generated_path)

    sns.set(style="whitegrid")

    sns.kdeplot(log_returns_real, shade=True, color="blue", label='Real')

    sns.kdeplot(log_returns_generated, shade=True, color="red", label='Generated')

    plt.yscale('log')

    plt.xlabel('Log Returns')
    plt.ylabel('Log Frequency')
    plt.title('Minutely Log Returns Comparison')

    plt.legend()
    file_name = "log_return.pdf"
    generated_path = os.path.dirname(generated_path)
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)
    plt.close()
    #plt.show()


if __name__ == '__main__':
    main()