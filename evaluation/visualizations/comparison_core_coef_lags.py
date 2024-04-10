import pandas as pd
import numpy as np
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

    def compute_correlation_by_lag(log_returns, max_lag):
        correlations = []
        for lag in range(1, max_lag + 1):
            corr = log_returns.corr(log_returns.shift(lag))
            correlations.append(corr)
        return correlations

    log_returns_real = load_and_compute_log_returns(real_path)
    log_returns_generated = load_and_compute_log_returns(generated_path) 

    correlations_real = compute_correlation_by_lag(log_returns_real, 10)
    correlations_generated = compute_correlation_by_lag(log_returns_generated, 10)

    plt.plot(range(1, 11), correlations_real, marker='o', linestyle='-', label='Real')
    plt.plot(range(1, 11), correlations_generated, marker='o', linestyle='-', label='Generated')

    plt.xlabel('Lag')
    plt.ylabel('Correlation Coefficient')
    plt.title('Volatility Clustering/Long Range Dependence')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()