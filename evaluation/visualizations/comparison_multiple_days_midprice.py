import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

def main(days_paths):
    dfs = []
    scaler = StandardScaler()  

    for path in days_paths:
        df = pd.read_csv(path)
        df.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)

        mid_price = np.array(df['MID_PRICE']).reshape(-1, 1)
        mid_price_normalized = scaler.fit_transform(mid_price)
        df['MID_PRICE'] = mid_price_normalized.flatten()

        time = pd.to_datetime(df['TIME'])
        dfs.append((time, df['MID_PRICE']))

    plt.figure(figsize=(10, 6))
    for i, (time, mid_price) in enumerate(dfs):
        plt.plot(time, mid_price, label=f'Days {i+1}')

    plt.xlabel('Trading Period')
    plt.ylabel('Normalized MID_PRICE')
    plt.title('Normalized Mid Price Comparison')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_name = "comparison_multiple_days_midprice.png"
    file_path = os.path.join(days_paths[0], file_name)
    plt.savefig(file_path)
    plt.show()

if __name__ == '__main__':
    main()
