from matplotlib import dates
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
        df = df.query("ask_price_1 < 9999999")
        df = df.query("bid_price_1 < 9999999")
        df = df.query("ask_price_1 > -9999999")
        df = df.query("bid_price_1 > -9999999")

        df.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)
        time = pd.to_datetime(df['TIME'])
        dfs.append((time, df['MID_PRICE']))

    plt.figure(dpi=300,figsize=(10, 6))
    for i, (time, mid_price) in enumerate(dfs):
        if i == 0:
            plt.plot(time, mid_price, label='Real')
        else:
            plt.plot(time, mid_price)

    time_format = dates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(time_format)
    plt.xlabel('Trading Period')
    plt.ylabel('Normalized Mid Price')
    plt.title('Normalized Mid Price Multiple Seeds')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_name = "comparison_multiple_days_midprice_29.pdf"
    dir_path = os.path.dirname(days_paths[-1])
    parent_dir = os.path.dirname(dir_path)
    file_path = os.path.join(parent_dir, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()

if __name__ == '__main__':
    main()
