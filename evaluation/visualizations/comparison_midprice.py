import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def main(real_path, generated_path):
    df1 = pd.read_csv(real_path)
    df2 = pd.read_csv(generated_path)

    df1.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)
    df2.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)

    time1 = pd.to_datetime(df1['TIME'])
    mid_price1 = df2['MID_PRICE']

    time2 = pd.to_datetime(df2['TIME'])
    mid_price2 = df2['MID_PRICE']

    plt.figure(figsize=(10, 6))
    plt.plot(time1, mid_price1, label='real data', color='blue')
    plt.plot(time2, mid_price2, label='generated data', color='red')
    plt.xlabel('Trading Period')
    plt.ylabel('MID_PRICE')
    plt.title('Mid Price comparison')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_name = "comparison_midprice.png"
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)
    plt.show()


if __name__ == '__main__':
    main()