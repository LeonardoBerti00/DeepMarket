import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import visualizations_constants as cst
import seaborn as sns
import visualizations_constants as cst

def main():
    df1 = pd.read_csv(cst.REAL_PATH)
    df2 = pd.read_csv(cst.GENERATED_PATH)

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
    plt.show()


if __name__ == '__main__':
    main()