import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import constants as cst
import seaborn as sns

def main():
    df1 = pd.read_csv(cst.GENERATED_PATH)
    data1 = df1["ORDER_VOLUME_IMBALANCE"]

    df2 = pd.read_csv(cst.REAL_PATH)
    data2 = df2["ORDER_VOLUME_IMBALANCE"]

    sns.kdeplot(data1, bw=0.5, color='blue', label='Generated')

    sns.kdeplot(data2, bw=0.5, color='orange', label='Real')

    plt.title("Volume")
    plt.xlabel("Shares (qta)")
    plt.ylabel("Density")

    plt.legend()

    plt.show()

    df1 = pd.read_csv(cst.GENERATED_PATH)
    data1 = df1["PRICE"]

    df2 = pd.read_csv(cst.REAL_PATH)
    data2 = df2["PRICE"]

    sns.kdeplot(data1, bw=0.5, color='blue', label='Generated')

    sns.kdeplot(data2, bw=0.5, color='orange', label='Real')

    plt.title("Price")
    plt.xlabel("DOLLAR ($)")
    plt.ylabel("Density")

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()