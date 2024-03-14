import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def main(real_path, generated_path):
    df1 = pd.read_csv(generated_path)
    data1 = df1["ORDER_VOLUME_IMBALANCE"]

    df2 = pd.read_csv(real_path)
    data2 = df2["ORDER_VOLUME_IMBALANCE"]

    sns.kdeplot(data1, bw=0.5, color='blue', label='Generated')

    sns.kdeplot(data2, bw=0.5, color='orange', label='Real')

    plt.title("Volume")
    plt.xlabel("Shares (qta)")
    plt.ylabel("Density")

    plt.legend()
    file_name = "comparison_distribution_volume.png"
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)
    plt.show()

    df1 = pd.read_csv(generated_path)
    data1 = df1["PRICE"]

    df2 = pd.read_csv(real_path)
    data2 = df2["PRICE"]

    sns.kdeplot(data1, bw=0.5, color='blue', label='Generated')

    sns.kdeplot(data2, bw=0.5, color='orange', label='Real')

    plt.title("Price")
    plt.xlabel("DOLLAR ($)")
    plt.ylabel("Density")

    plt.legend()
    file_name = "comparison_distribution_price.png"
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)
    plt.show()


if __name__ == '__main__':
    main()