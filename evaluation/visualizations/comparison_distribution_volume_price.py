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

    plt.title("Order Volume Imbalance")
    plt.xlabel("Shares")
    plt.ylabel("Density")

    plt.legend()
    file_name = "order_volume_imbalance.pdf"
    dir_path = os.path.dirname(generated_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()


    data1 = df1["SIZE"]
    data2 = df2["SIZE"]

    sns.kdeplot(data1, color='blue', label='Generated')

    sns.kdeplot(data2, color='orange', label='Real')

    plt.title("Order Size")
    xmin = min(data1.min(), data2.min())-1000
    xmax = 3000
    plt.xlim([xmin, xmax])
    plt.xlabel("Shares")
    plt.ylabel("Density")

    plt.legend()
    file_name = "size.pdf"
    dir_path = os.path.dirname(generated_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()


    data1 = df1["VWAP"]
    data2 = df2["VWAP"]

    sns.kdeplot(data1, bw=0.5, color='blue', label='Generated')

    sns.kdeplot(data2, bw=0.5, color='orange', label='Real')

    plt.title("VWAP")
    plt.xlabel("Dollar")
    plt.ylabel("Density")

    plt.legend()
    file_name = "vwap.pdf"
    dir_path = os.path.dirname(generated_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()


    data1 = df1["PRICE"]
    data2 = df2["PRICE"]

    sns.kdeplot(data1, bw=0.5, color='blue', label='Generated')

    sns.kdeplot(data2, bw=0.5, color='orange', label='Real')

    plt.title("Price")
    plt.xlabel("Dollar")
    plt.ylabel("Density")

    plt.legend()
    file_name = "price.pdf"
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()

    data1 = df1["MID_PRICE"]
    data2 = df2["MID_PRICE"]

    sns.kdeplot(data1, bw=0.5, color='blue', label='Generated')

    sns.kdeplot(data2, bw=0.5, color='orange', label='Real')

    plt.title("Mid Price")
    plt.xlabel("Dollar")
    plt.ylabel("Density")

    plt.legend()
    file_name = "midprice.pdf"
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()


if __name__ == '__main__':
    main()