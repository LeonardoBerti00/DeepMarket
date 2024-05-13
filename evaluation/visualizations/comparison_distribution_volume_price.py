import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def main(real_path, cdt_path):
    df1 = pd.read_csv(cdt_path)
    df1 = df1.query("ask_price_1 < 9999999")
    df1 = df1.query("bid_price_1 < 9999999")
    df1 = df1.query("ask_price_1 > -9999999")
    df1 = df1.query("bid_price_1 > -9999999")
    data1 = df1["ORDER_VOLUME_IMBALANCE"]

    df2 = pd.read_csv(real_path)
    df2 = df2.query("ask_price_1 < 9999999")
    df2 = df2.query("bid_price_1 < 9999999")
    df2 = df2.query("ask_price_1 > -9999999")
    df2 = df2.query("bid_price_1 > -9999999")
    data2 = df2["ORDER_VOLUME_IMBALANCE"]
    '''
    df3 = pd.read_csv(iabs_path)
    df3 = df3.query("ask_price_1 < 9999999")
    df3 = df3.query("bid_price_1 < 9999999")
    df3 = df3.query("ask_price_1 > -9999999")
    df3 = df3.query("bid_price_1 > -9999999")
    data3 = df3["ORDER_VOLUME_IMBALANCE"]
    '''
    if "IABS" in cdt_path:
        label = "IABS"
    elif "CDT" in cdt_path:
        label = "CDT"
    elif "GAN" in cdt_path:
        label = "CGAN"
    else:
        label = "CDT"
        
    sns.kdeplot(data1, bw=0.1, color='blue', shade=True, label=label)
    sns.kdeplot(data2, bw=0.1, color='orange', shade=True, label='Real')
    #sns.kdeplot(data3, bw=0.1, color='green', shade=True, label='IABS')
    
    plt.title("Order Volume Imbalance")
    plt.xlabel("Shares")
    plt.ylabel("Density")

    plt.legend()
    file_name = "order_volume_imbalance.pdf"
    dir_path = os.path.dirname(cdt_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()


    data1 = df1["SIZE"]
    data2 = df2["SIZE"]
    #data3 = df3["SIZE"]

    sns.kdeplot(data1, color='blue', shade=True, label=label)
    sns.kdeplot(data2, color='orange', shade=True, label='Real')
    #sns.kdeplot(data3, color='green', shade=True, label='IABS')

    plt.title("Order Size")
    xmin = min(data1.min(), data2.min())-1000
    xmax = 3000
    plt.xlim([xmin, xmax])
    plt.xlabel("Shares")
    plt.ylabel("Density")

    plt.legend()
    file_name = "size.pdf"
    dir_path = os.path.dirname(cdt_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()


    data1 = df1["VWAP"]
    data2 = df2["VWAP"]
    #data3 = df3["VWAP"]

    sns.kdeplot(data1, bw=0.1, color='blue', shade=True, label=label)
    sns.kdeplot(data2, bw=0.1, color='orange', shade=True, label='Real')
    #sns.kdeplot(data3, bw=0.1, color='green', shade=True, label='IABS')

    plt.title("VWAP")
    plt.xlabel("Dollar")
    plt.ylabel("Density")

    plt.legend()
    file_name = "vwap.pdf"
    dir_path = os.path.dirname(cdt_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()


    data1 = df1["PRICE"]
    data2 = df2["PRICE"]
    #data3 = df3["PRICE"]

    sns.kdeplot(data1, bw=0.1, color='blue', shade=True,label=label)
    sns.kdeplot(data2, bw=0.1, color='orange',shade=True, label='Real')
    #sns.kdeplot(data3, bw=0.1, color='green', shade=True, label='IABS')

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
    #data3 = df3["MID_PRICE"]

    sns.kdeplot(data1, bw=0.1, color='blue', shade=True, label=label)
    sns.kdeplot(data2, bw=0.1, color='orange', shade=True, label='Real')
    #sns.kdeplot(data3, bw=0.1, color='green', shade=True, label='IABS')

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