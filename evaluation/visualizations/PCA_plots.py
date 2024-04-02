import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
import os

class PCA2D(torch.nn.Module):
    def __init__(self, n_components=2):
        super(PCA2D, self).__init__()
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def forward(self, x):
        x = x.values
        x = torch.from_numpy(x)
        x = self.pca.fit_transform(x)
        return x
    
def preprocess_data(df):

    df = df[['PRICE', 'SIZE', 'ask_price_1', 'ask_size_1', 'bid_price_1', 'bid_size_1', 'MID_PRICE', 'ORDER_VOLUME_IMBALANCE', 'VWAP', 'SPREAD']]
    #drop the row with inf and nan values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    # Standardization on price and size
    df['PRICE'] = (df['PRICE'] - df['PRICE'].mean())/df['PRICE'].std()
    df['SIZE'] = (df['SIZE'] - df['SIZE'].mean())/df['SIZE'].std()
    df['ask_price_1'] = (df['ask_price_1'] - df['ask_price_1'].mean())/df['ask_price_1'].std()
    df['ask_size_1'] = (df['ask_size_1'] - df['ask_size_1'].mean())/df['ask_size_1'].std()
    df['bid_price_1'] = (df['bid_price_1'] - df['bid_price_1'].mean())/df['bid_price_1'].std()
    df['bid_size_1'] = (df['bid_size_1'] - df['bid_size_1'].mean())/df['bid_size_1'].std()
    df['MID_PRICE'] = (df['MID_PRICE'] - df['MID_PRICE'].mean())/df['MID_PRICE'].std()
    df['ORDER_VOLUME_IMBALANCE'] = (df['ORDER_VOLUME_IMBALANCE'] - df['ORDER_VOLUME_IMBALANCE'].mean())/df['ORDER_VOLUME_IMBALANCE'].std()
    df['VWAP'] = (df['VWAP'] - df['VWAP'].mean())/df['VWAP'].std()
    df['SPREAD'] = (df['SPREAD'] - df['SPREAD'].mean())/df['SPREAD'].std()
    return df

def plot_data(pca2d, pca2d_, generated_path):
    # Plot pca2d in red
    plt.scatter(pca2d[:, 0], pca2d[:, 1], color='red', label='real')

    # Plot pca2d_ in blue
    plt.scatter(pca2d_[:, 0], pca2d_[:, 1], color='blue', label='generated')

    # Limit x and y axes
    #compute the limit depending on max and min of the data
    x_min = min(np.min(pca2d[:, 0]), np.min(pca2d_[:, 0]))
    x_max = max(np.max(pca2d[:, 0]), np.max(pca2d_[:, 0]))
    y_min = min(np.min(pca2d[:, 1]), np.min(pca2d_[:, 1]))
    y_max = max(np.max(pca2d[:, 1]), np.max(pca2d_[:, 1]))
    plt.xlim(x_min-1, x_max+1)
    plt.ylim(y_min-1, y_max+1)

    # Add legend and title
    plt.legend()
    file_name = "PCA_plot.png"
    generated_path = os.path.dirname(generated_path)
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)
    plt.title('PCA2D Plot')

    # Show the plot
    plt.show()



def main(real_path, generated_path):
    df_real = pd.read_csv(real_path,header=0)
    df_gen = pd.read_csv(generated_path,header=0)

    df_real = preprocess_data(df_real)
    df_gen = preprocess_data(df_gen)

    real_pca2d = PCA2D(n_components=2).forward(df_real)
    gen_pca2d = PCA2D(n_components=2).forward(df_gen)

    plot_data(real_pca2d, gen_pca2d, generated_path)


if __name__ == '__main__':
    main()


    


