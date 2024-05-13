import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
import os 

    
class TSNE2D(torch.nn.Module):
    def __init__(self, n_components=2):
        super(TSNE2D, self).__init__()
        self.n_components = n_components
        self.tsne = TSNE(n_components=n_components, n_iter=300, perplexity=50)

    def forward(self, x):
        x = x.values
        x = torch.from_numpy(x)  
        x = self.tsne.fit_transform(x)
        return x
    
def preprocess_data(df):

    df = df[['PRICE', 'SIZE', 'ask_price_1', 'ask_size_1', 'bid_price_1', 'bid_size_1', 'MID_PRICE', 'ORDER_VOLUME_IMBALANCE', 'VWAP', 'SPREAD']]
    #drop the row with inf and nan values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.query("ask_price_1 < 9999999")
    df = df.query("bid_price_1 < 9999999")
    df = df.query("ask_price_1 > -9999999")
    df = df.query("bid_price_1 > -9999999")
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

def plot_data(tsna2d_real, tsna2d_gen, generated_path):
    
    if "IABS" in generated_path:
        label = "IABS"
    elif "CDT" in generated_path:
        label = "CDT"
    elif "GAN" in generated_path:
        label = "CGAN"
    else:
        label = "CDT"
    
    # Plot tsna2d in red
    plt.scatter(tsna2d_real[:, 0], tsna2d_real[:, 1], color='tab:red', label='Real', alpha=0.1, s=10)

    # Plot tsna2d_ in blue
    plt.scatter(tsna2d_gen[:, 0], tsna2d_gen[:, 1], color='tab:blue', label=label, alpha=0.1, s=10)

    x_min = min(np.min(tsna2d_real[:, 0])-20, np.min(tsna2d_gen[:, 0])-20)+1
    x_max = max(np.max(tsna2d_real[:, 0])+20, np.max(tsna2d_gen[:, 0])+20)
    y_min = min(np.min(tsna2d_real[:, 1])-20, np.min(tsna2d_gen[:, 1])-20)+1
    y_max = max(np.max(tsna2d_real[:, 1])+20, np.max(tsna2d_gen[:, 1])+20)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Add legend and title
    plt.legend()
    file_name = "TSNE_plot.pdf"
    generated_path = os.path.dirname(generated_path)
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)
    plt.title('t-SNE Plot')
    plt.close()


def main(real_path, generated_path):
    df_real = pd.read_csv(real_path,header=0)
    df_gen = pd.read_csv(generated_path,header=0)

    df_real = preprocess_data(df_real)
    df_gen = preprocess_data(df_gen)

    tsna2d_real = TSNE2D(n_components=2).forward(df_real)
    tsna2d_gen = TSNE2D(n_components=2).forward(df_gen)

    plot_data(tsna2d_real, tsna2d_gen, generated_path)


if __name__ == '__main__':
    main()


    


# TODO: modify the plot adding density areas