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
        self.tsne = TSNE(n_components=n_components)

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

def plot_data(tsna2d, tsna2d_, generated_path):
    plt.figure(figsize=(10, 8))

    # Use seaborn's kdeplot for both real and generated data to visualize density
    sns.kdeplot(x=tsna2d[:, 0], y=tsna2d[:, 1], cmap="Reds", shade=True, bw_adjust=.5, label='real', alpha=0.8)
    sns.kdeplot(x=tsna2d_[:, 0], y=tsna2d_[:, 1], cmap="Blues", shade=True, bw_adjust=.5, label='generated', alpha=0.8)

    # Add legend and title
    plt.legend()
    plt.title('t-SNE 2D Density Plot')

    # Save the plot
    file_name = "tSNE_density_plot.pdf"
    if not os.path.exists(generated_path):
        os.makedirs(generated_path)
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)

    # Show the plot
    plt.show()



def main(real_path, generated_path):
    df_real = pd.read_csv(real_path,header=0)
    df_gen = pd.read_csv(generated_path,header=0)

    df_real = preprocess_data(df_real)
    df_gen = preprocess_data(df_gen)

    tsna2d = TSNE2D(n_components=2).forward(df_real)
    tsna2d_ = TSNE2D(n_components=2).forward(df_gen)

    plot_data(tsna2d, tsna2d_, generated_path)


if __name__ == '__main__':
    main()


    


# TODO: modify the plot adding density areas