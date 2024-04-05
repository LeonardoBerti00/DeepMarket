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
        if isinstance(x, pd.DataFrame):
            x = x.values
        x = torch.from_numpy(x).float()
        x = self.pca.fit_transform(x)
        return x
    
def preprocess_data(df):
    df = df[['PRICE', 'SIZE', 'ask_price_1', 'ask_size_1', 'bid_price_1', 'bid_size_1', 'MID_PRICE', 'ORDER_VOLUME_IMBALANCE', 'VWAP', 'SPREAD']]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    # Standardization
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df

def plot_data(pca2d, pca2d_, generated_path):
    plt.figure(figsize=(10, 8))

    # Use seaborn's kdeplot for both real and generated data to visualize density
    sns.kdeplot(x=pca2d[:, 0], y=pca2d[:, 1], cmap="Reds", shade=True, bw_adjust=.5, label='real', alpha=0.8)
    sns.kdeplot(x=pca2d_[:, 0], y=pca2d_[:, 1], cmap="Blues", shade=True, bw_adjust=.5, label='generated', alpha=0.8)

    # Add legend and title
    plt.legend()
    plt.title('PCA2D Density Plot')

    file_name = "PCA_density_plot.pdf"
    if not os.path.exists(generated_path):
        os.makedirs(generated_path)
    generated_path = os.path.dirname(generated_path)
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)

    # Show the plot
    plt.show()

def main(real_path, generated_path):
    df_real = pd.read_csv(real_path, header=0)
    df_gen = pd.read_csv(generated_path, header=0)

    df_real = preprocess_data(df_real)
    df_gen = preprocess_data(df_gen)

    real_pca2d = PCA2D(n_components=2).forward(df_real)
    gen_pca2d = PCA2D(n_components=2).forward(df_gen)

    plot_data(real_pca2d, gen_pca2d, generated_path)

if __name__ == '__main__':
    main()
