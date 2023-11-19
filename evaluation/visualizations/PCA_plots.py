import torch
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

class PCA2D(torch.nn.Module):
    def __init__(self, n_components=2):
        super(PCA2D, self).__init__()
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def forward(self, x):
        x = torch.Tensor(x)
        x = self.pca.fit_transform(x)
        return x
    

# open a file csv as pandas dataframe
df = pd.read_csv(r'C:\Users\marco\OneDrive\Documenti\afc\afc_project\Diffusion-Models-for-Time-Series\data\TSLA\TSLA_2015-01-02_2015-01-30\TSLA_2015-01-05_34200000_57600000_message_10.csv', header=None) # insert real data
df_ = pd.read_csv(r'C:\Users\marco\OneDrive\Documenti\afc\afc_project\Diffusion-Models-for-Time-Series\data\TSLA\TSLA_2015-01-02_2015-01-30\TSLA_2015-01-06_34200000_57600000_message_10.csv', header=None) # insert synthetic data

def preprocess_data(df):
    # Rename the columns into time, event_type, size, price, direction
    df.columns = ['time', 'event_type', 'size', 'price', 'direction']

    # Divide the column time by 100000
    df['time'] = df['time']/100000

    # One hot encoding for the feature event_type
    df = pd.get_dummies(df, columns=['event_type'])

    # Standardization on price and size
    df['price'] = (df['price'] - df['price'].mean())/df['price'].std()
    df['size'] = (df['size'] - df['size'].mean())/df['size'].std()

    # One hot encoding for the feature direction
    df = pd.get_dummies(df, columns=['direction'])

    return df

import matplotlib.pyplot as plt

def plot_data(pca2d, pca2d_):
    # Plot pca2d in red
    plt.scatter(pca2d[:, 0], pca2d[:, 1], color='red', label='pca2d')

    # Plot pca2d_ in blue
    plt.scatter(pca2d_[:, 0], pca2d_[:, 1], color='blue', label='pca2d_')

    # Limit x and y axes
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # Add legend and title
    plt.legend()
    plt.title('PCA2D Plot')

    # Show the plot
    plt.show()

df = preprocess_data(df)
df_ = preprocess_data(df_)

pca2d = PCA2D(n_components=2).forward(df)
pca2d_ = PCA2D(n_components=2).forward(df_)

plot_data(pca2d, pca2d_)


    


