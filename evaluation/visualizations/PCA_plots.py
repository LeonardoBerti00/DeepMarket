import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import os
import matplotlib.pyplot as plt


class pca(torch.nn.Module):
    def __init__(self, n_components=2):
        super(pca, self).__init__()
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def forward(self, x):
        x = x.values
        x = torch.from_numpy(x)
        pca = self.pca.fit_transform(x)
        pca = self.remove_outliers(pca)
        hull = ConvexHull(pca)
        return pca, hull
    
    def remove_outliers(self, points):
        # Ensure points is a 2D array
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input points array must be of shape (n, 2)")

        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = np.percentile(points, 5, axis=0)
        Q3 = np.percentile(points, 95, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter points that are within the lower and upper bounds
        filtered_points = points[
            (points[:, 0] >= lower_bound[0]) & (points[:, 0] <= upper_bound[0]) &
            (points[:, 1] >= lower_bound[1]) & (points[:, 1] <= upper_bound[1])
        ]
        return filtered_points

    
def preprocess_data(df):

    df = df[['PRICE', 'SIZE', 'ask_price_1', 'ask_size_1', 'bid_price_1', 'bid_size_1', 'MID_PRICE', 'ORDER_VOLUME_IMBALANCE', 'VWAP', 'SPREAD']]
    df = df.query("ask_price_1 < 9999999")
    df = df.query("bid_price_1 < 9999999")
    df = df.query("ask_price_1 > -9999999")
    df = df.query("bid_price_1 > -9999999")
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

def plot_data(pca, pca2, generated_path):
    if "IABS" in generated_path:
        label = "IABS"
    elif "CDT" in generated_path:
        label = "CDT"
    elif "GAN" in generated_path:
        label = "CGAN"
    else:
        label = "CDT"
    # Plot pca in red
    plt.scatter(pca[:, 0], pca[:, 1], color='tab:red', label='Real', alpha=0.1, s=10)

    # Plot pca2 in blue
    plt.scatter(pca2[:, 0], pca2[:, 1], color='tab:blue', label=label, alpha=0.1, s=10)

    # Limit x and y axes
    #compute the limit depending on max and min of the data
    x_min = min(np.min(pca[:, 0])-1, np.min(pca2[:, 0])-1)
    x_max = max(np.max(pca[:, 0])+1, np.max(pca2[:, 0])+1)
    y_min = min(np.min(pca[:, 1])-1, np.min(pca2[:, 1])-1)
    y_max = max(np.max(pca[:, 1])+1, np.max(pca2[:, 1])+1)
    plt.xlim(x_min-1, x_max+1)
    plt.ylim(y_min-1, y_max+1)

    # Add legend and title
    plt.legend()
    file_name = "PCA_plot.png"
    generated_path = os.path.dirname(generated_path)
    file_path = os.path.join(generated_path, file_name)
    plt.title(f'PCA 2D Plot for {label}')
    plt.savefig(file_path)
    # Show the plot
    #plt.show()
    plt.close()

def main(real_path, generated_path):
    df_real = pd.read_csv(real_path,header=0)
    df_gen = pd.read_csv(generated_path,header=0)

    df_real = preprocess_data(df_real)
    df_gen = preprocess_data(df_gen)

    real_pca, real_hull = pca(n_components=2).forward(df_real)
    gen_pca, gen_hull = pca(n_components=2).forward(df_gen)

    plot_data(real_pca, gen_pca, generated_path)
    real_polygon = Polygon(real_pca[real_hull.vertices])
    gen_polygon = Polygon(gen_pca[gen_hull.vertices])
    
    inters_area = real_polygon.intersection(gen_polygon).area
    real_area = real_polygon.area
    coverage_percentage = inters_area/real_area * 100
    print(f"Coverage percentage: {coverage_percentage}")


if __name__ == '__main__':
    main()


    


# TODO: modify the plot adding density areas