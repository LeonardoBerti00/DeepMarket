# take the time; compute the difference between the current time and the previous time; and plot the distribution of the interarrival times.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main(real_path, TRADES_path, iabs_path, cgan_path):
    # Define a function to convert TIME to seconds
    def time_to_seconds(time_str):
        time = pd.to_datetime(time_str)
        return (time - time.dt.floor('d')).dt.total_seconds()

    # Read the CSV files into DataFrames with labels and colors
    data_list = [
        {'df': pd.read_csv(real_path), 'label': 'Real', 'color': 'blue'},
        {'df': pd.read_csv(TRADES_path), 'label': 'TRADES', 'color': 'red'},
        {'df': pd.read_csv(iabs_path), 'label': 'IABS', 'color': 'green'},
        {'df': pd.read_csv(cgan_path), 'label': 'CGAN', 'color': 'orange'}
    ]

    # Process each DataFrame
    for data in data_list:
        df = data['df']
        # Rename 'Unnamed: 0' to 'TIME' if necessary
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)
        # Convert 'TIME' to seconds
        df['seconds'] = time_to_seconds(df['TIME'])
        # Compute 'inter_arrival' times
        df['inter_arrival'] = df['seconds'].diff()
        # Remove non-positive 'inter_arrival' times
        df = df[df['inter_arrival'] > 0]
        # Update the DataFrame in the list
        data['df'] = df

    # Determine min and max bins across all DataFrames
    min_bin = min(data['df']['inter_arrival'].min() for data in data_list)
    max_bin = max(data['df']['inter_arrival'].max() for data in data_list)
    bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 50)

    # Plotting
    plt.figure(dpi=300, figsize=(10, 5))

    # Plot histograms and mean/median lines for each dataset
    for data in data_list:
        df = data['df']
        plt.hist(df['inter_arrival'], bins=bins, alpha=0.5, color=data['color'], label=data['label'])
        plt.axvline(df['inter_arrival'].mean(), color=data['color'], linestyle='dashed', linewidth=2)
        plt.axvline(df['inter_arrival'].median(), color=data['color'], linestyle='dotted', linewidth=2)

    plt.xscale('log')
    plt.legend()
    plt.title('Inter-arrival Times on a Log-x Scale')
    plt.xlabel('Inter-arrival Time (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    file_name = "interarrival_time_plot.pdf"
    dir_path = os.path.dirname(TRADES_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()



if __name__ == '__main__':
    main()