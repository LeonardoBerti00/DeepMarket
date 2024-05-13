# take the time; compute the difference between the current time and the previous time; and plot the distribution of the interarrival times.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main(real_path, generated_path):
    df1 = pd.read_csv(generated_path)
    df2 = pd.read_csv(real_path)
    
    df1.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)
    df2.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)

    def time_to_seconds(time_str):
        time = pd.to_datetime(time_str)
        return (time - time.dt.floor('d')).dt.total_seconds()

    df1['seconds'] = time_to_seconds(df1['TIME'])
    df2['seconds'] = time_to_seconds(df2['TIME'])

    df1['inter_arrival'] = df1['seconds'].diff().dropna()
    df2['inter_arrival'] = df2['seconds'].diff().dropna()

    df1 = df1[df1['inter_arrival'] > 0]
    df2 = df2[df2['inter_arrival'] > 0]

    min_bin = min(df1['inter_arrival'].min(), df2['inter_arrival'].min())
    max_bin = max(df1['inter_arrival'].max(), df2['inter_arrival'].max())
    bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 50)

    if "IABS" in generated_path:
        label = "IABS"
    elif "CDT" in generated_path:
        label = "CDT"
    elif "GAN" in generated_path:
        label = "CGAN"
    else:
        label = "CDT"
         
    plt.figure(dpi=300,figsize=(10, 5))

    plt.hist(df1['inter_arrival'], bins=bins, alpha=0.5, color='red', label=label)
    plt.hist(df2['inter_arrival'], bins=bins, alpha=0.5, color='blue', label='Real')

    plt.axvline(df1['inter_arrival'].mean(), color='red', linestyle='dashed', linewidth=2)
    plt.axvline(df1['inter_arrival'].median(), color='red', linestyle='dotted', linewidth=2)
    plt.axvline(df2['inter_arrival'].mean(), color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(df2['inter_arrival'].median(), color='blue', linestyle='dotted', linewidth=2)

    plt.xscale('log')

    plt.legend()
    plt.title('Inter-arrival Times on a Log-x Scale')
    plt.xlabel('Inter-arrival Time (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    file_name = "interarrival_time_plot.pdf"
    generated_path = os.path.dirname(generated_path)
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)
    plt.close()



if __name__ == '__main__':
    main()