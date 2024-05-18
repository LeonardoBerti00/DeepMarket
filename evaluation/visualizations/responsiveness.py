from matplotlib import dates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


def plot_avg_diff_and_std(days_paths_wo_exp, days_paths_w_exp):
    assert len(days_paths_wo_exp) == len(days_paths_w_exp)
    diffs = []

    for path_wo_exp, path_w_exp in zip(days_paths_wo_exp, days_paths_w_exp):
        df_wo_exp = pd.read_csv(path_wo_exp)
        df_w_exp = pd.read_csv(path_w_exp)

        df_wo_exp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
        df_wo_exp['time'] = pd.to_datetime(df_wo_exp['time'])
        df_wo_exp['minute'] = df_wo_exp['time'].dt.floor('T')
        df_wo_exp['second'] = df_wo_exp['time'].dt.second
        #compute the average mid price for each second
        df_wo_exp = df_wo_exp.query("ask_price_1 < 9999999")
        df_wo_exp = df_wo_exp.query("bid_price_1 < 9999999")
        df_wo_exp = df_wo_exp.query("ask_price_1 > -9999999")
        df_wo_exp = df_wo_exp.query("bid_price_1 > -9999999")
        df_wo_exp = df_wo_exp.groupby(['minute'])['MID_PRICE'].mean().reset_index()        
        
        df_w_exp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
        df_w_exp['time'] = pd.to_datetime(df_w_exp['time'])
        df_w_exp['minute'] = df_w_exp['time'].dt.floor('T')
        df_w_exp['second'] = df_w_exp['time'].dt.second
        df_w_exp = df_w_exp.query("ask_price_1 < 9999999")
        df_w_exp = df_w_exp.query("bid_price_1 < 9999999")
        df_w_exp = df_w_exp.query("ask_price_1 > -9999999")
        df_w_exp = df_w_exp.query("bid_price_1 > -9999999")
        df_w_exp = df_w_exp.groupby(['minute'])['MID_PRICE'].mean().reset_index()
        
        # Ensure the dataframes are of the same length
        min_len = min(len(df_wo_exp), len(df_w_exp))
        df_wo_exp = df_wo_exp.iloc[:min_len]
        df_w_exp = df_w_exp.iloc[:min_len]

        # Compute the difference in 'MID_PRICE' and append to the list
        diff = df_w_exp['MID_PRICE'] - df_wo_exp['MID_PRICE']
        #window_size = 100
        #smooth_diff = diff.rolling(window_size).mean()
        diffs.append(diff.values)
        #convert minute and second in time
        time_datetime = pd.to_datetime(df_w_exp['minute'])
        time = pd.to_datetime(df_w_exp['minute'])
        #take only the minutes from time
        
    diffs = np.stack([diffs[0], diffs[1]], axis=0)
    # Compute the average and standard deviation of the differences
    avg_diff = np.mean(diffs, axis=0)
    std_diff = np.nan_to_num(np.std(diffs, axis=0))
    print(std_diff)
    plt.figure(dpi=300, figsize=(6, 4))
    #plt set y lim
    plt.ylim(-0.1, 0.1)
    # Plot the smoothed data
    plt.plot(time, avg_diff, color='blue')
    # Plot the average difference with error bars representing the standard deviation
    
    plt.fill_between(time, avg_diff - std_diff, avg_diff+std_diff, alpha=0.2)
    time_format = dates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(time_format)
    plt.xlabel('Index')
    plt.ylabel('Average Difference in Mid Price')
    plt.title('Average Difference in Mid Price with Standard Deviation')
    plt.grid(True)
    start_time = time[15] 
    end_time = time[60] 
    plt.axvspan(start_time, end_time, color='gray', alpha=0.1)
    #plt.show()
    file_name = "responsiveness_29.pdf"
    dir_path = os.path.dirname(days_paths_wo_exp[-1])
    parent_dir = os.path.dirname(dir_path)
    file_path = os.path.join(parent_dir, file_name)
    plt.savefig(file_path)
    plt.close()
