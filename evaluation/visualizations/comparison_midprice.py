from matplotlib import dates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def main(real_path, generated_path):
    df1 = pd.read_csv(real_path)
    df2 = pd.read_csv(generated_path)
    
    df1 = df1.query("ask_price_1 < 9999999")
    df1 = df1.query("bid_price_1 < 9999999")
    df1 = df1.query("ask_price_1 > -9999999")
    df1 = df1.query("bid_price_1 > -9999999")

    df2 = df2.query("ask_price_1 < 9999999")
    df2 = df2.query("bid_price_1 < 9999999")
    df2 = df2.query("ask_price_1 > -9999999")
    df2 = df2.query("bid_price_1 > -9999999")
    
    df1.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)
    df2.rename(columns={'Unnamed: 0': 'TIME'}, inplace=True)

    time1 = pd.to_datetime(df1['TIME'])
    mid_price1 = df1['MID_PRICE']

    time2 = pd.to_datetime(df2['TIME'])
    mid_price2 = df2['MID_PRICE']

    plt.figure(dpi=300,figsize=(10, 6))
    # compute the mean mid price every second for the real data
    #mid_price1 = mid_price1.groupby([time1.dt.hour, time1.dt.minute, time1.dt.second]).mean()
    #time1 = time1.groupby([time1.dt.hour, time1.dt.minute, time1.dt.second]).first()
    # compute the mean mid price every second for the generated data
    #mid_price2 = mid_price2.groupby([time2.dt.hour, time2.dt.minute, time2.dt.second]).mean()
    #time2 = time2.groupby([time2.dt.hour, time2.dt.minute, time2.dt.second]).first()
    # extract only time from time1 which is a datetime object
    '''
    time1 = time1.dt.strftime('%d-%m-%Y %H:%M:%S')
    time2 = time2.dt.strftime('%d-%m-%Y %H:%M:%S')
    time1 = pd.to_datetime(time1)
    time2 = pd.to_datetime(time2)
    time1 = time1.dt.time
    time2 = time2.dt.time
    # convert time1 to datetime object
    time1 = pd.to_datetime(time1, format='%H:%M:%S.%f')
    time1 = pd.to_datetime(time1, format='%H:%M') 
    # convert time2 to datetime object
    time2 = pd.to_datetime(time2, format='%H:%M:%S.%f')
    time2 = pd.to_datetime(time2, format='%H:%M')
    # convert time1 to matplotlib date format
    time1 = dates.date2num(time1)
    # convert time2 to matplotlib date format
    time2 = dates.date2num(time2)
    '''
    if "IABS" in generated_path:
        label = "IABS"
    elif "TRADES" in generated_path:
        label = "TRADES"
    elif "GAN" in generated_path:
        label = "CGAN"
    else:
        label = "TRADES"
    plt.plot(time1, mid_price1, label='Real', color='blue')
    #get the min and max of the mid price
    #xmin = min(mid_price1.min(), mid_price2.min())-100
    #xmax = max(mid_price1.max(), mid_price2.max())+100
    #plt.xlim([xmin, xmax])
    plt.plot(time2, mid_price2, label=label, color='red')
    #plt.xlim([xmin, xmax])
    # Formatting the x-axis with hour:minute format
    time_format = dates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(time_format)
    plt.xlabel('Trading Period')
    plt.ylabel('Mid Price')
    plt.title('Mid Price comparison')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_name = "comparison_midprice.pdf"
    generated_path = os.path.dirname(generated_path)
    file_path = os.path.join(generated_path, file_name)
    plt.savefig(file_path)
    #plt.show()
    plt.close()


if __name__ == '__main__':
    main()