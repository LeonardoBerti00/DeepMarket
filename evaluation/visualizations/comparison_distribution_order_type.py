import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def main(real_path, TRADES_path, iabs_path, cgan_path):
    df1 = pd.read_csv(real_path)
    df2 = pd.read_csv(TRADES_path)
    df3 = pd.read_csv(iabs_path)
    df4 = pd.read_csv(cgan_path)
    # select the column that contains the feature
    column = "TYPE"

    # compute the percentage of each value of the feature in the two dataframes
    percentage_real = df1[column].value_counts(normalize=True)
    percentage_TRADES = df2[column].value_counts(normalize=True)
    percentage_iabs = df3[column].value_counts(normalize=True)
    percentage_cgan = df4[column].value_counts(normalize=True)
    
    # join the two percentages in a single dataframe
    df_combined = pd.DataFrame({
        'Features values': percentage_TRADES.index,
        'Percentage_TRADES': percentage_TRADES.values,
        'Percentage_real': percentage_real.values,
        'Percentage_iabs': percentage_iabs.values,
        'Percentage_cgan': percentage_cgan.values
    })

    plt.figure(dpi=300,figsize=(10, 10))

    bar_width = 0.2

    ind = np.arange(len(df_combined['Features values']))

    plt.bar(ind, df_combined['Percentage_real'], width=bar_width, color="blue", label="Real")
    plt.bar(ind + bar_width, df_combined['Percentage_TRADES'], width=bar_width, color="red", label="TRADES")
    plt.bar(ind + 2 * bar_width, df_combined['Percentage_iabs'], width=bar_width, color="green", label="IABS")
    plt.bar(ind + 3 * bar_width, df_combined['Percentage_cgan'], width=bar_width, color="orange", label="CGAN")
        
    plt.title("Comparison distribution order type")
    plt.xlabel("Order Type")
    plt.ylabel("Percentage")
    plt.xticks(ind + bar_width, df_combined['Features values'])
    plt.legend()
    dir_path = os.path.dirname(TRADES_path)
    file_name = f"order_type_join.pdf"
    file_path = os.path.join(dir_path, file_name)
    #print(file_path)
    plt.savefig(file_path)
    #plt.show()
    plt.close()

if __name__ == '__main__':
    main()