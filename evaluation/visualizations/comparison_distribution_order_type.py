import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def main(real_path, cdt_path, iabs_path):
    df1 = pd.read_csv(real_path)
    df2 = pd.read_csv(cdt_path)
    df3 = pd.read_csv(iabs_path)
    # select the column that contains the feature
    column = "TYPE"

    # compute the percentage of each value of the feature in the two dataframes
    percentage_real = df1[column].value_counts(normalize=True)
    percentage_cdt = df2[column].value_counts(normalize=True)
    percentage_iabs = df3[column].value_counts(normalize=True)
    
    # join the two percentages in a single dataframe
    df_combined = pd.DataFrame({
        'Features values': percentage_cdt.index,
        'Percentage_cdt': percentage_cdt.values,
        'Percentage_real': percentage_real.values,
        'Percentage_iabs': percentage_iabs.values
    })

    plt.figure(dpi=300,figsize=(10, 6))

    bar_width = 0.2

    ind = np.arange(len(df_combined['Features values']))

    plt.bar(ind, df_combined['Percentage_real'], width=bar_width, color="blue", label="Real")
    plt.bar(ind + bar_width, df_combined['Percentage_cdt'], width=bar_width, color="red", label="CDT")
    plt.bar(ind + 2 * bar_width, df_combined['Percentage_iabs'], width=bar_width, color="green", label="IABS")
    
    plt.title("Comparison distribution order type")
    plt.xlabel("Order Type")
    plt.ylabel("Percentage")
    plt.xticks(ind + bar_width, df_combined['Features values'])
    plt.legend()
    simulated_day = real_path.split('/')[-2].split('_')[3]
    dir_path = os.path.dirname(cdt_path)
    file_name = f"corr_type_join.pdf"
    file_path = os.path.join(dir_path, file_name)
    #print(file_path)
    plt.savefig(file_path)
    #plt.show()
    plt.close()

if __name__ == '__main__':
    main()