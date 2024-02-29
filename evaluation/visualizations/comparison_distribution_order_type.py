import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import visualizations_constants as cst
import os


def main():
    df1 = pd.read_csv(cst.GENERATED_PATH)
    df2 = pd.read_csv(cst.REAL_PATH)

    # select the column that contains the feature
    column = "TYPE"

    # compute the percentage of each value of the feature in the two dataframes
    percentage_gen = df1[column].value_counts(normalize=True)
    percentage_real = df2[column].value_counts(normalize=True)

    # join the two percentages in a single dataframe
    df_combined = pd.DataFrame({
        'Features values': percentage_gen.index,
        'Percentage_gen': percentage_gen.values,
        'Percentage_real': percentage_real.values
    })

    plt.figure()

    bar_width = 0.35

    ind = np.arange(len(df_combined['Features values']))

    plt.bar(ind, df_combined['Percentage_gen'], width=bar_width, color="blue", label="generated")
    plt.bar(ind + bar_width, df_combined['Percentage_real'], width=bar_width, color="red", label="real")

    plt.title("Comparison distribution order type")
    plt.xlabel("Order Type")
    plt.ylabel("Percentage")
    plt.xticks(ind + bar_width / 2, df_combined['Features values'])
    plt.legend()

    file_name = "comparison_distribution_order_type.png"
    file_path = os.path.join(cst.folder_save_path, file_name)
    plt.savefig(file_path)

    plt.show()

if __name__ == '__main__':
    main()