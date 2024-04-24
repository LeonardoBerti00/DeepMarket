import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def main(real_path, generated_path):
    df1 = pd.read_csv(real_path)
    df2 = pd.read_csv(generated_path)

    # select the column that contains the feature
    column = "TYPE"

    # compute the percentage of each value of the feature in the two dataframes
    percentage_real = df1[column].value_counts(normalize=True)
    percentage_gen = df2[column].value_counts(normalize=True)

    # join the two percentages in a single dataframe
    df_combined = pd.DataFrame({
        'Features values': percentage_gen.index,
        'Percentage_gen': percentage_gen.values,
        'Percentage_real': percentage_real.values
    })

    plt.figure(dpi=300,)

    bar_width = 0.35

    ind = np.arange(len(df_combined['Features values']))

    plt.bar(ind, df_combined['Percentage_gen'], width=bar_width, color="blue", label="generated")
    plt.bar(ind + bar_width, df_combined['Percentage_real'], width=bar_width, color="red", label="real")

    plt.title("Comparison distribution order type")
    plt.xlabel("Order Type")
    plt.ylabel("Percentage")
    plt.xticks(ind + bar_width / 2, df_combined['Features values'])
    plt.legend()

    file_name = "order_type.pdf"
    generated_path = os.path.dirname(generated_path)
    file_path = os.path.join(generated_path, file_name)
    print(file_path)
    plt.savefig(file_path)
    #plt.show()
    plt.close()

if __name__ == '__main__':
    main()