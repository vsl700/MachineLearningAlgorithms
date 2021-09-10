import pandas as pd


def get_data():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/abalone/abalone.data")

    print(url)
    abalone = pd.read_csv(url, header=None)
    abalone.columns = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Rings"
    ]

    abalone = abalone.drop("Sex", axis=1)

    print(abalone.head(15))
    print("Rows:", len(abalone["Rings"]))  # 4177

    # abalone["Rings"].hist(bins=15)
    # plt.show()

    # Correlation in this case shows whether we can predict the amount of rings with the physical measurements
    # The closer the numbers to 1, the better
    correlation_matrix = abalone.corr()
    print('\nCorrelation:\n', correlation_matrix["Rings"], '\n', sep='\n')

    return [abalone.drop("Rings", axis=1).values, abalone["Rings"].values]
