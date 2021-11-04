import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("heart.csv")
    print(df.describe())
    print(df.info())

    for name in df.columns:
        print(len(pd.unique(df[name])), name, " unique values: " )

    sns.boxplot(data=df)
    plt.show()
    df.hist(figsize=(8, 8))
    plt.show()






if __name__ == "__main__":
    main()