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

    # sns.boxplot(data=df)
    # plt.show()
    # df.hist(figsize=(8, 8))
    # plt.show()

    df['age_qcut'] = pd.qcut(df['age'], 10)
    df['age_cut'] = pd.cut(df['age'], 10)
    df['trestbps_cut'] = pd.cut(df['trestbps'], 10)
    df['chol_cut'] = pd.cut(df['chol'], 10)
    df['chol_cut'] = pd.cut(df['chol'], 10)
    df['thalach_cut'] = pd.cut(df['thalach'], 10)
    df['oldpeak_cut'] = pd.cut(df['oldpeak'], 10)

    print(df[['age', 'age_qcut', 'age_cut']].head(10))
    print(df[['age_qcut']].value_counts())
    print(df[['age_cut']].value_counts())


if __name__ == "__main__":
    main()