import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Node:
    def __init__(self):
        self.value = None
        self.next = None
        self.parent = None
        self.childs = None


def analysingData(df):
    print(df.describe())
    print(df.info())

    for name in df.columns:
        print(len(pd.unique(df[name])), name, " unique values: ")

    # sns.boxplot(data=df)
    # plt.show()
    # df.hist(figsize=(8, 8))
    # plt.show()

def createDiscreteValues(df):
    # df['age_qcut'] = pd.qcut(df['age'], 10)
    df['age_cut'] = pd.cut(df['age'], 10)
    df['trestbps_cut'] = pd.cut(df['trestbps'], 10)
    df['chol_cut'] = pd.cut(df['chol'], 10)
    df['thalach_cut'] = pd.cut(df['thalach'], 10)
    df['oldpeak_cut'] = pd.cut(df['oldpeak'], 10)

    # print(df[['age', 'age_qcut', 'age_cut']].head(10))
    # print(df[['age_qcut']].value_counts())
    # print(df[['age_cut']].value_counts())

    dfDiscrete = df.drop(columns=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])

    print("\n\n\n")
    for name in dfDiscrete.columns:
        print(len(pd.unique(dfDiscrete[name])), name, " unique values: ")

    return dfDiscrete

def datasetEntropy(df):

    print("hello")


def main():

    df = pd.read_csv("heart.csv")

    analysingData(df)

    dfDiscrete = createDiscreteValues(df)

    datasetEntropy(dfDiscrete)






if __name__ == "__main__":
    main()