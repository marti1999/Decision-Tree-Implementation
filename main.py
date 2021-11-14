import pandas as pd
import numpy as np

# numero més petit possible, per quan obtenim un 0 al denominador fer 0 + eps
eps = np.finfo(float).eps

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

    """
        per cada valor únic al target (hasCancer):
            p = probabilitat del valor únic
            totalEntropy = totalentropy + p*log_2(p)
    """

    entropy = 0
    uniqueValues = df.target.unique()
    for v in uniqueValues:
        p = df.target.value_counts()[v]/len(df.target)
        entropy += p*np.log2(p)

    entropy = entropy * -1
    print(entropy)
    return entropy

def attributeEntropy(df, attribute):
    results = df.target.unique()
    attrValues = df[attribute].unique()

    attrEntropy = 0
    for value in attrValues:
        entropyEachValue = 0
        for result in results:
            num = len(df[attribute][df[attribute] == value][df.target == result])
            den = len(df[attribute][df[attribute] == value])
            innerFraction = num/(den+eps)
            entropyEachValue += -innerFraction * np.log2(innerFraction + eps)
        outerFraction = den/len(df)
        attrEntropy += -outerFraction * entropyEachValue

    return abs(attrEntropy)

def gain(eDf, eAttr):
    return eDf-eAttr


def main():

    df = pd.read_csv("heart.csv")

    analysingData(df)

    dfDiscrete = createDiscreteValues(df)
    dfEntropy = datasetEntropy(dfDiscrete)

    a = dfDiscrete.keys().tolist()
    a.remove('target')
    entropyDictionary = {k : attributeEntropy(dfDiscrete, k) for k in a}
    print(entropyDictionary)

    gainDictionary = {k:gain(dfEntropy, entropyDictionary[k]) for k in entropyDictionary}
    print(gainDictionary)


if __name__ == "__main__":
    main()