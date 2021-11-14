import pandas as pd
import numpy as np
import pprint
from sklearn.metrics import accuracy_score,  precision_score, recall_score

# numero més petit possible, per quan obtenim un 0 al denominador fer 0 + eps
from sklearn.model_selection import train_test_split

eps = np.finfo(float).eps

import seaborn as sns
import matplotlib.pyplot as plt

class Node:
    def __init__(self):
        self.value = None
        self.next = None
        self.parent = None
        self.childs = None

class DecisionTree:
    def __init__(self):
        self.tree = None
        self.predictions = []

    def datasetEntropy(self, df):

        """
            per cada valor únic al target (hasCancer):
                p = probabilitat del valor únic
                totalEntropy = totalentropy + p*log_2(p)
        """

        entropy = 0
        uniqueValues = df.target.unique()
        for v in uniqueValues:
            p = df.target.value_counts()[v] / len(df.target)
            entropy += p * np.log2(p)

        entropy = entropy * -1
        # print(entropy)
        return entropy

    def attributeEntropy(self, df, attribute):
        results = df.target.unique()
        attrValues = df[attribute].unique()

        attrEntropy = 0
        for value in attrValues:
            entropyEachValue = 0
            for result in results:
                num = len(df[attribute][df[attribute] == value][df.target == result])
                den = len(df[attribute][df[attribute] == value])
                innerFraction = num / (den + eps)
                entropyEachValue += -innerFraction * np.log2(innerFraction + eps)
            outerFraction = den / len(df)
            attrEntropy += -outerFraction * entropyEachValue

        return abs(attrEntropy)

    def gain(self, eDf, eAttr):
        return eDf - eAttr

    def findBestAttribute(self, df):
        gains = []
        a = df.keys().tolist()
        a.remove('target')
        for key in a:
            gains.append(self.datasetEntropy(df) - self.attributeEntropy(df, key))
        return a[np.argmax(gains)]

    def get_subtable(self, df, node, value):
        # https://www.sharpsightlabs.com/blog/pandas-reset-index/
        return df[df[node] == value].reset_index(drop=True)

    def createTree(self, df, tree2=None):
        features = df.keys().tolist()
        features.remove('target')
        Class = features

        # Busquem l'atribut amb el màxim Gain d'informació
        node = self.findBestAttribute(df)

        # Agafem tots els valors únics de l'atribut amb més Gain
        attValue = np.unique(df[node])

        # Creem el diccionari que servirà d'arbre
        if tree2 is None:
            tree2 = {}
            tree2[node] = {}

        # L'arbre es construirà anant cridant la funció de forma recursiva.

        # Cada valor portarà a un dels noos nodes (atributs)
        for value in attValue:

            # mirem si amb aquest valor tots els resultats són iguals
            subtable = self.get_subtable(df, node, value)
            clValue, counts = np.unique(subtable['target'], return_counts=True)

            # si tots són iguals llavors tenim una fulla
            if len(counts) == 1:  # Checking purity of subset
                tree2[node][value] = clValue[0]
            # sinó el valor portarà a un nou node amb un altre atribut
            else:
                tree2[node][value] = self.createTree(subtable)

        return tree2

    def fit(self, df):
        self.tree = self.createTree(df)


    def lookupOutput(self, row, subTree=None):

        if not isinstance(subTree, dict):
            self.predictions.append(subTree)
            return

        #TODO esborrar for, realment només hi ha un valor al diccionari.
        for key, value in subTree.items():
            rowValue = getattr(row, key)
            if rowValue in value:
                self.lookupOutput(row, value[rowValue])
            else:
                self.predictions.append(2)





    def predict(self, df):
        self.predictions = []
        # print(row.age_cut)

        for row in df.itertuples():
            self.lookupOutput(row, self.tree)

        return self.predictions


def analysingData(df):
    print(df.describe())
    print(df.info())

    for name in df.columns:
        print(len(pd.unique(df[name])), name, " unique values: ")

    sns.boxplot(data=df)
    plt.show()
    df.hist(figsize=(8, 8))
    plt.show()

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

    #print("\n\n\n")
    #for name in dfDiscrete.columns:
        #print(len(pd.unique(dfDiscrete[name])), name, " unique values: ")

    return dfDiscrete





def main():

    df = pd.read_csv("heart.csv")

    #analysingData(df)

    dfDiscrete = createDiscreteValues(df)
    # dfEntropy = datasetEntropy(dfDiscrete)
    #
    # a = dfDiscrete.keys().tolist()
    # a.remove('target')
    # entropyDictionary = {k : attributeEntropy(dfDiscrete, k) for k in a}
    # print(entropyDictionary)
    #
    # gainDictionary = {k:gain(dfEntropy, entropyDictionary[k]) for k in entropyDictionary}
    # print(gainDictionary)


    print("un 2 significa que l'arbre no té el valor per algun atribut i per tant no pot arribar a cap fulla")
    totalAccuracy = 0
    for i in range(0, 10):
        train, test = train_test_split (dfDiscrete, test_size=0.20, random_state=np.random)
        decisionTree = DecisionTree()
        decisionTree.fit(train)
        predictions = decisionTree.predict(test)
        groundTruth = test['target'].tolist()
        print("\n")
        print(groundTruth)
        print(predictions)
        accuracy = accuracy_score(groundTruth, predictions)
        totalAccuracy = totalAccuracy + accuracy
        # print("Accuracy: ",accuracy)
        # pprint.pprint(decisionTree.tree)

    print("Mean accuracy: ", totalAccuracy/10)


if __name__ == "__main__":
    main()