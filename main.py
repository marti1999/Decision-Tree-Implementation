import statistics
import sys
from datetime import datetime
from random import random, seed, randint, randrange

import pandas as pd
import numpy as np
import pprint

import sklearn
from sklearn.metrics import accuracy_score,  precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
import json
from sklearn.model_selection import train_test_split, cross_validate, KFold

# numero més petit possible, per quan obtenim un 0 al denominador fer 0 + eps
eps = np.finfo(float).eps

import seaborn as sns
import matplotlib.pyplot as plt

class Node:
    def __init__(self):
        self.id = None
        self.value = None
        self.probabilityClass1 = None
        self.parentID = None
        self.childIDs = None


# Inherit de sklearn.base.BaseEstimator
# https://scikit-learn.org/stable/developers/develop.html
# bàsicament per tal de poder utilitzar cross validation i altres mètriques
#   "All estimators in the main scikit-learn codebase should inherit from sklearn.base.BaseEstimator."
class DecisionTree(sklearn.base.BaseEstimator):
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

    def splitInfo(self, df, attribute):
        '''results = df.target.unique()
        attrValues = df[attribute].unique()

        attrSplitInfo = 0
        for value in attrValues:
            entropyEachValue = 0
            for result in results:
                num = len(df[attribute][df[attribute] == value][df.target == result])
                den = len(df[attribute][df[attribute] == value])
                innerFraction = num / (den + eps)
                entropyEachValue += -innerFraction * np.log2(innerFraction + eps)
            outerFraction = den / len(df)
            attrSplitInfo += -outerFraction * np.log2(den/len(df))

        return abs(attrSplitInfo)'''
        counts = []
        for value in df[attribute].unique():
            subset = df[df[attribute] == value]
            counts.append(subset.shape[0])
        totalCount = np.sum(counts)
        splitInfo = -np.sum(np.multiply(np.divide(counts, totalCount), np.log2(np.divide(counts, totalCount))))
        return splitInfo

    def calculateGini(self, df, attribute):
        counts = []
        giniValues  = []
        giniValuesSub = []
        for value in df[attribute].unique():
            subset = df[df[attribute] == value]
            counts.append(subset.shape[0])
            positives = subset[subset['target'] == 1]
            negatives = subset[subset['target'] != 1]
            giniSubindex = np.subtract(1, np.add(
                np.square(np.divide(positives.shape[0], subset.shape[0])),
                np.square(np.divide(negatives.shape[0], subset.shape[0]))))
            giniValuesSub.append(giniSubindex)
        totalCount = np.sum(counts)
        gini = np.sum(np.multiply(giniValuesSub, np.divide(counts, totalCount)))
        giniValues.append(gini)

        return giniValues



    def gain(self, eDf, eAttr):
        return eDf - eAttr

    def findBestAttribute(self, df, heuristica="id3"):
        gains = []
        gini = []
        attributes = df.keys().tolist()
        attributes.remove('target')
        for attr in attributes:
            if (heuristica == 'id3'):
                gains.append(self.datasetEntropy(df) - self.attributeEntropy(df, attr))
            elif (heuristica == 'c45'):
                gains.append((self.datasetEntropy(df) - self.attributeEntropy(df, attr))/(self.splitInfo(df, attr)))
            elif (heuristica == "gini"):
                gini.append(self.calculateGini(df, attr))

        if(heuristica != "gini"):
            return attributes[np.argmax(gains)]
        else:
            return attributes[np.argmin(gini)]

    def get_subtable(self, df, node, value):
        # https://www.sharpsightlabs.com/blog/pandas-reset-index/
        return df[df[node] == value].reset_index(drop=True)

    def createTree(self, df, tree2=None, heuristica="id3"):
        features = df.keys().tolist()
        features.remove('target')
        Class = features

        # Busquem l'atribut amb el màxim Gain d'informació
        node = self.findBestAttribute(df, heuristica=heuristica)

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
            # li passem el dataset amb les dades que entrarien dins d'aquest node
            else:
                tree2[node][value] = self.createTree(subtable, heuristica=heuristica)

        return tree2

    def setEventProbabilityOnNodes(self,df ):
        # TODO s'hauria de fer en algun moment...
        # https://www.geeksforgeeks.org/python-update-nested-dictionary/
        for row in df.itertuples():
            print(1)



    def fit(self, df, Y=None, heuristica='id3'):
        self.tree = self.createTree(df, heuristica=heuristica)
        # TODO descomentar quan estigui completa
        # self.setEventProbabilityOnNodes(df)


    def lookupOutput(self, row, subTree=None):
        # TODO s'haurà de modificar la funció un cop s'hagi implementat les probabilitats de 0 i 1 a cada node

        if not isinstance(subTree, dict):
            self.predictions.append(subTree)
            return

        # TODO esborrar for, realment només hi ha un valor al diccionari.
        for key, value in subTree.items():
            rowValue = getattr(row, key)
            if rowValue in value:
                self.lookupOutput(row, value[rowValue])
            else:
                # TODO de moment quan arriba un valor de l'atribut que no està a l'arbre direm que SÍ té enfermetat
                #  mes endavant, quan tinguem les probabilitats a cada node ja s'aplicarà el prune.
                self.predictions.append(1)





    def predict(self, df):
        self.predictions = []

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

def createDiscreteValues(df, categoriesNumber):
    # df['age_qcut'] = pd.qcut(df['age'], 10)
    df['age_cut'] = pd.cut(df['age'], categoriesNumber).cat.codes
    df['trestbps_cut'] = pd.cut(df['trestbps'], categoriesNumber).cat.codes
    df['chol_cut'] = pd.cut(df['chol'], categoriesNumber).cat.codes
    df['thalach_cut'] = pd.cut(df['thalach'], categoriesNumber).cat.codes
    df['oldpeak_cut'] = pd.cut(df['oldpeak'], categoriesNumber).cat.codes

    # print(df[['age', 'age_qcut', 'age_cut']].head(10))
    # print(df[['age_qcut']].value_counts())
    # print(df[['age_cut']].value_counts())

    dfDiscrete = df.drop(columns=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])

    #print("\n\n\n")
    #for name in dfDiscrete.columns:
        #print(len(pd.unique(dfDiscrete[name])), name, " unique values: ")

    return dfDiscrete


def showMetricPlots(crossValScoresByMetric, metrics=None):
    if metrics is None:
        return

    for metric in metrics:
        crossValScoresByN = crossValScoresByMetric[metric]
        labels, data = crossValScoresByN.keys(), crossValScoresByN.values()
        plt.boxplot(data)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylim([0.5, 1])
        plt.xlabel("number of categories")
        plt.ylabel(metric)
        plt.show()
        for k in crossValScoresByN.keys():
            crossValScoresByN[k] = statistics.mean(crossValScoresByN[k])
        plt.plot(list(crossValScoresByN.keys()), list(crossValScoresByN.values()))
        plt.ylim([0.5, 1])
        plt.xlabel("number of categories")
        plt.ylabel("average " + metric)
        plt.show()
        #print(json.dumps(crossValScoresByN, indent=4))

def main():


    df = pd.read_csv("heart.csv")

    #analysingData(df)

    dfDiscrete = createDiscreteValues(df, categoriesNumber=7)
    # train, test = train_test_split(dfDiscrete, test_size=0.2)
    # decisionTree = DecisionTree()
    # #model = ['id3', 'c45', 'gini']
    # heuristica = "id3"
    # decisionTree.fit(train, heuristica)
    # predictions = decisionTree.predict(test)
    # groundTruth = test['target'].tolist()
    # print("\n")
    # print(groundTruth)
    # print(predictions)
    # accuracy = accuracy_score(groundTruth, predictions)
    # print("Accuracy: ", accuracy, " categories: ", 7)
    # pprint.pprint(decisionTree.tree)


    # train, test = trainTestSplit(df, 0.8)
    model = DecisionTree()
    crossValidation(model, dfDiscrete, 5)


    """
    CROSS VALIDATION MANUAL
    accuracyByCategoryNumber = {}
    for n in [4,6,7,8,9,10,11,12,13,14,15]:
        accuracyByCategoryNumber[n] = []
        dfDiscrete = createDiscreteValues(df, n)

        for i in range(0, 10):
            seed(n*i+1)

            train, test = train_test_split (dfDiscrete, test_size=0.2)
            decisionTree = DecisionTree()
            decisionTree.fit(train, c45=False)
            predictions = decisionTree.predict(test)
            groundTruth = test['target'].tolist()
            print("\n")
            print(groundTruth)
            print(predictions)
            accuracy = accuracy_score(groundTruth, predictions)
            accuracyByCategoryNumber[n].append(accuracy)
            print("Accuracy: ",accuracy, " categories: ", n)
            # pprint.pprint(decisionTree.tree)

    showMetricPlots(accuracyByCategoryNumber)
    """

    # crossValidationSklearn(df)



    # compareWithSklearn(df)


def trainTestSplit(df, trainSize=0.8, shuffle=True):
    df = df.sample(frac=1).reset_index(drop=True)
    sizeDf = df.shape[0]
    train = df.head(int(trainSize*sizeDf))
    test = df.tail(int((1-trainSize)*sizeDf))
    return train, test

def kFoldSplit(df, n_splits=5):
    splits = []
    sizeDf = df.shape[0]
    splitSize = int(df.shape[0]/n_splits)
    for i in range(n_splits):
        split = df.head(splitSize*i+splitSize).tail(splitSize)
        splits.append(split)
    return splits


def crossValidation(model, df, n_splits, scoring=['accuracy'], shuffle=True):
    sizeDf = df.shape[0]
    features = df.keys().tolist()
    if shuffle: df = df.sample(frac=1).reset_index(drop=True)
    splits = kFoldSplit(df, n_splits)
    print(10)

    for i in range(n_splits):
        train = []
        for j in range(n_splits):
            if j != i:
                train.append(splits[j])
        test = splits[i]
        train = pd.concat(train)

        model.fit(train, heuristica='id3')
        predictions = model.predict(test)
        groundTruth = test['target'].tolist()
        print("\n")
        print(groundTruth)
        print(predictions)
        print(accuracy_score(groundTruth, predictions))




def crossValidationSklearn(df):
    kf = KFold(n_splits=10, random_state=None, shuffle=True)
    crossValScoresByMetric = {}
    metrics = ('accuracy', 'precision')
    for metric in metrics:
        crossValScoresByMetric[metric] = {}
    for n in [4, 6, 7, 8, 9, 10, 11, 12]:
        dfDiscrete = createDiscreteValues(df, categoriesNumber=n)

        decisionTree = DecisionTree()
        X = dfDiscrete
        y = dfDiscrete["target"]
        # TODO veure com passar-li arguments pròpis del model al cross_validate (pel tema del C4.5)
        cv_results = cross_validate(decisionTree, X, y, cv=kf, scoring=metrics)
        print(cv_results)

        for metric in metrics:
            crossValScoresByMetric[metric][n] = cv_results["test_" + metric]
    showMetricPlots(crossValScoresByMetric, metrics=['accuracy', 'precision'])


def compareWithSklearn(df):
    print("\n\n UTILITZANT EL DEL SKLEARN")
    y = df["target"]
    X = df.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=6)
    dt.fit(X_train, y_train)
    dt_predicted = dt.predict(X_test)
    dt_acc_score = accuracy_score(y_test, dt_predicted)
    print("Accuracy of DecisionTreeClassifier:", dt_acc_score, '\n')


if __name__ == "__main__":
    main()