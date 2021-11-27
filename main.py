import copy
import statistics
import sys
from collections import Counter
from datetime import datetime
from random import random, seed, randint, randrange

import pandas as pd
import numpy as np
import pprint

import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
import json
from sklearn.model_selection import train_test_split, cross_validate, KFold

# numero més petit possible, per quan obtenim un 0 al denominador fer 0 + eps
eps = np.finfo(float).eps

import seaborn as sns
import matplotlib.pyplot as plt


# no s'utilitza, de moment...
class Node:
    def __init__(self, nId, isLeaf, attribute, attValues = []):
        self.id = nId
        self.isLeaf = isLeaf
        self.attribute = attribute
        self.attValues = attValues
        self.childIDs = []
        self.probabilityClass1 = None
        self.parentID = None


# Inherit de sklearn.base.BaseEstimator
# https://scikit-learn.org/stable/developers/develop.html
# bàsicament per tal de poder utilitzar cross validation i altres mètriques
#   "All estimators in the main scikit-learn codebase should inherit from sklearn.base.BaseEstimator."
# només és necessari per quan volem comparar els nostres mètodes amb els de sklearn. Si no s'utilitza res de sklearn, es pot treure.
class DecisionTree(sklearn.base.BaseEstimator):

    def __init__(self, heuristic='id3', enableProbabilisticApproach=True):
        self.tree = None
        self.heuristic = heuristic
        self.enableProbabilisticApproach = enableProbabilisticApproach # True fa probabilistic approach, False posa un 1 quan no troba el valor
        self.predictions = [] # guarda les prediccions que retorna el predict
        self.class0 = 0 # variable que servirà de comptador quan busquem tots els possibles outcomes de valors no existents a l'arbre
        self.class1 = 0 # variable que servirà de comptador quan busquem tots els possibles outcomes de valors no existents a l'arbre
        self.doingProbabilisticApproach = False # indica si la prediccio actual és normal o ha de fer probabilistic approach


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
        '''counts = []
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
        '''
        gini_classes = {}
        x = df[attribute]
        ret = 0
        #Comptem la quantitat d'atributs differents que tenim
        for element in x.unique():
            gini_classes[element] = 0
            #Cada element different fa de index per a un enter que es el nombre de vegades que es repeteix aquell valor.
            for valor in x:
                if (valor == element):
                    gini_classes[element] += 1;
        #Fem el sumatori, per cada valor diferent, calculem la proporció de vegades que apareix respecte al total i la elevem al quadrat.
        for gini in gini_classes:
            ret += (gini_classes[gini]/df.shape[0])**2

        return 1-ret


        #return giniValues



    def gain(self, eDf, eAttr):
        return eDf - eAttr

    def findBestAttribute(self, df):

        # TODO Exercici 1: el ID3 funciona, el c45 a mitges, el gini sembla ser que no
        gains = []
        gini = []
        attributes = df.keys().tolist()
        attributes.remove('target')

        if len(attributes) == 0:
            print("aquí")

        for attr in attributes:
            if (self.heuristic == 'id3'):
                gains.append(self.datasetEntropy(df) - self.attributeEntropy(df, attr))
            elif (self.heuristic == 'c45'):
                gains.append((self.datasetEntropy(df) - self.attributeEntropy(df, attr))/(self.splitInfo(df, attr)))
            elif (self.heuristic == "gini"):
                gini.append(self.calculateGini(df, attr))

        if(self.heuristic != "gini"):
            return attributes[np.argmax(gains)]
        else:
            return attributes[np.argmin(gini)]

    def get_subtable(self, df, node, value):
        # https://www.sharpsightlabs.com/blog/pandas-reset-index/
        return df[df[node] == value].reset_index(drop=True)

    def createTree(self, df, tree2=None):
        # Busquem l'atribut amb el màxim Gain d'informació
        millorAtribut = self.findBestAttribute(df)

        # Agafem tots els valors únics de l'atribut amb més Gain
        attValue = np.unique(df[millorAtribut])

        # Creem el diccionari que servirà d'arbre
        if tree2 is None:
            tree2 = {}
            tree2[millorAtribut] = {}

        # L'arbre es construirà anant cridant la funció de forma recursiva.
        # Cada valor portarà a un dels nous nodes (atributs)
        for value in attValue:

            # mirem si amb aquest valor tots els resultats són iguals
            subtable = self.get_subtable(df, millorAtribut, value)
            clValue, counts = np.unique(subtable['target'], return_counts=True)



            # si tots són iguals llavors tenim una fulla
            if len(counts) == 1:
                # guardem tant el resultat com el nombre de casos que arriben a aquesta fulla
                tree2[millorAtribut][value] = (clValue[0], counts[0])
            # sinó el valor portarà a un nou node amb un altre atribut
            # li passem el dataset amb les dades que entrarien dins d'aquest node, també treiem l'atribut que ja s'ha mirat doncs no el necessitem més
            else:

                # cas en que ja no quedi cap més atribut a preguntar però hi hagi diferents outcomes
                if subtable.shape[1] == 2:
                    count0 = subtable[subtable['target'] == 0].shape[0]
                    count1 = subtable[subtable['target'] == 1].shape[0]
                    outcome = 1
                    count = count1
                    if count0 > count1:
                        outcome = 0
                        count = count0
                    tree2[millorAtribut][value] = (outcome, count)
                else:

                    tree2[millorAtribut][value] = self.createTree(subtable.drop(columns=[millorAtribut]))

        return tree2


    def fit(self, df, Y=None):
        self.tree = self.createTree(df)


    def lookupOutput(self, row, subTree=None):

        # si no és un diccionari significa que hem arribat a una fulla, busquem quin és el seu output
        if not isinstance(subTree, dict):

            # si hem arribat sense haver de fer probabilistic approach simplement guardem la nova predicció
            if not self.doingProbabilisticApproach:
                self.predictions.append(subTree[0])
            # en cas d'estar fent probabilistic approach, llavors estem buscant múltiples camins la vegada
            # en comptes d'afegir el output de la fulla, sumem al contador de la classe pertinent quantes mostres del training han arribat a aquesta fulla
            # mes endevant es mirarà quina classe té més mostres i s'fegirà coma predicció
            else:
                if subTree[0] == 0:
                    self.class0 += subTree[1]
                else:
                    self.class1 += subTree[1]
            return

        # TODO (es pot deixar per quan ja funcioni tot) esborrar for, realment només hi ha un valor al diccionari (per alguna rao, peta sense el for).
        for atribut_a_preguntar, valorsAtribut in subTree.items():

            valorAtributDelaMostra = getattr(row, atribut_a_preguntar)

            # si el valor de l'atribut ja estisteix a l'arbre, simplement seguim baixant i li passem el subarbre
            if valorAtributDelaMostra in valorsAtribut:
                self.lookupOutput(row, valorsAtribut[valorAtributDelaMostra])

            # en cas que no existeixi aquest valor a l'arbre
            else:
                if  self.enableProbabilisticApproach:
                    # indiquem que fem probabilistic approach per tal que ho sàpiguen les pròximes crides
                    self.doingProbabilisticApproach = True
                    # cada una d'aquestes iteracions explorarà una branca del node actual (múltiples camins)
                    for val in valorsAtribut.keys():
                        self.lookupOutput(row, valorsAtribut[val])
                else:
                    # El que hi havia abans de fer el probabilistic approach. Simplement dèiem que tenia la malaltia.
                    self.predictions.append(1)


    def predict(self, df):
        self.predictions = []

        for row in df.itertuples():

            # reset d'atributs que segurament s'han modificat a la predicció anterior
            self.class0 = 0
            self.class1 = 0
            self.doingProbabilisticApproach = False

            # recorrem l'abre i busquem el resultat
            self.lookupOutput(row, self.tree)

            # si durant l'exploració de l'arbre s'ha agut de fer probabilistic approach,
            # mirem quina classe té més probabilitats de ser la real i guardem el resultat a prediccions
            if self.doingProbabilisticApproach:
                possibleOutcome = 1
                if self.class0 > self.class1:
                    possibleOutcome = 0
                self.predictions.append(possibleOutcome)


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
    # TODO Exercici 3: de moment es fa amb intervals especificats, caldrà programar també el 2-way partition
    # TODO Exercici 3: a part del 2-way partition, també es pot mirar de trobar el nombre d'invervals òptims per cada atribut

    # el cat.cades passa d'intervals a valors numèrics, necessari per tractar després amb ells (bàscicament f el mateix que el label encoding)
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

def fixMissingAndWrongValues(df):

    # Realment no hi ha valors nuls, però si hi haguèssin es faria així per dades categòriques i continues
    # df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    # df['age'] = df['age'].fillna(df['age'].mean())

    # a continuació s'actualitzen els valors que no són correctes
    # s'ha de fer manualment, doncs cada un dels atributs té el seu rang propi
    df.loc[df['ca'] > 3, 'ca'] = df['ca'].mode()[0]
    df.loc[df['ca'] < 0, 'ca'] = df['ca'].mode()[0]
    df.loc[df['thal'] < 1, 'thal'] = df['thal'].mode()[0]
    df.loc[df['thal'] > 3, 'thal'] = df['thal'].mode()[0]
    df.loc[df['restecg'] > 2, 'restecg'] = df['restecg'].mode()[0]
    df.loc[df['restecg'] < 0, 'restecg'] = df['restecg'].mode()[0]
    df.loc[df['cp'] < 0, 'cp'] = df['cp'].mode()[0]
    df.loc[df['cp'] > 3, 'cp'] = df['cp'].mode()[0]


    return df


def detectOutliers(df, atributs, maxOutliers):
    # maxOutliers és el nombre màxim d'outliers permesos per mostra

    indexsOutliers = []

    # iterem sobre tots els atributs
    for atr in atributs:
        # primer quartil
        Q1 = np.percentile(df[atr], 25)
        # tercer quartil
        Q3 = np.percentile(df[atr], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # zona de tall
        cutOff = 1.5 * IQR

        # Busquem els indexs dels registres fora de la zona de tall
        indexsOutliersAtr = df[(df[atr] < Q1 - cutOff) | (df[atr] > Q3 + cutOff)].index

        # Els guardem a la llista general
        indexsOutliers.extend(indexsOutliersAtr)

    # contem quantes vegades ha aparegut cada índex i seleccinem els que sobrepasen el límit especificat
    indexsOutliers = Counter(indexsOutliers)
    indexsDrop = list(k for k, v in indexsOutliers.items() if v > maxOutliers)

    return indexsDrop

def deleteRowsByIndex(df, indexs):
    rows = df.index[indexs]
    # https://stackoverflow.com/questions/43893457/understanding-inplace-true
    df.drop(rows, inplace=True)
    return df


def showMetricPlots(crossValScoresByMetric, metrics=None):

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


def trainTestSplit(df, trainSize=0.8, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    sizeDf = df.shape[0]
    train = df.head(int(trainSize*sizeDf))
    test = df.tail(int((1-trainSize)*sizeDf))
    return train, test


def kFoldSplit(df, n_splits=5):
    splits = []
    splitSize = int(df.shape[0]/n_splits)
    for i in range(n_splits):
        split = df.head(splitSize*i+splitSize).tail(splitSize)
        splits.append(split)
    return splits


def crossValidation(model, df, n_splits=5, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    splits = kFoldSplit(df, n_splits)

    scores = {}
    accuracy = []
    precision = []
    recall = []
    f1Score = []
    for i in range(n_splits):
        train = []
        for j in range(n_splits):
            if j != i:
                train.append(splits[j])
        test = splits[i]
        train = pd.concat(train)

        model.fit(train)
        predictions = model.predict(test)
        groundTruth = test['target'].tolist()
        # print("\n")
        # print(groundTruth)
        # print(predictions)
        accuracy.append(accuracy_score(groundTruth, predictions))
        precision.append(precision_score(groundTruth, predictions))
        recall.append(recall_score(groundTruth, predictions))
        f1Score.append(f1_score(groundTruth, predictions))
    scores['test_accuracy'] = accuracy
    scores['test_precision'] = precision
    scores['test_recall'] = recall
    scores['test_f1Score'] = f1Score
    return scores


def TwoWaySplit(df, attributes, initialIntervals = 11):


    for attribute in attributes:
        attribute_cut = attribute + '_cut'
        attribute_codes = attribute + '_codes'
        maxIntervals = min(initialIntervals, df[attribute].nunique())

        df[attribute_cut] = pd.cut(df[attribute], maxIntervals)
        df[attribute_codes] = df[attribute_cut].cat.codes
        sorted = df.sort_values(attribute_codes)
        sortedValues = list(set(sorted[attribute_codes].tolist()))

        bestSplitPoint = None
        bestWeightedEntropy = 1

        for i in range(len(sortedValues)-1):
            midpoint = (sortedValues[i] + sortedValues[i+1])/2
            dfLeft = df[df[attribute_codes] <= midpoint]
            dfRight = df[df[attribute_codes] > midpoint]

            countLeft0 = dfLeft[dfLeft['target']== 0].shape[0]+eps
            countLeft1 = dfLeft[dfLeft['target']== 1].shape[0]+eps
            countLeft = dfLeft.shape[0]
            eLeft = -(countLeft0 / countLeft) * np.log2(countLeft0 / countLeft) - (countLeft1 / countLeft) * np.log2(countLeft1 / countLeft)

            countRight0 = dfRight[dfRight['target'] == 0].shape[0] +eps
            countRight1 = dfRight[dfRight['target'] == 1].shape[0]+eps
            countRight = dfRight.shape[0]
            eRight = -(countRight0 / countRight) * np.log2(countRight0 / countRight) - (countRight1 / countRight) * np.log2( countRight1 / countRight)

            weigthedEntropy = countLeft / (countLeft + countRight) * eLeft + countRight / (countLeft + countRight) * eRight
            if weigthedEntropy < bestWeightedEntropy:
                bestWeightedEntropy = weigthedEntropy
                bestSplitPoint = midpoint


        splitPointInt = int(bestSplitPoint)
        attributeMidPoint = int( df[df[attribute_codes] == splitPointInt].head(1)[attribute_cut].tolist()[0].right )
        # https://stackoverflow.com/questions/45307376/pandas-df-itertuples-renaming-dataframe-columns-when-printing
        # attributeSplitName = attribute + '>' + str(attributeMidPoint)
        attributeSplitName = attribute + 'GT' + str(attributeMidPoint)
        df[attributeSplitName] = np.nan

        df.loc[df[attribute_codes] <= bestSplitPoint, attributeSplitName] = int(0)
        df.loc[df[attribute_codes] > bestSplitPoint, attributeSplitName] = int(1)

        df = df.drop(columns=[attribute, attribute_codes,attribute_cut])

    return df


def main():

    df = pd.read_csv("heart.csv")

    # analysingData(df)
    df = fixMissingAndWrongValues(df)

    outliersToDrop = detectOutliers(df, df.columns.values.tolist(), 2)
    # TODO en comptes d'esborrar les mostres outliers, donar un nou valor a l'atribut en qüestió
    df = deleteRowsByIndex(df, outliersToDrop)


    # UN SOL MODEL PER FER PROVES
    # dfDiscrete = createDiscreteValues(df, categoriesNumber=7)
    dfDiscrete = TwoWaySplit(df, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], initialIntervals=15)
    # train, test = trainTestSplit(dfDiscrete, trainSize=0.8)
    train, test = train_test_split(dfDiscrete, test_size=0.2, random_state=0) # per si es necessita tenir sempre el mateix split
    decisionTree = DecisionTree(heuristic='gini', enableProbabilisticApproach=True)
    decisionTree.fit(train)
    y_pred = decisionTree.predict(test)
    y_test = test['target'].tolist()
    print(y_test)
    print(y_pred)
    print("Accuracy: ", accuracy_score(y_test, y_pred), " categories: ", 7)
    pprint.pprint(decisionTree.tree)



    # PER PROVAR EL NOSTRE CROSS VALIDATION
    # metrics = ('accuracy', 'precision', 'recall', 'f1Score')
    # crossValScoresByMetric = {}
    # for metric in metrics:
    #     crossValScoresByMetric[metric] = {}
    # for n in [4,6,7,8,9,10,11,12,13,14]:
    #     dfDiscrete = createDiscreteValues(df, categoriesNumber=n)
    #     cv_results = crossValidation(DecisionTree(heuristic='id3', enableProbabilisticApproach=True), dfDiscrete, n_splits=10)
    #     print(cv_results)
    #     for metric in metrics:
    #         crossValScoresByMetric[metric][n] = cv_results["test_" + metric]
    # showMetricPlots(crossValScoresByMetric, metrics=list(metrics))


    # IMPLEMENTACIONS AMB SKLEARN, PER FER COMPARACIONS
    # crossValidationSklearn(df)
    # compareWithSklearn(df)






def crossValidationSklearn(df):
    kf = KFold(n_splits=10, random_state=None, shuffle=True)
    crossValScoresByMetric = {}
    metrics = ('accuracy', 'precision')
    for metric in metrics:
        crossValScoresByMetric[metric] = {}
    for n in [4, 6, 7]:
        dfDiscrete = createDiscreteValues(df, categoriesNumber=n)

        decisionTree = DecisionTree()
        X = dfDiscrete
        y = dfDiscrete["target"]
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