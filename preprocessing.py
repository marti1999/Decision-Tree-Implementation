from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

eps = np.finfo(float).eps


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