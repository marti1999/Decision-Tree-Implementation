import pandas as pd
import numpy as np

from preprocessing import fixMissingAndWrongValues, detectOutliers, deleteRowsByIndex
from testing import testCrossvalidationHeuristics, testCrossvalidationProbabilisticApproach, test1Model, \
    crossValidationSklearn, compareWithSklearn

eps = np.finfo(float).eps

class Node:
    def __init__(self, nId, isLeaf, attribute, attValues = []):
        self.id = nId
        self.isLeaf = isLeaf
        self.attribute = attribute
        self.attValues = attValues
        self.childIDs = []
        self.probabilityClass1 = None
        self.parentID = None


def main():

    df = pd.read_csv("heart.csv")

    # PROVES PER FER PRINT DE L'ARBRE
    # tree = {'Weather': {            'Sunny': {'Humidity': {                'High': 'NO',                'Normal': 'YES'            }            },            'Cloudy': 'YES',            'Rainy': {'Wind': {                'Strong': 'NO',                'Weak': 'YES'            }            }}}
    # print(json.dumps(tree, indent=4))
    # pprint.pprint(tree)
    # recursive_print_dict(tree, 2)

    # analysingData(df)
    #plotNull(df);

    df = fixMissingAndWrongValues(df)
    outliersToDrop = detectOutliers(df, df.columns.values.tolist(), 2)
    # TODO en comptes d'esborrar les mostres outliers, donar un nou valor a l'atribut en qüestió
    df = deleteRowsByIndex(df, outliersToDrop)

    # UN SOL MODEL PER FER PROVES
    test1Model(df)

    # PER PROVAR EL NOSTRE CROSS VALIDATION I DIFERENTS HEURÍSTIQUES
    testCrossvalidationHeuristics(df, ['id3', 'c45', 'gini'], intervals=[4, 6])

    # PER PROVAR EL NOSTRE CROSS VALIDATION I PROBABILISTIC APPROACH
    testCrossvalidationProbabilisticApproach(df, [False, True], intervals=[4, 6], heuristic='gini')

    # IMPLEMENTACIONS AMB SKLEARN, PER FER COMPARACIONS
    crossValidationSklearn(df)
    compareWithSklearn(df)


if __name__ == "__main__":
    main()