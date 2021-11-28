import copy

import pandas as pd
import numpy as np

from preprocessing import fixMissingAndWrongValues, detectOutliers, deleteRowsByIndex
from testing import *

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
    # test1Model(copy.deepcopy(df))

    # PER PROVAR EL NOSTRE CROSS VALIDATION I DIFERENTS HEURÍSTIQUES
    # testCrossvalidationHeuristics(copy.deepcopy(df), ['id3', 'c45', 'gini'], intervals=[4, 6])

    # PER PROVAR EL NOSTRE CROSS VALIDATION I PROBABILISTIC APPROACH
    # testCrossvalidationProbabilisticApproach(copy.deepcopy(df), [False, True], intervals=[4, 6], heuristic='gini')

    # PER PROVAR EL TWO-WAY SPLIT
    # testCrossvalidationTwoWaySplit(copy.deepcopy(df), intervals=[5, 10, 20,30, 40, 50, 60, 70, 100, 200, 500], heuristic='gini')

    # PER COMPARAR TEMPS D'EXECUCIÓ INTERVALS VS 2-WAY SPLIT
    # testExecutionTime2waysplitVSintervals(copy.deepcopy(df), test_size=0.5, proba=True)

    # IMPLEMENTACIONS AMB SKLEARN, PER FER COMPARACIONS
    # crossValidationSklearn(copy.deepcopy(df))
    # compareWithSklearn(copy.deepcopy(df))


if __name__ == "__main__":
    main()