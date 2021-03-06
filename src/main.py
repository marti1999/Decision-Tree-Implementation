import copy

import pandas as pd
import numpy as np

from preprocessing import fixMissingAndWrongValues, detectOutliers, deleteRowsByIndex, analysingData
from testing import *

def main():

    df = pd.read_csv("heart.csv")


    # analysingData(df)

    df = fixMissingAndWrongValues(df)
    outliersToDrop = detectOutliers(df, df.columns.values.tolist(), 2)
    df = deleteRowsByIndex(df, outliersToDrop)


    # UN SOL MODEL PER MOSTRAR LA GENERACIÓ DE L'ARBRE
    test1Model(copy.deepcopy(df))

    # PER PROVAR EL NOSTRE CROSS VALIDATION I DIFERENTS HEURÍSTIQUES
    print("Fent apartat 1 (l'heurística c45 triga prop de 10 minuts)")
    testCrossvalidationHeuristics(copy.deepcopy(df), ['id3', 'c45', 'gini'], intervals=[4, 6, 7, 8, 9, 10, 11])

    # PER PROVAR EL NOSTRE CROSS VALIDATION I PROBABILISTIC APPROACH
    print("\nFent apartat 2")
    testCrossvalidationProbabilisticApproach(copy.deepcopy(df), [False, True], intervals=[4, 6, 7, 8, 9, 10, 11], heuristic='gini')

    # PER PROVAR EL TWO-WAY SPLIT
    print("\nFent apartat 3")
    testCrossvalidationTwoWaySplit(copy.deepcopy(df), intervals=[5, 10, 20,30, 40, 50, 60, 70, 100, 200, 500], heuristic='gini')

    # PER COMPARAR TEMPS D'EXECUCIÓ INTERVALS VS 2-WAY SPLIT
    testExecutionTime2waysplitVSintervals(copy.deepcopy(df), test_size=0.2, proba=True)

    # PER PROVAR EL RANDOM FOREST
    print("\nFent apartat 4")
    testRandomForest(copy.deepcopy(df))

    # PER COMPARAR DECISION TREE VS RANDOM FOREST
    testCrossValRandomForestVSDecisionTree(copy.deepcopy(df), n_trees=5, proba=True)

    # IMPLEMENTACIONS AMB SKLEARN, PER FER COMPARACIONS
    print("Comparant amb model de sklearn")
    # crossValidationSklearn(copy.deepcopy(df))
    compareWithSklearn(copy.deepcopy(df))


if __name__ == "__main__":
    main()