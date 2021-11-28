import copy

import pandas as pd
import numpy as np

from preprocessing import fixMissingAndWrongValues, detectOutliers, deleteRowsByIndex, analysingData
from testing import *

def main():

    df = pd.read_csv("heart.csv")


    analysingData(df)

    df = fixMissingAndWrongValues(df)
    outliersToDrop = detectOutliers(df, df.columns.values.tolist(), 2)
    df = deleteRowsByIndex(df, outliersToDrop)


    # UN SOL MODEL PER FER PROVES
    test1Model(copy.deepcopy(df))

    # PER PROVAR EL NOSTRE CROSS VALIDATION I DIFERENTS HEURÍSTIQUES
    testCrossvalidationHeuristics(copy.deepcopy(df), ['id3', 'c45', 'gini'], intervals=[4, 6, 7, 8, 9, 10, 11])

    # PER PROVAR EL NOSTRE CROSS VALIDATION I PROBABILISTIC APPROACH
    testCrossvalidationProbabilisticApproach(copy.deepcopy(df), [False, True], intervals=[4, 6, 7, 8, 9, 10, 11], heuristic='gini')

    # PER PROVAR EL TWO-WAY SPLIT
    testCrossvalidationTwoWaySplit(copy.deepcopy(df), intervals=[5, 10, 20,30, 40, 50, 60, 70, 100, 200, 500], heuristic='gini')

    # PER COMPARAR TEMPS D'EXECUCIÓ INTERVALS VS 2-WAY SPLIT
    testExecutionTime2waysplitVSintervals(copy.deepcopy(df), test_size=0.5, proba=True)

    # IMPLEMENTACIONS AMB SKLEARN, PER FER COMPARACIONS
    crossValidationSklearn(copy.deepcopy(df))
    compareWithSklearn(copy.deepcopy(df))


if __name__ == "__main__":
    main()