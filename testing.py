import pprint
import time

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from arbredecisio import DecisionTree
from preprocessing import createDiscreteValues, TwoWaySplit
from plots import showMetricPlots, showMetricTwoHeuristicsPlots, showBarPlot, plotConfusionMatrix, showBarPlot2
from modelSelection import crossValidation
from randomForest import RandomForest


def testCrossvalidationHeuristics(df, heuristics, intervals=[4, 6, 7, 8, 7, 9, 10, 11, 12, 13], proba=False,
                                  n_splits=10):
    metrics = ('accuracy', 'precision', 'recall', 'f1Score')
    crossValScores = []

    for heuristic in heuristics:
        print(heuristic)
        crossValScoresByMetric = {}
        for metric in metrics:
            crossValScoresByMetric[metric] = {}
        for n in intervals:
            dfDiscrete = createDiscreteValues(df, categoriesNumber=n)
            cv_results = crossValidation(DecisionTree(heuristic=heuristic, enableProbabilisticApproach=proba),
                                         dfDiscrete, n_splits=n_splits)
            # print(cv_results)
            for metric in metrics:
                crossValScoresByMetric[metric][n] = cv_results["test_" + metric]
        crossValScores.append(crossValScoresByMetric)

    showMetricTwoHeuristicsPlots(crossValScores, metrics=list(metrics), legend=heuristics)


def testCrossvalidationProbabilisticApproach(df, proba=[False, True], intervals=[4, 6, 7, 8, 7, 9, 10, 11, 12, 13],
                                             heuristic='gini'):
    metrics = ('accuracy', 'precision', 'recall', 'f1Score')
    crossValScores = []

    for prob in proba:
        print(heuristic)
        crossValScoresByMetric = {}
        for metric in metrics:
            crossValScoresByMetric[metric] = {}
        for n in intervals:
            dfDiscrete = createDiscreteValues(df, categoriesNumber=n)
            cv_results = crossValidation(DecisionTree(heuristic=heuristic, enableProbabilisticApproach=prob),
                                         dfDiscrete)
            # print(cv_results)
            for metric in metrics:
                crossValScoresByMetric[metric][n] = cv_results["test_" + metric]
        crossValScores.append(crossValScoresByMetric)

    showMetricTwoHeuristicsPlots(crossValScores, metrics=list(metrics), legend=proba,
                                 title='Probabilistic approach, heuristica = ' + heuristic)


def testCrossvalidationTwoWaySplit(df, intervals=[5, 10, 20, 50, 500], heuristic='gini'):
    metrics = ('accuracy', 'precision', 'recall', 'f1Score')
    crossValScores = []

    for n in intervals:
        print(n)
        crossValScoresByMetric = {}
        for metric in metrics:
            crossValScoresByMetric[metric] = {}
        dfDiscrete = TwoWaySplit(df, attributes=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], initialIntervals=n)
        cv_results = crossValidation(DecisionTree(heuristic=heuristic, enableProbabilisticApproach=True), dfDiscrete)
        print(cv_results)
        for metric in metrics:
            crossValScoresByMetric[metric][n] = cv_results["test_" + metric]
        crossValScores.append(crossValScoresByMetric)

    showBarPlot(crossValScores, metrics=list(metrics), legend=intervals,
                title='2-way partitioning, heuristica = ' + heuristic)


def testExecutionTime2waysplitVSintervals(df, heuristic='gini', n_splits=10, proba=False, test_size=0.2):
    dfInterrvals10 = createDiscreteValues(df, categoriesNumber=4)
    dfInterrvals30 = createDiscreteValues(df, categoriesNumber=30)
    df2waysplit = TwoWaySplit(df, initialIntervals=500)

    train, test = train_test_split(dfInterrvals10, test_size=test_size, random_state=0)
    decisionTree = DecisionTree(heuristic='gini', enableProbabilisticApproach=proba)
    startTime = time.time()
    decisionTree.fit(dfInterrvals10)
    print("Temps Fit 4 intervals en segons: ", time.time() - startTime)
    startTime = time.time()
    decisionTree.predict(test)
    print("Temps Predict 4 intervals en segons: ", time.time() - startTime)

    train, test = train_test_split(dfInterrvals30, test_size=test_size, random_state=0)
    decisionTree = DecisionTree(heuristic='gini', enableProbabilisticApproach=proba)
    startTime = time.time()
    decisionTree.fit(dfInterrvals30)
    print("Temps Fit 30 intervals en segons: ", time.time() - startTime)
    startTime = time.time()
    decisionTree.predict(test)
    print("Temps Predict 30 intervals en segons: ", time.time() - startTime)

    train, test = train_test_split(df2waysplit, test_size=test_size, random_state=0)
    decisionTree = DecisionTree(heuristic='gini', enableProbabilisticApproach=proba)
    startTime = time.time()
    decisionTree.fit(df2waysplit)
    print("Temps Fit 2-way split en segons: ", time.time() - startTime)
    startTime = time.time()
    decisionTree.predict(test)
    print("Temps Predict 2-way split en segons: ", time.time() - startTime)

def test1Model(df):
    # dfDiscrete = createDiscreteValues(df, categoriesNumber=7)
    dfDiscrete = TwoWaySplit(df, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], initialIntervals=15)
    # train, test = trainTestSplit(dfDiscrete, trainSize=0.8)
    train, test = train_test_split(dfDiscrete, test_size=0.2,
                                   random_state=0)  # per si es necessita tenir sempre el mateix split
    decisionTree = DecisionTree(heuristic='gini', enableProbabilisticApproach=True)
    decisionTree.fit(train)
    y_pred = decisionTree.predict(test)
    y_test = test['target'].tolist()
    print(y_test)
    print(y_pred)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    pprint.pprint(decisionTree.tree)


    plotConfusionMatrix(y_pred, y_test)
    print(classification_report(y_test, y_pred, labels=[0,1]))

def testRandomForest(df):
    dfDiscrete = TwoWaySplit(df, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], initialIntervals=15)
    train, test = train_test_split(dfDiscrete, test_size=0.2,
                                   random_state=0)  # per si es necessita tenir sempre el mateix split
    rf = RandomForest(n_trees=5, heuristic='gini', enableProbabilisticApproach=True)
    rf.fit(train)
    y_pred = rf.predict(test)
    y_test = test['target'].tolist()
    plotConfusionMatrix(y_pred, y_test)
    print(classification_report(y_test, y_pred))

def testCrossValRandomForestVSDecisionTree(df, n_trees = 5, heuristic='gini', proba=True):
    metrics = ('accuracy', 'precision', 'recall', 'f1_micro')
    crossValScores = []

    # tot i tenir el nostre kfold i cross validation, utilitzem el de sklearn per assegurar-nos que ambdos models tenen el mateix split
    kf = KFold(n_splits=10, random_state=None, shuffle=True)
    crossValScoresByMetric = {}
    for metric in metrics:
        crossValScoresByMetric[metric] = {}

    dfDiscrete = TwoWaySplit(df, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], initialIntervals=15)
    X = dfDiscrete
    y = dfDiscrete["target"]

    decisionTree = DecisionTree(heuristic='gini', enableProbabilisticApproach=False)
    rf = RandomForest(n_trees=n_trees, heuristic='gini', enableProbabilisticApproach=proba)
    rf2 = RandomForest(n_trees=n_trees*2, heuristic='gini', enableProbabilisticApproach=proba)


    crossValScores.append(cross_validate(decisionTree, X, y, cv=kf, scoring=metrics))
    crossValScores.append(cross_validate(rf, X, y, cv=kf, scoring=metrics))
    crossValScores.append(cross_validate(rf2, X, y, cv=kf, scoring=metrics))


    showBarPlot2(crossValScores, list(metrics), legend=['DT', 'RF '+str(n_trees)+' trees', 'RF '+str(n_trees*2)+' trees'], title='Decision Tree VS Random Forest', )


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
    plotConfusionMatrix(dt_predicted, y_test)
    print(classification_report(y_test, dt_predicted, labels=[0,1]))
