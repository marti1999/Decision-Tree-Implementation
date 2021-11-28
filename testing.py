import pprint

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from arbredecisio import DecisionTree
from preprocessing import createDiscreteValues
from plots import showMetricPlots, showMetricTwoHeuristicsPlots
from modelSelection import crossValidation


def testCrossvalidationHeuristics(df, heuristics, intervals=[4,6,7,8,7,9,10,11,12,13], proba=False, n_splits=10):

    metrics = ('accuracy', 'precision', 'recall', 'f1Score')
    crossValScores = []

    for heuristic in heuristics:
        print(heuristic)
        crossValScoresByMetric = {}
        for metric in metrics:
            crossValScoresByMetric[metric] = {}
        for n in intervals:
            dfDiscrete = createDiscreteValues(df, categoriesNumber=n)
            cv_results = crossValidation(DecisionTree(heuristic=heuristic, enableProbabilisticApproach=proba), dfDiscrete, n_splits=n_splits)
            # print(cv_results)
            for metric in metrics:
                crossValScoresByMetric[metric][n] = cv_results["test_" + metric]
        crossValScores.append(crossValScoresByMetric)

    showMetricTwoHeuristicsPlots(crossValScores, metrics=list(metrics), legend = heuristics)


def testCrossvalidationProbabilisticApproach(df, proba=[False, True], intervals=[4,6,7,8,7,9,10,11,12,13], heuristic='gini'):

    metrics = ('accuracy', 'precision', 'recall', 'f1Score')
    crossValScores = []

    for prob in proba:
        print(heuristic)
        crossValScoresByMetric = {}
        for metric in metrics:
            crossValScoresByMetric[metric] = {}
        for n in intervals:
            dfDiscrete = createDiscreteValues(df, categoriesNumber=n)
            cv_results = crossValidation(DecisionTree(heuristic=heuristic, enableProbabilisticApproach=prob), dfDiscrete)
            # print(cv_results)
            for metric in metrics:
                crossValScoresByMetric[metric][n] = cv_results["test_" + metric]
        crossValScores.append(crossValScoresByMetric)

    showMetricTwoHeuristicsPlots(crossValScores, metrics=list(metrics), legend = proba,title='Probabilistic approach, heuristica = ' + heuristic)


def test1Model(df):
    # UN SOL MODEL PER FER PROVES
    dfDiscrete = createDiscreteValues(df, categoriesNumber=7)
    # dfDiscrete = TwoWaySplit(df, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], initialIntervals=15)
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