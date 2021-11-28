import sklearn
from  scipy.stats import mode

# Inherit de sklearn.base.BaseEstimator
# https://scikit-learn.org/stable/developers/develop.html
# bàsicament per tal de poder utilitzar cross validation i altres mètriques
#   "All estimators in the main scikit-learn codebase should inherit from sklearn.base.BaseEstimator."
# només és necessari per quan volem comparar els nostres mètodes amb els de sklearn. Si no s'utilitza res de sklearn, es pot treure.
from arbredecisio import DecisionTree


class RandomForest(sklearn.base.BaseEstimator):
    # ens hem basat en el següent article que explica la teoria dels random forests
    # https://towardsdatascience.com/understanding-random-forest-58381e0602d2

    def __init__(self, n_trees=5,  heuristic='id3', enableProbabilisticApproach=True):
        self.models = []
        self.n_trees = n_trees
        self.heuristic = heuristic
        self.enableProbabilisticApproach = enableProbabilisticApproach

    def fit(self, df, Y=None):
        for m in range(self.n_trees):
            self.models.append(DecisionTree(self.heuristic, self.enableProbabilisticApproach))

        for m in self.models:
            newdf = self.trainWithReplacement(df)
            m.fit(newdf)

    def predict(self, test):
        predictions = []

        for m in self.models:
            predictions.append(m.predict(test))

        result = self.vote(predictions)
        return result

    def vote(self, list):
        val, count = mode(list, axis=0)
        votes =  val.ravel().tolist()
        return votes

    def trainWithReplacement(self, X):
        newX = X.sample(n=X.shape[0], replace=True).reset_index(drop=True)
        return newX
