import numpy as np
import sklearn

# numero més petit possible, per quan obtenim un 0 al denominador fer 0 + eps
eps = np.finfo(float).eps

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

        x = df[attribute]
        splitIClasses = {}
        ret = 0
        for element in x.unique():
            splitIClasses[element] = 0
            # Cada element different fa de index per a un enter que es el nombre de vegades que es repeteix aquell valor.
            for valor in x:
                if (valor == element):
                    splitIClasses[element] += 1;
        for classe in splitIClasses:
            ret += (splitIClasses[classe] / df.shape[0]) * np.log2(splitIClasses[classe] / df.shape[0])
        ret = -ret

        if (ret == 0):
            ret = eps
        return -ret


    def gini_impurity(self, y):

        # el np.sum funciona com a sumatori, simplifica molt la feina
        p = y.value_counts() / y.shape[0]
        gini = 1 - np.sum(p ** 2)
        return (gini)


    def gain(self, eDf, eAttr):
        return eDf - eAttr

    def findBestAttribute(self, df):

        gains = []
        gini = []
        attributes = df.keys().tolist()
        attributes.remove('target')


        for attr in attributes:
            if (self.heuristic == 'id3'):
                gains.append(self.datasetEntropy(df) - self.attributeEntropy(df, attr))
            elif (self.heuristic == 'c45'):
                gains.append((self.datasetEntropy(df) - self.attributeEntropy(df, attr))/(self.splitInfo(df, attr)))
            elif (self.heuristic == "gini"):
                gini.append(self.gini_impurity(df[attr]))

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

        # L'arbre es construirà cridant a la funció de forma recursiva.
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
            else:

                # cas en que ja no quedi cap més atribut a preguntar però hi hagi diferents outcomes es crea una fulla
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
                    # li passem el dataset amb les dades que entrarien dins d'aquest node, també treiem l'atribut que ja s'ha mirat doncs no el necessitem més
                    tree2[millorAtribut][value] = self.createTree(subtable.drop(columns=[millorAtribut]))

        return tree2


    def fit(self, df, Y=None):
        self.tree = self.createTree(df)


    def lookupOutput(self, row, subTree=None):

        # si no és un diccionari, significa que hem arribat a una fulla, busquem quin és el seu output
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