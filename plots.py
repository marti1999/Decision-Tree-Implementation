import statistics

import seaborn as sns
from matplotlib import pyplot as plt


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


def showMetricTwoHeuristicsPlots(crossValScoresByMetric, metrics=None, legend=None, title="Comparació heurístiques, probabilistic approach = false"):
    for metric in metrics:

        crossValScoresByN = []
        for cvs in crossValScoresByMetric:
            crossValScoresByN.append(cvs[metric])

        for cvs in crossValScoresByN:
            for k in cvs.keys():
                cvs[k] = statistics.mean(cvs[k])

        plt.title(title)

        for cvs in crossValScoresByN:
            plt.plot(list(cvs.keys()), list(cvs.values()))

        plt.ylim([0.5, 1])
        plt.xlabel("Number of intervals")
        plt.ylabel("Average " + metric)
        plt.legend(legend)
        plt.show()


def showBarPlot(crossValScoresByMetric, metrics=None, legend=None, title="Comparació heurístiques, probabilistic approach = false"):
    for metric in metrics:

        crossValScoresByN = []
        for cvs in crossValScoresByMetric:
            crossValScoresByN.append(cvs[metric])

        scores = []
        for cvs in crossValScoresByN:
            for k in cvs.keys():
                scores.append(statistics.mean(cvs[k]))

        xlabels = legend
        yvalues = scores

        x_pos = [i for i, _ in enumerate(xlabels)]
        plt.bar(x_pos, yvalues)
        plt.xlabel("Initial invertals")
        plt.ylabel(metric)
        plt.title(title)
        plt.xticks(x_pos, xlabels)
        plt.ylim([0.6, 0.9])
        plt.show()


def recursive_print_dict( d, indent = 0 ):
    for k, v in d.items():
        if isinstance(v, dict):
            print("\t" * indent, f"{k}:")
            recursive_print_dict(v, indent+1)
        else:
            print("\t" * indent, f"{k}:{v}")


def plotNull(df):
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()


def plotConfusionMatrix(y_pred, y_test):
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    f = sns.heatmap(cm, annot=True, fmt='f')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()
