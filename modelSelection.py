import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def trainTestSplit(df, trainSize=0.8, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    sizeDf = df.shape[0]
    train = df.head(int(trainSize*sizeDf))
    test = df.tail(int((1-trainSize)*sizeDf))
    return train, test


def kFoldSplit(df, n_splits=10):
    splits = []
    splitSize = int(df.shape[0]/n_splits)
    for i in range(n_splits):
        split = df.head(splitSize*i+splitSize).tail(splitSize)
        splits.append(split)
    return splits


def crossValidation(model, df, n_splits=10, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    splits = kFoldSplit(df, n_splits)

    scores = {}
    accuracy = []
    precision = []
    recall = []
    f1Score = []
    for i in range(n_splits):
        train = []
        for j in range(n_splits):
            if j != i:
                train.append(splits[j])
        test = splits[i]
        train = pd.concat(train)

        model.fit(train)
        predictions = model.predict(test)
        groundTruth = test['target'].tolist()
        accuracy.append(accuracy_score(groundTruth, predictions))
        precision.append(precision_score(groundTruth, predictions))
        recall.append(recall_score(groundTruth, predictions))
        f1Score.append(f1_score(groundTruth, predictions))
    scores['test_accuracy'] = accuracy
    scores['test_precision'] = precision
    scores['test_recall'] = recall
    scores['test_f1Score'] = f1Score
    return scores