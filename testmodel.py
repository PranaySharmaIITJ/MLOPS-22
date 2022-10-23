import sys
sys.path.extend([".", ".."])

import os
import numpy as np
from sklearn import datasets
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# import utils
from utils import preprocess, data_split, get_scores, digitsClassifier, save_model


# random classifier
def get_random_scores(x, y):
    # predict randomly from 0 to 9
    predicted = np.random.randint(10, size=len(x))
    a = round(accuracy_score(y, predicted), 4)
    p = round(precision_score(y, predicted, average='macro', zero_division=0), 4)
    r = round(recall_score(y, predicted, average='macro', zero_division=0), 4)
    f1 = round(f1_score(y, predicted, average='macro', zero_division=0), 4)

    return [a, p, r, f1]


def test_model_writing():
    # data
    digits = datasets.load_digits()
    data_org = digits.images
    target = digits.target

    # preprocess
    data = preprocess(data_org)

    # split
    x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target)
    # train
    clf = digitsClassifier(x_train, y_train)

    path = "../models/test_save_model.pkl"
    save_model(clf, path)

    assert os.path.isfile(path) == True, f"model not saved at {path}!"


def test_small_data_overfit_checking():
    k = 50
    threshold = 0.9

    # data
    digits = datasets.load_digits()

    # take small sample
    data_org = digits.images[:k]
    target = digits.target[:k]

    # preprocess
    data = preprocess(data_org)

    # split
    x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target)
    # train
    clf = digitsClassifier(x_train, y_train)
     
    # get_scores returns list of [acc, precision, recall, f1]
    train_scores = get_scores(clf, x_train, y_train)
    random_scores = get_random_scores(x_train, y_train)

    print("train:", train_scores)
    print("random:", random_scores)

    assert train_scores[0] > threshold, "didn't overfit as per accuracy metric!"
    assert train_scores[3] > threshold, "didn't overfit as per f1 metric!"

    assert train_scores[0] > random_scores[0], "didn't overfit as per accuracy metric when compared with random model!"
    assert train_scores[3] > random_scores[3], "didn't overfit as per f1 metric when compared with random model!"
