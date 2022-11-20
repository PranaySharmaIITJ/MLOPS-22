import sys
sys.path.extend([".", ".."])

import os
import numpy as np
from sklearn import datasets
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# import utils
#from src.utils import preprocess, data_split, get_scores, digitsClassifier, save_model
import numpy as np
import pickle

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn import tree

# Importing rescale, resize, reshape
from skimage.transform import rescale, resize, downscale_local_mean 

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def preprocess(data, scale_factor=1):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    print("\ndata:", data.shape)
    if scale_factor == 1:
        return data

    img_rescaled = []
    for img in data:
        img_rescaled.append(rescale(img, scale_factor, anti_aliasing=False))
    img_rescaled = np.array(img_rescaled)
    print("\nimg_rescaled:", img_rescaled.shape)
    return img_rescaled


def data_split(x, y, train_size=0.7, test_size=0.2, val_size=0.1, debug=True):
    # if train_size + test_size + val_size != 1:
    #     print("Invalid ratios: train:test:val split isn't 1!")
    #     return -1
    
    # print("\n from data split:", x.shape, y.shape,train_size, test_size, val_size)
    # split data into train and (test + val) subsets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(test_size + val_size))

    # split test into test and val
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=val_size/((test_size + val_size)))

    if debug:
        print("\n(x, y) shape:", x.shape, y.shape)
        print("(x_train, y_train) shape:", x_train.shape, y_train.shape)
        print("(x_test, y_test) shape:", x_test.shape, y_test.shape)
        print("(x_val, y_val) shape:", x_val.shape, y_val.shape, end="\n\n")

    return x_train, x_test, x_val, y_train, y_test, y_val


def get_scores(clf, x, y):
    # Predict the value of the digit on the train subset
    predicted = clf.predict(x)
    a = round(accuracy_score(y, predicted), 4)
    p = round(precision_score(y, predicted, average='macro', zero_division=0), 4)
    r = round(recall_score(y, predicted, average='macro', zero_division=0), 4)
    f1 = round(f1_score(y, predicted, average='macro', zero_division=0), 4)

    return [a, p, r, f1]


def digitsClassifier(x, y, gamma=0.001, kernel='rbf', C=1.0):
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma, C=C, kernel=kernel)
    # Learn the digits on the train subset
    clf.fit(x, y)
    return clf

def decisionClassifier(x, y):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x, y)
    return clf


def save_model(clf, path):
    print("\nSaving the best model...")
    save_file = open(path, 'wb')
    pickle.dump(clf, save_file)
    save_file.close()

def load_model(path):
    print("\nloading the model...")
    load_file = open(path, "rb")
    loaded_model = pickle.load(load_file)
    return 



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