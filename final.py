'''
Note: load the best svm_model from the disc -- do not train it during the test case.
1. Add one positive test case per class. For example, "def test_digit_correct_0" function tests if the prediction of an actual digit-0 sample indeed 0 or not, i.e. `assert prediction==0`. (Total of 10 such test cases)
2. [Bonus] Add a test case that checks that accuracy on each class is greater than a certain threshold. i.e. `assert acc_digit[0] > min_acc_req`
'''

import sys

import numpy
sys.path.extend([".", ".."])

# import utils
#from src.utils import preprocess, data_split
import pickle
from sklearn import datasets
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

svm_best_model_path = 'best_svm_model.joblib'
decision_best_model_path = 'best_svm_model.joblib'

print("\nloading the svm_model...")
load_file = open(svm_best_model_path, "rb")
svm_model = pickle.load(load_file)

print("\nloading the decision_model...")
load_file = open(svm_best_model_path, "rb")
decision_model = pickle.load(load_file)

# data
digits = datasets.load_digits()
data_org = digits.images
target = digits.target

# preprocess
data = preprocess(data_org)

# split
x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target)
samples = []
targets = []

print(y_test[:10])

# making samples
for i in range(10):
    idx_i = [y_test==i]
    samples.append(x_test[idx_i][0])
    targets.append(i)

samples = numpy.array(samples)
print(f"len samples:{len(samples)}")
print(f"targets:{targets}")


# svm
def test_digit_correct_0():
    prediction = svm_model.predict(samples[0].reshape(1, -1))
    print(samples[0].reshape(1, -1))
    print(prediction)
    print(prediction[0], targets[0])
    assert prediction[0] == targets[0], f"Prediction incorrect"

def test_digit_correct_1():
    prediction = svm_model.predict(samples[1].reshape(1, -1))
    assert prediction[0] == targets[1], f"Prediction incorrect"

def test_digit_correct_2():
    prediction = svm_model.predict(samples[2].reshape(1, -1))
    assert prediction[0] == targets[2], f"Prediction incorrect"

def test_digit_correct_3():
    prediction = svm_model.predict(samples[3].reshape(1, -1))
    assert prediction[0] == targets[3], f"Prediction incorrect"

def test_digit_correct_4():
    prediction = svm_model.predict(samples[4].reshape(1, -1))
    assert prediction[0] == targets[4], f"Prediction incorrect"

def test_digit_correct_5():
    prediction = svm_model.predict(samples[5].reshape(1, -1))
    assert prediction[0] == targets[5], f"Prediction incorrect"

def test_digit_correct_6():
    prediction = svm_model.predict(samples[6].reshape(1, -1))
    assert prediction[0] == targets[6], f"Prediction incorrect"

def test_digit_correct_7():
    prediction = svm_model.predict(samples[7].reshape(1, -1))
    assert prediction[0] == targets[7], f"Prediction incorrect"

def test_digit_correct_8():
    prediction = svm_model.predict(samples[8].reshape(1, -1))
    assert prediction[0] == targets[8], f"Prediction incorrect"

def test_digit_correct_9():
    prediction = svm_model.predict(samples[9].reshape(1, -1))
    assert prediction[0] == targets[9], f"Prediction incorrect"



# decision
def test_decision_digit_correct_0():
    prediction = decision_model.predict(samples[0].reshape(1, -1))
    print(samples[0].reshape(1, -1))
    print(prediction)
    print(prediction[0], targets[0])
    assert prediction[0] == targets[0], f"Prediction incorrect"

def test_decision_digit_correct_1():
    prediction = decision_model.predict(samples[1].reshape(1, -1))
    assert prediction[0] == targets[1], f"Prediction incorrect"

def test_decision_digit_correct_2():
    prediction = decision_model.predict(samples[2].reshape(1, -1))
    assert prediction[0] == targets[2], f"Prediction incorrect"

def test_decision_digit_correct_3():
    prediction = decision_model.predict(samples[3].reshape(1, -1))
    assert prediction[0] == targets[3], f"Prediction incorrect"

def test_decision_digit_correct_4():
    prediction = decision_model.predict(samples[4].reshape(1, -1))
    assert prediction[0] == targets[4], f"Prediction incorrect"

def test_decision_digit_correct_5():
    prediction = decision_model.predict(samples[5].reshape(1, -1))
    assert prediction[0] == targets[5], f"Prediction incorrect"

def test_decision_digit_correct_6():
    prediction = decision_model.predict(samples[6].reshape(1, -1))
    assert prediction[0] == targets[6], f"Prediction incorrect"

def test_decision_digit_correct_7():
    prediction = decision_model.predict(samples[7].reshape(1, -1))
    assert prediction[0] == targets[7], f"Prediction incorrect"

def test_decision_digit_correct_8():
    prediction = decision_model.predict(samples[8].reshape(1, -1))
    assert prediction[0] == targets[8], f"Prediction incorrect"

def test_decision_digit_correct_9():
    prediction = decision_model.predict(samples[9].reshape(1, -1))
    assert prediction[0] == targets[9], f"Prediction incorrect"