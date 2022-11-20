"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, tree

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# import utils.py
from utils import preprocess, data_split, get_scores, digitsClassifier, save_model, load_model, decisionClassifier


def train_svm(x_train, y_train, x_val, y_val, x_test, y_test, gammas):
    results_gamma = []
    best_f1 = -1
    gamma_opt = -1
    thres_f1 = 0.11

    for gamma in gammas:
        clf = digitsClassifier(x_train, y_train, gamma)
        
        # predict on train and val sets and get scores
        res_train = get_scores(clf, x_train, y_train)
        res_val = get_scores(clf, x_val, y_val)

        # skippping gammas where accuracy is less than thres_f1
        # validation f1 is 4th elem
        if res_val[3] < thres_f1:
            print(f">> skipping for gamma: {gamma} as {res_val[3]} is less than {thres_f1}")
            continue 
        
        res = [res_train, res_val]
        results_gamma.append(res)

        print(f"\ngamma: {gamma}")
        for s,r in zip(["train", "val"], res):
            print(f"\t{s + ' scores:':<15} {r}")
        print("")

        # validation f1 is 4th elem
        if res_val[3] > best_f1:
            best_f1 = res_val[3]
            gamma_opt = gamma
            best_clf = clf
            best_metrics = res

    # should run only for best gamma 
    res_test = get_scores(best_clf, x_test, y_test)

    print(f"\n\nbest validation f1 score is {best_f1} for optimal gamma {gamma_opt}") 
    print(f"\ttrain scores:     {best_metrics[0]}")
    print(f"\tval scores:       {best_metrics[1]}")
    print(f"\ttest scores:    {res_test}\n\n")

    return gamma_opt


def train_decisionTree(x_train, y_train, x_val, y_val, x_test, y_test, depths):
    results_depth = []
    best_f1 = -1
    depth_opt = -1

    for depth in depths:
        clf = decisionClassifier(x_train, y_train)
        # predict on train and val sets and get scores
        res_train = get_scores(clf, x_train, y_train)
        res_val = get_scores(clf, x_val, y_val)

        res = [res_train, res_val]
        results_depth.append(res)

        print(f"\ndepth: {depth}")
        for s,r in zip(["train", "val"], res):
            print(f"\t{s + ' scores:':<15} {r}")
        print("")


        if res_val[3] > best_f1:
            best_f1 = res_val[3]
            depth_opt = depth
            best_clf = clf
            best_metrics = res

    # should run only for best depth 
    res_test = get_scores(best_clf, x_test, y_test)

    print(f"\n\nbest validation f1 score is {best_f1} for depth {depth_opt}") 
    print(f"\ttrain scores:     {best_metrics[0]}")
    print(f"\tval scores:       {best_metrics[1]}")
    print(f"\ttest scores:      {res_test}\n\n")

    # print("max_depth:", clf.tree_.max_depth)
    return depth_opt


def train_classifier(x_train, y_train, x_val, y_val, x_test, y_test, clf_type, clf_param):
    if clf_type == 'svm':
        clf = svm.SVC(gamma=clf_param)
    elif clf_type == 'decision':
        clf = tree.DecisionTreeClassifier(max_depth=clf_param)
    else:
        print("Not a valid classifier name.")
        return -1

    clf.fit(x_train, y_train)
    
    # predict on train and val sets and get scores
    res_train = get_scores(clf, x_train, y_train)
    res_val = get_scores(clf, x_val, y_val)
    res_test = get_scores(clf, x_test, y_test)
    
    return [res_train, res_val, res_test]



digits = datasets.load_digits()

print("shape of data:", digits.images.shape)
print("shape of single image:", digits.images[0].shape, end="\n\n")

data_org = digits.images
target = digits.target

data = preprocess(data_org)
x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target, debug=False)

# print("\n***tuning gamma parameter for svm classifier***\n")
# gammas =  [0.000005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
# opt_gamma = train_svm(x_train, y_train, x_val, y_val, x_test, y_test, gammas)

# print("\n***tuning max_depth parameter for decision tree classifier***\n")
# depths = [10, 12, 14, 16, 18, 20] 
# opt_depth = train_decisionTree(x_train, y_train, x_val, y_val, x_test, y_test, depths)


# running the classifiers on 5 train:test:val splits for comparison
results_svm = []
results_dt = []

x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target, train_size=0.6, test_size=0.3, val_size=0.1, debug=False)

print("\n***tuning gamma parameter for svm classifier***\n")
gammas =  [0.000005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
opt_gamma = train_svm(x_train, y_train, x_val, y_val, x_test, y_test, gammas)

print("\n***tuning max_depth parameter for decision tree classifier***\n")
depths = [10, 12, 14, 16, 18, 20] 
opt_depth = train_decisionTree(x_train, y_train, x_val, y_val, x_test, y_test, depths)

print(f"optimal gamma:{opt_gamma}   optimal depth:{opt_depth}\n")
results_svm.append(train_classifier(x_train, y_train, x_val, y_val, x_test, y_test, clf_type='svm', clf_param=opt_gamma))

results_dt.append(train_classifier(x_train, y_train, x_val, y_val, x_test, y_test,clf_type='decision', clf_param=opt_depth))


# converting results into numpy array
results_svm_np = np.array([elem[2] for elem in results_svm])
results_dt_np = np.array([elem[2] for elem in results_dt])

print("\n***Printing test scores of both classifiers on 5 train:val:test sets***\n")

print("                         SVM                         Decision Tree\n")
for i in range(5):
    print(f"split {i}:     {results_svm_np[i].round(4)}    {results_dt_np[i].round(4)}\n")

# mean and sd
print(f"mean:        {results_svm_np.mean(axis=0).round(4)}     {results_dt_np.mean(axis=0).round(4)}")

print(f"std dev:     {results_svm_np.std(axis=0).round(4)}     {results_dt_np.std(axis=0).round(4)}\n")
