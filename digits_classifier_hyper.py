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

# import utils.py
from utils import preprocess, data_split, get_scores, digitsClassifier, save_model, load_model, decisionClassifier

def tune_svm(x_train, y_train, x_val, y_val, x_test, y_test, gammas, C, kernels, save_best_model=False, verbose=False):
    results_dict = {}
    best_f1 = -1

    for gamma in gammas:
        for kernel in kernels:
            for c in C:
                print(f"gamma: {gamma} | kernel:{kernel} | c:{c}")
                clf = digitsClassifier(x_train, y_train, gamma, kernel, c)
                
                # predict on train and val sets and get scores
                res_train = get_scores(clf, x_train, y_train)
                res_val = get_scores(clf, x_val, y_val)
                res_test = get_scores(clf, x_test, y_test)

                # returns acc, p, r, f1
                res = [res_train[3], res_val[3], res_test[3]]
                results_dict[(gamma, kernel, c)] = res
                
                if verbose:
                    for s,r in zip(["train", "val", "test"], res):
                        print(f"\t{s + ' scores:':<15} {r}")
                    print("")

                # validation f1 is 4th elem
                if res_val[3] > best_f1:
                    best_f1 = res_val[3]
                    gamma_opt = gamma
                    kernel_opt = kernel
                    c_opt = c
                    best_clf = clf
                    best_metrics = res

                    if save_best_model:
                        print("\nSaving the best model for this run")
                        save_file = open(SAVE_MODEL_PATH + f"gamma-{gamma}_kernel-{kernel}_c-{c}.pkl", 'wb')
                        pickle.dump(clf, save_file)
                        save_file.close()

    print(f"\n\nbest validation f1 score is {best_f1} for optimal values of gamma {gamma_opt} | kernel:{kernel_opt} | c:{c_opt}") 
    print(f"\tbest scores on train/val/test: {res}")

    return results_dict, best_clf, gamma_opt, kernel_opt, c_opt



digits = datasets.load_digits()

print("shape of data:", digits.images.shape)
print("shape of single image:", digits.images[0].shape, end="\n\n")

data_org = digits.images
target = digits.target

data = preprocess(data_org)

SAVE_MODEL_PATH = "best_svm_model.joblib"

gammas = [0.001, 0.01]
kernels = ["rbf", "linear"]
C = [1, 10]
num_runs = 3

results = {}

for i in range(num_runs):
    print("\n", "="*20, "\n")
    print(f"run_{i}:")
    x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target, train_size=0.7, test_size=0.15, val_size=0.15, debug=False)

    results_dict, best_clf, gamma_opt, kernel_opt, c_opt = tune_svm(x_train, y_train, x_val, y_val, x_test, y_test, gammas, C, kernels, save_best_model=True)

    # running the classifiers on 5 train:test:val splits for comparison
    results[f"run_{i}"] = results_dict


    print("hyper_param, train_f1, val_f1, test_f1")
    for key, val in results[f"run_{i}"].items():
        print(key, val)


# display results for all runs
print("\n", "="*20, "\n")
keys = results[f"run_{0}"].keys()

print("hyper_param, [run1_f1: train/val/test] [run2_f1: train/val/test], [run3_f1: train/val/test]")
for k in keys:
    print(k, end="\t")
    for i in range(num_runs):
        print(results[f"run_{i}"][k], end="\t")
    print("")




