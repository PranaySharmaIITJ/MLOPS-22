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
from sklearn import datasets

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
from utils import preprocess, data_split, get_scores, digitsClassifier, save_model, load_model

digits = datasets.load_digits()

print("shape of data:", digits.images.shape)
print("shape of single image:", digits.images[0].shape, end="\n\n")

data_org = digits.images
target = digits.target

data = preprocess(data_org)
x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target)

gammas =  [0.000005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
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


# saving model
model_path = f'/home/ada/codes/ML-Ops_Scikit/models/best_clf_{best_f1}.pkl'
save_model(best_clf, model_path)

# should run only for best gamma 
res_test = get_scores(best_clf, x_test, y_test)

print(f"\n\nbest validation f1 score is {best_f1} for optimal gamma {gamma_opt}") 
print(f"\ttest scores:    {res_test}\n\n")


# loading the saved model
loaded_model = load_model(model_path)
print(loaded_model)

# predicting from loaded model
print("\npredicting from loaded model:") 
res_test = get_scores(loaded_model, x_test, y_test)
print(f"\ttest scores:    {res_test} \n\n")
