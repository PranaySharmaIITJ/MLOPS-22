"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

import pickle
# Standard scientific Python imports
import matplotlib.pyplot as plt

import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Importing rescale, resize, reshape
from skimage.transform import rescale, resize, downscale_local_mean 

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

digits = datasets.load_digits()

print("shape of data:", digits.images.shape)
print("shape of single image:", digits.images[0].shape, end="\n\n")


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

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




def digitsClassifier(gamma=0.001):
    #print("\n\ndata shape:", data.shape)
    #print("train-test split is:", 1-test_size,":",test_size, "\n\n")


    # printing the shapes
    # print(f"X_train:{X_train.shape}, X_test:{X_test.shape}, \ny_train:{y_train.shape},\
    #  y_test:{y_test.shape}, \nX_val:{X_val.shape},y_val:{y_val.shape}")

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the train subset
    predicted = clf.predict(X_train)
    a_train = round(accuracy_score(y_train, predicted), 4)
    p_train = round(precision_score(y_train, predicted, average='macro', zero_division=0), 4)
    r_train = round(recall_score(y_train, predicted, average='macro', zero_division=0), 4)
    f1_train = round(f1_score(y_train, predicted, average='macro', zero_division=0), 4)

    # Predict the value of the digit on the test subset
    # predicted = clf.predict(X_test)
    # a_test = round(accuracy_score(y_test, predicted), 4)
    # p_test = round(precision_score(y_test, predicted, average='macro', zero_division=0), 4)
    # r_test = round(recall_score(y_test, predicted, average='macro', zero_division=0), 4)
    # f1_test = round(f1_score(y_test, predicted, average='macro', zero_division=0), 4)    
    

    # Predict the value of the digit on the validation subset
    predicted = clf.predict(X_val)
    a_val = round(accuracy_score(y_val, predicted), 4)
    p_val = round(precision_score(y_val, predicted, average='macro', zero_division=0), 4)
    r_val = round(recall_score(y_val, predicted, average='macro', zero_division=0), 4)
    f1_val = round(f1_score(y_val, predicted, average='macro', zero_division=0), 4)


    return clf, [[a_train, p_train, r_train, f1_train], [a_val, p_val, r_val, f1_val]]
    

# take gamma as command line argument 0
# gammas = list(map(float, input().split()))
# print("gamma", gammas)


# checking for different gamma values
data = digits.images

# flatten the images
n_samples = len(data)
data = data.reshape((n_samples, -1))

# split data into 70% train and 30% (test + val) subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.3, shuffle=False)

# split test into test(15%) and val(15%)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)


gammas =  [0.000005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
results_gamma = []
best_f1 = -1
gamma_opt = -1
best_metrics = []


for gamma in gammas:
    clf, res = digitsClassifier(gamma=gamma)

    # skippping gammas where accuracy is less than 0.11
    if res[1][3] < 0.11:
        print(f"\nskipping for {gamma}\n")
        continue 

    results_gamma.append(res)

    print(f"\ngamma: {gamma}")
    for s,r in zip(["train", "val"], res):
        print(f"\t{s + ' scores:':<15} {r}")

    # validation f1 is in 3rd row, 4th col
    if res[1][3] > best_f1:
        best_f1 = res[1][3]
        gamma_opt = gamma
        best_metrics = res
        best_clf = clf

print("Saving the best model...")
save_file = open(f'/home/ada/codes/ML-Ops_Scikit/models/best_clf_{best_f1}.pkl', 'wb')
pickle.dump(best_clf, save_file)
save_file.close()


# should run only for best gamma 
predicted = best_clf.predict(X_test)
a_test = round(accuracy_score(y_test, predicted), 4)
p_test = round(precision_score(y_test, predicted, average='macro', zero_division=0), 4)
r_test = round(recall_score(y_test, predicted, average='macro', zero_division=0), 4)
f1_test = round(f1_score(y_test, predicted, average='macro', zero_division=0), 4)    

print(f"\n\nbest validation f1 score is {best_f1} for optimal gamma {gamma_opt}") 
print(f"\ttest scores:   {[a_test, p_test, r_test, f1_test]}\n\n")

# for s,r in zip(["train", "test", "val"], best_metrics):
#     print(f"\t{s + ' scores:':<15} {r}")

# remove for loop (for gamma in gammas) and take gamma as command line argument 



# loading the saved model
print("loading the model:")
load_file = open(f'/home/ada/codes/ML-Ops_Scikit/models/best_clf_{best_f1}.pkl', 'rb')
loaded_model = pickle.load(load_file)
print(loaded_model)

print("\nPredicting from loaded model:")
# should run only for best gamma 
predicted = loaded_model.predict(X_test)
a_test = round(accuracy_score(y_test, predicted), 4)
p_test = round(precision_score(y_test, predicted, average='macro', zero_division=0), 4)
r_test = round(recall_score(y_test, predicted, average='macro', zero_division=0), 4)
f1_test = round(f1_score(y_test, predicted, average='macro', zero_division=0), 4)    

print(f"\ttest scores:   {[a_test, p_test, r_test, f1_test]}\n\n")
