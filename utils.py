import numpy as np
import pickle
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

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


def data_split(x, y, train_size=0.7, test_size=0.15, val_size=0.15):
    if train_size + test_size + val_size != 1:
        print("Invalid ratios: train:test:val split isn't 1!")
        return -1
    
    # split data into train and (test + val) subsets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(test_size + val_size), shuffle=False)

    # split test into test and val
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=val_size/((test_size + val_size)), shuffle=False)

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


def digitsClassifier(x, y, gamma=0.001):
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma)
    # Learn the digits on the train subset
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
    return loaded_model
