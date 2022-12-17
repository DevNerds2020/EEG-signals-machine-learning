import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import random
import os


def main():
    seed = 57

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    x = pickle.load(open('x.pkl', 'rb'))
    y = pickle.load(open('y.pkl', 'rb'))

    x_normal = np.concatenate((x[:300], x[400:]), axis=0)
    x_seizure = x[300:400]
    print(x_normal.shape)
    print(x_seizure.shape)
    sampling_freq = 173.6  # based on info from website

    b, a = butter(3, [0.5, 40], btype='bandpass', fs=sampling_freq)

    x_normal_filtered = np.array([lfilter(b, a, x_normal[ind, :]) for ind in range(x_normal.shape[0])])
    x_seizure_filtered = np.array([lfilter(b, a, x_seizure[ind, :]) for ind in range(x_seizure.shape[0])])
    print(x_normal.shape)
    print(x_seizure.shape)

    x_normal = x_normal_filtered
    x_seizure = x_seizure_filtered

    x = np.concatenate((x_normal, x_seizure))
    y = np.concatenate((np.zeros((400, 1)), np.ones((100, 1))))
    # add feature extraction for signals here
    # first add variance, mean, max, min, median, skewness, kurtosis, and standard deviation
    # then add frequency domain features
    var = np.var(x, axis=1)
    mean = np.mean(x, axis=1)
    max = np.max(x, axis=1)
    min = np.min(x, axis=1)
    median = np.median(x, axis=1)
    # skewness = np.mean((x - mean) ** 3, axis=1) / np.mean((x - mean) ** 2, axis=1) ** 1.5
    # kurtosis = np.mean((x - mean) ** 4, axis=1) / np.mean((x - mean) ** 2, axis=1) ** 2
    std = np.std(x, axis=1)
    #add entropy feature
    entropy = np.sum(-x * np.log(x), axis=1)
    #LBP based features
    #time domain features
    #frequency domain features


    # now test features
    newX = np.concatenate((var.reshape(-1, 1), mean.reshape(-1, 1), max.reshape(-1, 1), min.reshape(-1, 1),
                            median.reshape(-1, 1), std.reshape(-1, 1)), axis=1)
    #train test split
    x_train, x_test, y_train, y_test = train_test_split(newX, y, random_state=seed, test_size=0.2)
    # svc
    custom_clf = SVC(kernel='linear')
    custom_clf.fit(x_train, y_train)
    y_pred = custom_clf.predict(x_test)
    print("###=>>>>>(svc)((accuracy_score)-(recall_score)-(precision_score))",accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred))

    # random forest
    custom_clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    custom_clf.fit(x_train, y_train)
    y_pred = custom_clf.predict(x_test)
    print("###=>>>>>(random forest)((accuracy_score)-(recall_score)-(precision_score))",accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred))
    

    # knn
    custom_clf = KNeighborsClassifier(n_neighbors=3)
    custom_clf.fit(x_train, y_train)
    y_pred = custom_clf.predict(x_test)
    print("###=>>>>>(knn)((accuracy_score)-(recall_score)-(precision_score))",accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed, test_size=0.2)

    print(x_test.shape)

    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)


    y_pred = clf.predict(x_test)

    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
