import numpy as np
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#import itertools

class gcForest(object):

    def __init__(self, n_tree=51, n_pseudoRF=1, n_completeRF=1, target_accuracy=None):

        self.n_layer = 1
        self.n_pseudoRF = int(n_pseudoRF)
        self.n_completeRF = int(n_completeRF)
        self.n_tree = int(n_tree)
        self.tgt_acc = target_accuracy

#    def mg_scanning(self):

#    def cascade_forest(self):


    def _window_slicing(self, X, window):

        sliced_X = []
        len_iter = np.shape(X)[0] - window[0] + int(1)
        for i in range(len_iter):
            sliced_X.append(X[i:i+window[0]])

        return np.asarray(sliced_X)

    def _pseudoRF(self, X, y):

        prf = RandomForestClassifier(n_estimators=self.n_tree, max_features='sqrt')
        score = cross_val_score(estimator=prf, X=X, y=y, cv=3)
        pred = prf.predict(X)

        return pred, score

    def _completeRF(self, X, y):

        crf = RandomForestClassifier(n_estimators=self.n_tree, max_features=None)
        score = cross_val_score(estimator=crf, X=X, y=y, cv=3)
        pred = crf.predict(X)

        return pred, score
