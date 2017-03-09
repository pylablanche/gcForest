import numpy as np
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import itertools

class gcForest(object):

    def __init__(self, n_tree=51, n_pseudoRF=1, n_completeRF=1, target_accuracy=None):

        self.n_layer = 1
        self.n_pseudoRF = int(n_pseudoRF)
        self.n_completeRF = int(n_completeRF)
        self.n_tree = int(n_tree)
        self.tgt_acc = target_accuracy

    def mg_scanning(self, X, y, window=None, shape=None):

        self.X = X
        self.shape_data = shape
        self.y = y

        for wdw_size in window:
            sliced_X = self._window_slicing(wdw_size)
#            pred_pseudoRF = self._pseudoRF_mgs(sliced_X, self.y)

        return sliced_X

    def _window_slicing_img(self, window):

        if any(s < window for s in self.shape_data):
             raise ValueError('window must be smaller than both dimensions for an image')

        sliced_img = []
        refs = np.arange(0, (self.shape_data[0]-window)*self.shape_data[1], self.shape_data[1])

        iterx = list(range(self.shape_data[0]-window+1))
        itery = list(range(self.shape_data[1]-window+1))

        for ix, iy in itertools.product(iterx, itery):
            rind = refs+ix+8*iy
            sliced_img.append(np.ravel([self.X[i:i+4] for i in rind]))

        return sliced_img

#    def _pseudoRF_mgs(self, X, y):
#
#        prf = RandomForestClassifier(n_estimators=self.n_tree, max_features='sqrt')
#        prf.fit(sliced_X, np.c_[self.y])
#        pred_proba = prf.predict_proba(self.X)
#
#        return pred_proba

#    def cascade_forest(self, X, y):

#        if not self.X :
#            self.X = X
#        if not self.y :
#            self.y = y

#    def _cascade_layer(self, X_input, y):

#        pseudoRF_pred_proba = []
#        completeRF_pred_proba = []

#        for

#        new_features = np.concatenate([self.X, pseudoRF_pred_proba, completeRF_pred_proba])

#        return new_features

#    def _pseudoRF_casc(self, X, y):
#
#        prf = RandomForestClassifier(n_estimators=self.n_tree, max_features='sqrt')
#        prf.fit(X,y)
#        score = cross_val_score(estimator=prf, X=X, y=y, cv=3)
#        pred = prf.predict_proba(X)
#
#        return pred, score


#    def _completeRF_casc(self, X, y):
#
#        crf = RandomForestClassifier(n_estimators=self.n_tree, max_features=None)
#        score = cross_val_score(estimator=crf, X=X, y=y, cv=3)
#        pred = crf.predict_proba(X)
#
#        return pred, score
