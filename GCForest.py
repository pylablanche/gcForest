import itertools
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score


class gcForest(object):

    def __init__(self, n_tree=51, n_pseudoRF=1, n_completeRF=1, target_accuracy=None):

        setattr(self, 'n_layer', 1)
        setattr(self, 'n_pseudoRF', int(n_pseudoRF))
        setattr(self, 'n_completeRF', int(n_completeRF))
        setattr(self, 'n_tree', int(n_tree))
        setattr(self, 'target_accuracy', target_accuracy)

    def load_data(self, X, y, shape_1X, safe=False):

        for attr in ['X','y','shape_1X']:
            if safe and not hasattr(self, attr):
                setattr(self, attr, eval(attr))
            elif not safe:
                setattr(self, attr, eval(attr))

    def mg_scanning(self, X, y, shape_1X, window=None):

        if len(shape_1X)<2:
            raise ValueError('shape parameter must be a tuple')

        self.load_data(X, y, shape_1X)
        mg_pred_prob = []

        for wdw_size in window:
            wdw_pred_prob = self.window_slicing_pred_prob(window=wdw_size)
            mg_pred_prob.append(wdw_pred_prob)

        setattr(self, 'mgs_features', np.ravel(mg_pred_prob))

    def window_slicing_pred_prob(self, window, n_tree=30, min_samples=0.1):

        pred_prob_prf, pred_prob_crf = [], []
        prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                     min_samples_split=min_samples)
        crf = RandomForestClassifier(n_estimators=n_tree, max_features=None,
                                     min_samples_split=min_samples)

        if self.shape_1X[1]>1:
            sliced_X, sliced_y = self._window_slicing_img(window=window)
            prf.fit(sliced_X, sliced_y)
            crf.fit(sliced_X, sliced_y)
            for sliceX in sliced_X:
                pred_prob_prf.append(prf.predict_proba(sliceX))
                pred_prob_crf.append(crf.predict_proba(sliceX))
        else:
            sliced_X, sliced_y = self._window_slicing_img(window=window)
            prf.fit(sliced_X, sliced_y)
            crf.fit(sliced_X, sliced_y)
            for sliceX in sliced_X:
                pred_prob_prf.append(prf.predict_proba(sliceX))
                pred_prob_crf.append(crf.predict_proba(sliceX))

        return np.concatenate([pred_prob_prf,pred_prob_crf])

    def _window_slicing_img(self, window):

        if any(s < window for s in self.shape_1X):
             raise ValueError('window must be smaller than both dimensions for an image')

        sliced_imgs = []
        sliced_target = []
        refs = np.arange(0, (self.shape_1X[0]-window)*self.shape_1X[1], self.shape_1X[1])

        iterx = list(range(self.shape_1X[0]-window+1))
        itery = list(range(self.shape_1X[1]-window+1))

        for img, ix, iy in itertools.product(enumerate(self.X), iterx, itery):
            rind = refs + ix + self.shape_1X[0] * iy
            sliced_imgs.append(np.ravel([img[1][i:i+window] for i in rind]))
            sliced_target.append(self.y[img[0]])

        return np.c_[sliced_imgs], np.c_[sliced_target]

    def _window_slicing_sequence(self, window):

        if any(s < window for s in self.shape_1X):
             raise ValueError('window must be smaller than the sequence dimension')

        sliced_sqce = []
        sliced_target = []

        for sqce in enumerate(self.X):
            slice_sqce = [sqce[1][i:i+window] for i in np.arange(self.shape_1X[0]-window+1)]
            sliced_sqce.append(np.ravel(slice_sqce))
            sliced_target.append(self.y[sqce[0]])

        return np.c_[sliced_sqce], np.c_[sliced_target]


#    def cascade_forest(self, X=None, y=None, shape_1X=None, max_layers=5):
#
#        self.load_data(X, y, shape_1X, safe=True)
#        cf_pred_proba = []
#
#        if hasattr(self, 'mgs_features'):
#            features = getattr(self, 'mgs_features')
#        else:
#            features = getattr(self, 'X')
#
#        while accuracy > self.target_accuracy or self.n_layer <= max_layers:
#            for i in range(self.n_pseudoRF):
#                prf = RandomForestClassifier(n_estimators=self.n_tree, max_features='sqrt')
#                prf.fit(features,self.y)
#            for i in range(self.n_completeRF):
#                crf = RandomForestClassifier(n_estimators=self.n_tree, max_features=None)
#                crf.fit(features,self.y)

#    def _cascade_layer(self, X_input, y):

#        pseudoRF_pred_proba = []
#        completeRF_pred_proba = []
#        new_features = np.concatenate([self.X, pseudoRF_pred_proba, completeRF_pred_proba])

#        return new_features, accuracy

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
