import itertools
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score


class gcForest(object):

    def __init__(self, n_tree=51, n_cascadeRF=1, target_accuracy=None):

        setattr(self, 'n_layer', 1)
        setattr(self, 'n_cascadeRF', int(n_cascadeRF))
        setattr(self, 'n_tree', int(n_tree))
        setattr(self, 'target_accuracy', target_accuracy)

    def load_data(self, X, y, shape_1X, safe=False):

        for attr in ['X','y','shape_1X']:
            if safe and not hasattr(self, attr):
                setattr(self, attr, eval(attr))
            elif not safe:
                setattr(self, attr, eval(attr))

    def mg_scanning(self, X, y, shape_1X, window=None):

        if len(shape_1X) < 2:
            raise ValueError('shape parameter must be a tuple')

        self.load_data(X, y, shape_1X)
        mgs_pred_prob = []

        for wdw_size in window:
            wdw_pred_prob = self.window_slicing_pred_prob(window=wdw_size)
            mgs_pred_prob.append(wdw_pred_prob)

        setattr(self, 'mgs_features', np.concatenate(mgs_pred_prob, axis=1))

    def window_slicing_pred_prob(self, window, n_tree=30, min_samples=0.1):

        prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                     min_samples_split=min_samples)
        crf = RandomForestClassifier(n_estimators=n_tree, max_features=None,
                                     min_samples_split=min_samples)

        if self.shape_1X[1] > 1:
            print('Slicing Images...')
            sliced_X, sliced_y = self._window_slicing_img(window=window)
        else:
            print('Slicing Sequence...')
            sliced_X, sliced_y = self._window_slicing_sequence(window=window)

        print('Training Random Forests...')
        prf.fit(sliced_X, sliced_y)
        crf.fit(sliced_X, sliced_y)
        pred_prob_prf = prf.predict_proba(sliced_X)
        pred_prob_crf = crf.predict_proba(sliced_X)
        pred_prob = np.c_[pred_prob_prf, pred_prob_crf]

        return pred_prob.reshape([np.shape(self.X)[0],-1])

    def _window_slicing_img(self, window):

        dims = getattr(self, 'shape_1X')
        if any(s < window for s in dims):
             raise ValueError('window must be smaller than both dimensions for an image')

        sliced_imgs = []
        sliced_target = []
        refs = np.arange(0, (dims[0]-window)*dims[1], dims[1])

        iterx = list(range(dims[0]-window+1))
        itery = list(range(dims[1]-window+1))

        for img, ix, iy in itertools.product(enumerate(self.X), iterx, itery):
            rind = refs + ix + dims[0] * iy
            sliced_imgs.append(np.ravel([img[1][i:i+window] for i in rind]))
            sliced_target.append(self.y[img[0]])

        return np.asarray(sliced_imgs), np.asarray(sliced_target)

    def _window_slicing_sequence(self, window):

        dims = getattr(self, 'shape_1X')
        if dims[0] < window:
             raise ValueError('window must be smaller than the sequence dimension')

        sliced_sqce = []
        sliced_target = []

        for sqce in enumerate(self.X):
            slice_sqce = [sqce[1][i:i+window] for i in np.arange(dims[0]-window+1)]
            sliced_sqce.append(slice_sqce)
            sliced_target.append(np.repeat(self.y[sqce[0]], dims[0]-window+1))

        return np.reshape(sliced_sqce, [-1,window]), np.ravel(sliced_target)

    def cascade_forest(self, max_layers=5):

        accuracy = np.inf
#        new_feat = self.mgs_features

#        while accuracy > self.target_accuracy or self.n_layer <= max_layers:
#            new_feat, acc = self._cascade_layer(new_feat)

    def _cascade_layer(self, feat_arr, cv=3, min_samples=0.1):

        n_trees = getattr(self, 'n_tree')
        prf = RandomForestClassifier(n_estimators=n_trees, max_features='sqrt',
                                     min_samples_split=min_samples)
        crf = RandomForestClassifier(n_estimators=n_trees, max_features=None,
                                     min_samples_split=min_samples)

        prf_crf_pred, crf_pred =[], []
        for irf in range(n_trees):
            prf_crf_pred.append(cross_val_predict(prf, feat_arr, self.y, cv=cv, method='predict_proba'))
            prf_crf_pred.append(cross_val_predict(crf, feat_arr, self.y, cv=cv, method='predict_proba'))

        layer_pred_prob = np.mean(prf_crf_pred, axis=0)
        layer_pred = np.argmax(layer_pred_prob, axis=1)
        accuracy = accuracy_score(y_true=self.y, y_pred=layer_pred)

        return layer_pred_prob, accuracy




