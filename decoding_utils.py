import os
import tempfile

import nibabel as nb
import numpy as np
import pylab as pl

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import _ConstantPredictor

from reporting_utils import ClassificationReporterMixin
from reporting_utils import NiimgReporterMixin


def get_estimator_weights(estimator, weights_attr='coef_', transformer=None):

    if hasattr(estimator, 'estimators_'):
        coef_ = _get_meta_estimator_weights(estimator, weights_attr, )
    else:
        coef_ = _get_estimator_weights(estimator, weights_attr)

    if transformer is not None:
        return transformer.inverse_transform(coef_)
    return coef_


def _get_estimator_weights(estimator, weights_attr='coef_'):

    if hasattr(estimator, 'best_estimator_'):
        coef_ = _get_grid_search_weights(estimator, weights_attr)
    elif hasattr(estimator, 'steps'):
        coef_ = _get_pipeline_weights(estimator, weights_attr)
    elif hasattr(estimator, weights_attr):
        coef_ = _get_base_estimator_weights(estimator, weights_attr)
    elif isinstance(estimator, _ConstantPredictor):
        coef_ = None
    else:
        raise Exception('Estimator %s not supported' % estimator)

    return coef_


def _get_grid_search_weights(grid_search, weights_attr):
    estimator = grid_search.best_estimator_

    return _get_estimator_weights(estimator, weights_attr)


def _get_pipeline_weights(pipeline, weights_attr):
    estimator = pipeline.steps[-1][1]

    coef_ = _get_estimator_weights(estimator, weights_attr)

    if len(pipeline.steps) == 1:
        return coef_
    else:
        return pipeline.inverse_transform(coef_)


def _get_base_estimator_weights(estimator, weights_attr):
    if hasattr(estimator, weights_attr):
        return getattr(estimator, weights_attr)
    else:
        raise Exception('BaseEstimator %s does not '
                        'have an attribute called %s' % (estimator,
                                                         weights_attr))


def _get_meta_estimator_weights(estimator, weights_attr):
    W_ = []
    shape = None

    for est in estimator.estimators_:
        w = _get_estimator_weights(est, weights_attr=weights_attr)
        if shape is None and w is not None:
            shape = w.shape
        W_.append(w)

    for i, w in enumerate(W_):
        if w is None:
            W_[i] = np.zeros(shape)

    return np.vstack(W_)


class DecoderMixin(object):

    def _get_niimgs(self):
        coef = get_estimator_weights(self.estimator,
                                     self.weights_attr, self.transformer)

        if len(coef.shape) == 2:
            self.niimgs_ = [self.masker.inverse_transform(w) for w in coef]
        self.niimg_ = self.masker.inverse_transform(coef)

    def _get_scores(self):
        self.scores_ = []
        if len(self.y_true_.shape) == 2:
            Y_true = self.y_true_
            Y_pred = self.y_pred_

            if self.labels is None:
                self.labels = [''] * len(self.niimgs_)

            for label, y_true, y_pred in zip(self.labels, Y_true.T, Y_pred.T):
                self.scores_.append(
                    (label, precision_recall_fscore_support(y_true, y_pred)))


class Decoder(DecoderMixin, ClassificationReporterMixin, NiimgReporterMixin):

    def __init__(self, estimator, masker,
                 weights_attr='coef_', transformer=None,
                 report_params=None):
        self.estimator = estimator
        self.masker = masker
        self.weights_attr = weights_attr
        self.transformer = transformer
        self.report_params = report_params

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        self._get_niimgs()
        return self

    def predict(self, X):
        self._niimg_report()
        return self.estimator.predict(X)

    def score(self, X, y):
        self.y_true_, self.y_pred_ = y, self.predict(X)
        self._get_scores()
        return accuracy_score(self.y_true_, self.y_pred_)
