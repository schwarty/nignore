import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import _ConstantPredictor

from reporting_utils import ClassificationReporterMixin
from reporting_utils import NiimgReporterMixin


def get_estimated(estimator, name='coef_'):
    if hasattr(estimator, 'estimators_'):
        estimated_ = _get_estimated_meta(estimator, name, )
    elif hasattr(estimator, 'best_estimator_'):
        estimated_ = _get_estimated_grid_search(estimator, name)
    elif hasattr(estimator, 'steps'):
        estimated_ = _get_estimated_pipeline(estimator, name)
    elif hasattr(estimator, name):
        estimated_ = _get_estimated_base(estimator, name)
    elif isinstance(estimator, _ConstantPredictor):
        estimated_ = None
    else:
        raise Exception('Estimator %s not supported' % estimator)
    return estimated_


def _get_estimated_grid_search(grid_search, name):
    estimator = grid_search.best_estimator_
    return get_estimated(estimator, name)


def _get_estimated_pipeline(pipeline, name):
    estimator = pipeline.steps[-1][1]
    estimated_ = get_estimated(estimator, name)
    if len(pipeline.steps) == 1:
        return estimated_
    else:
        estimated_t = np.array(estimated_, copy=True)
        for name, step in pipeline.steps[:-1][::-1]:
            estimated_t = step.inverse_transform(estimated_t)
        return estimated_t
    return estimated_


def _get_estimated_base(estimator, name):
    if hasattr(estimator, name):
        return getattr(estimator, name)
    else:
        raise Exception(
            'BaseEstimator %s does not '
            'have an attribute called %s' % (estimator, name))


def _get_estimated_meta(estimator, name):
    estimated_ = []
    shape = None

    for estimator in estimator.estimators_:
        estimated = get_estimated(estimator, name=name)
        if shape is None and estimated is not None:
            shape = estimated.shape
        estimated_.append(estimated)

    for i, estimated in enumerate(estimated_):
        if estimated is None:
            estimated_[i] = np.zeros(shape)

    return np.vstack(estimated_)


class DecoderMixin(object):

    def _get_niimgs(self):
        estimated_ = get_estimated(self.estimator, self.estimated_name)

        if len(estimated_.shape) == 2:
            self.niimgs_ = [self.masker.inverse_transform(estimated)
                            for estimated in estimated_]
        self.niimg_ = self.masker.inverse_transform(estimated_)

    def _get_scores(self, y_true, y_pred):
        self.scores_ = []

        if len(y_true.shape) == 2:
            Y_true = y_true
            Y_pred = y_pred

            if self.labels is None:
                self.labels = [''] * len(self.niimgs_)

            for label, y_true, y_pred in zip(self.labels, Y_true.T, Y_pred.T):
                self.scores_.append(
                    (label, precision_recall_fscore_support(y_true, y_pred)))
        else:
            self.scores_ = (
                self.labels,
                precision_recall_fscore_support(y_true, y_pred))


class Decoder(DecoderMixin, ClassificationReporterMixin, NiimgReporterMixin):

    def __init__(self, estimator, masker, estimated_name='coef_',
                 report_params=None):
        self.estimator = estimator
        self.masker = masker
        self.estimated_name = estimated_name
        self.report_params = report_params

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        self._get_niimgs()
        return self

    def predict(self, X):
        self._niimg_report()
        return self.estimator.predict(X)

    def score(self, X, y):
        y_true, y_pred = y, self.predict(X)
        self._get_scores(y_true, y_pred)
        return accuracy_score(y_true, y_pred)
