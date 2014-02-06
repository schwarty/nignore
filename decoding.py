import warnings

import numpy as np
import nibabel as nb

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import _ConstantPredictor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import WardAgglomeration
from sklearn.cross_validation import ShuffleSplit
from sklearn import clone
from joblib import Memory, Parallel, delayed

from nilearn.input_data import NiftiMasker

from reporting import Reporter


def get_estimated(estimator, name='coef_'):
    if hasattr(estimator, 'estimators_'):
        estimated_ = _get_meta(estimator, name, )
    elif hasattr(estimator, 'best_estimator_'):
        estimated_ = _get_grid_search(estimator, name)
    elif hasattr(estimator, 'steps'):
        estimated_ = _get_pipeline(estimator, name)
    elif hasattr(estimator, name.split('.')[0]):
        estimated_ = _get_base(estimator, name)
    elif isinstance(estimator, _ConstantPredictor):
        estimated_ = None
    else:
        raise Exception('Estimator %s not supported' % estimator)
    return estimated_


def _get_grid_search(grid_search, name):
    estimator = grid_search.best_estimator_
    return get_estimated(estimator, name)


def _get_pipeline(pipeline, name):
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


def _get_base(estimator, name):
    if hasattr(estimator, name.split('.')[0]):
        return reduce(getattr, name.split('.'), estimator)
    else:
        raise Exception(
            'BaseEstimator %s does not '
            'have an attribute called %s' % (estimator, name))


def _get_meta(estimator, name):
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


def squeeze_niimg(niimg):
    return nb.Nifti1Image(niimg.get_data().squeeze(),
                          affine=niimg.get_affine())


class Decoder(BaseEstimator):

    def __init__(self, estimator=LinearSVC(),
                 masker=NiftiMasker(),
                 labelizer=LabelEncoder(),
                 reporter=Reporter(),
                 estimated_name='coef_'):
        self.estimator = estimator
        self.masker = masker
        self.labelizer = labelizer
        self.reporter = reporter
        self.estimated_name = estimated_name

    def fit(self, niimgs, target_names):
        y = self.labelizer.fit_transform(target_names)
        X = self.masker.fit_transform(niimgs, y)
        self.estimator.fit(X, y)
        self._plot_report()
        return self

    def predict(self, niimgs):
        X = self.masker.transform(niimgs)
        self.y_pred_ = self.estimator.predict(X)
        return self.labelizer.inverse_transform(self.y_pred_)

    def score(self, niimgs, target_names):
        y = self.labelizer.transform(target_names)
        self.y_true_, y_pred = y, self.predict(niimgs)
        self._eval_report()
        return accuracy_score(self.y_true_, self.y_pred_)

    def _plot_report(self):
        estimated_ = get_estimated(self.estimator, self.estimated_name)
        if estimated_.ndim == 2:
            self.niimgs_ = [squeeze_niimg(self.masker.inverse_transform(val))
                            for val in estimated_]
        else:
            self.niimgs_ = [squeeze_niimg(
                self.masker.inverse_transform(estimated_))]

        for title, niimg in zip(self.labelizer.classes_, self.niimgs_):
            self.reporter.plot_map(niimg, title)
        self.reporter.plot_contours(self.niimgs_, self.labelizer.classes_)
        self.reporter.plot_labels(self.niimgs_, self.labelizer.classes_)

    def _eval_report(self):
        self.reporter.eval_classif(self.y_true_, self.y_pred_,
                                   self.labelizer.classes_)
