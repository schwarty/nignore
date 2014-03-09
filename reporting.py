import matplotlib
matplotlib.use('Agg')

import sys
import os
import json
import tempfile
import warnings

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(pwd, 'externals'))

import numpy as np
import nibabel as nb
import pylab as pl

from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from nipy.labs.viz import plot_map
from nipy.labs.viz_tools import cm
from scipy.stats import scoreatpercentile

from externals import tempita
from externals import markdown
from viz_utils import plot_contour_atlas, plot_label_atlas

from utils import make_dir


def _check_plot_map_params(params):
    if params is None:
        params = {}
    _params = {'cmap': cm.cold_hot,
               'slicer': 'z',
               'cut_coords': 7}
    if 'threshold' not in params and 'percentile' not in params:
        _params['percentile'] = 90
    _params.update(params)
    return _params


def _check_save_params(params):
    if params is None:
        params = {}
    _params = {'dpi': 200}
    _params.update(params)
    return _params


def check_reporter(reporter):
    if isinstance(reporter, (str, unicode)):
        return Reporter(reporter)
    return reporter


class Reporter(BaseEstimator):

    def __init__(self, report_dir=None,
                 plot_map_params=None, save_params=None, safe_dir=True):
        self.report_dir = report_dir or tempfile.mkdtemp(prefix='report_')

        make_dir(self.report_dir, safe=safe_dir, strict=False)
        self.plot_map_params = _check_plot_map_params(plot_map_params)
        self.save_params = _check_save_params(save_params)

    def plot_map(self, niimg, title):
        data = niimg.get_data().squeeze()
        params = self.plot_map_params.copy()
        if 'percentile' in self.plot_map_params:
            threshold = scoreatpercentile(
                data.ravel(), self.plot_map_params['percentile'])
            params.pop('percentile')
            params['threshold'] = threshold
        # vmax = np.abs(data).max()
        vmax = np.percentile(np.abs(data), 99)
        plot_map(data,
                 affine=niimg.get_affine(),
                 vmin=-vmax,
                 vmax=vmax,
                 title=title,
                 **params)
        fname = title.replace(' ', '_').replace('/', '_')
        pl.savefig(os.path.join(
            self.report_dir, '%s.png' % fname), **self.save_params)
        path = os.path.join(self.report_dir, '%s.nii.gz' % fname)
        nb.save(niimg, path)
        pl.close('all')
        return path

    def plot_contours(self, niimgs, labels):
        plot_contour_atlas(niimgs, labels)
        pl.savefig(os.path.join(
            self.report_dir, 'contour_atlas.png'), **self.save_params)
        pl.close('all')

    def plot_labels(self, niimgs, labels):
        plot_label_atlas(niimgs, labels)
        pl.savefig(os.path.join(
            self.report_dir, 'label_atlas.png'), **self.save_params)
        pl.close('all')

    def eval_classif(self, y_true, y_pred, labels):
        with open(os.path.join(self.report_dir,
                               'classification_report.txt'), 'wb') as f:
            f.write(classification_report(y_true, y_pred, target_names=labels))
