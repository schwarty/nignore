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
from nipy.labs.viz import plot_map
from nipy.labs.viz_tools import cm

from externals import tempita
from externals import markdown
from viz_utils import plot_niimg


def _check_boundary_params(params):
    if params is None:
        params = {}
    _params = {'cmap': cm.cold_hot,
               'slicer': 'z',
               'cut_coords': 7}
    _params.update(params)
    return _params


def _check_save_params(params):
    if params is None:
        params = {}
    _params = {'dpi': 200}
    _params.update(params)
    return _params


class Reporter(BaseEstimator):

    def __init__(self, report_dir=None,
                 boundary_params=None, save_params=None):
        self.report_dir = report_dir or tempfile.mkdtemp(suffix='report_')
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
        self.boundary_params = _check_boundary_params(boundary_params)
        self.save_params = _check_save_params(save_params)

    def boundary(self, niimg, title):
        vmax = np.abs(niimg.get_data()).max()
        plot_map(niimg.get_data(),
                 affine=niimg.get_affine(),
                 vmin=-vmax,
                 vmax=vmax,
                 title=title,
                 **self.boundary_params)
        fname = title.replace(' ', '_').replace('/', '_')
        pl.savefig(os.path.join(
            self.report_dir, '%s.png' % fname), **self.save_params)
        nb.save(niimg, os.path.join(self.report_dir, '%s.nii.gz' % fname))

    # def evaluation(self, y_true, y_pred, title):
    #     fname = title.replace(' ', '_').replace('/', '_')
    #     with open(os.path.join(self.report_dir,
    #                            'evaluation_%s.json' % fname), 'wb') as f:
    #         json.dump(data, f)
