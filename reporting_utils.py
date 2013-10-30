import sys
import os
import glob
import tempfile
import warnings

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(pwd, 'externals'))

import pylab as pl
import nibabel as nb

from sklearn.metrics import precision_recall_fscore_support

from externals import tempita
from externals import markdown
from viz_utils import plot_estimator


class ReporterMixin(object):

    def _check_report_params(self):
        if not hasattr(self, 'report_dir'):
            self.report_dir = tempfile.gettempdir()
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
        if not hasattr(self, 'labels'):
            self.labels = None

    def _finalize_report(self):
        pass

    def _niimg_report(self):
        self._check_report_params()

        images = []

        if hasattr(self, 'niimgs_'):
            if self.labels is None:
                self.labels = [''] * len(self.niimgs_)
            for i, (label, niimg) in enumerate(zip(self.labels,
                                                   self.niimgs_)):
                label = i if label == '' else label
                # png image
                plot_estimator(niimg, label, self.report_params)
                fname = '%s.png' % label
                pl.savefig(os.path.join(self.report_dir, fname), dpi=200)
                images.append(fname)
                # nifti image
                fname = '%s.nii.gz' % label
                nb.save(niimg, os.path.join(self.report_dir, fname))

        elif hasattr(self, 'niimg_'):
            if self.labels is None:
                self.labels = 'estimator'
            # png image
            plot_estimator(self.niimg_, self.labels, self.report_params)
            fname = '%s.png' % label
            pl.savefig(os.path.join(self.report_dir, fname), dpi=200)
            images.append(fname)
            # nifti image
            fname = '%s.nii.gz' % label
            nb.save(niimg, os.path.join(self.report_dir, fname))
        else:
            warnings.warn('Object has not niimgs, could '
                          'not report generate report.')

    def _classification_report(self):
        self._check_report_params()

        scores = []
        if len(self.y_true_.shape) == 2:
            Y_true = self.y_true_
            Y_pred = self.y_pred_

            if self.labels is None:
                self.labels = [''] * len(self.niimgs_)

            for label, y_true, y_pred in zip(self.labels, Y_true.T, Y_pred.T):
                scores.append(
                    (label, precision_recall_fscore_support(y_true, y_pred)))

        return scores

    def configure(self, labels=None, report_dir=tempfile.gettempdir()):
        self.labels = labels
        self.report_dir = report_dir
        return self
