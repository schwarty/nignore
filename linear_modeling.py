import os
import inspect

import nibabel as nb
import numpy as np
import pylab as pl

from nipy.modalities.fmri.glm import GeneralLinearModel
from nipy.labs.viz import plot_map
from nipy.labs.viz_tools import cm
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import LeaveOneLabelOut
from joblib import Memory, Parallel, delayed

from reporting import Reporter, check_reporter

# def _fit_cv(cv, y):
#     y = np.array(y)

#     argspec = inspect.getargspec(cv.__init__)

#     kwds = {}
#     for k in cv.__dict__:
#         if k in argspec.args:
#             kwds[k] = cv.__dict__[k]
#     if 'n' in kwds:
#         kwds['n'] = y.size
#     elif 'y' in kwds:
#         kwds['y'] = y
#     elif 'labels' in kwds:
#         kwds['labels'] = y

#     return cv.__class__(**kwds)


# class SessionEncoder(object):

#     def __init__(self, cv=LeaveOneLabelOut([0])):
#         self.cv = cv

#     def fit(self, y):
#         self.cv = _fit_cv(self.cv, y)
#         return self

#     def __iter__(self):
#         for _, session in self.cv:
#             yield session

def get_loader(loader):
    if hasattr(loader, 'steps'):
        return _get_pipeline(loader)
    return loader


def _get_pipeline(pipeline):
    return get_loader(pipeline.steps[-1][1])


class LinearModeler(object):

    def __init__(self, masker=NiftiMasker(),
                 reporter=Reporter(),
                 # encoder=LabelBinarizer(),
                 glm_model='ols', hrf_model='canonical with derivative',
                 contrast_type='t', output_z=True, output_stat=False,
                 output_effects=False, output_variance=False,
                 memory=Memory(cachedir=None)):
        self.masker = masker
        self.reporter = check_reporter(reporter)
        # self.encoder = encoder
        self.glm_model = glm_model
        self.hrf_model = hrf_model
        self.contrast_type = contrast_type
        self.output_z = output_z
        self.output_stat = output_stat
        self.output_effects = output_effects
        self.output_variance = output_variance
        self.memory = memory

    def fit(self, niimgs, design_matrices):
        data = self.masker.fit_transform(niimgs)
        self.glm_ = []

        for session_data, design_matrix in zip(data, design_matrices):
            if not session_data is None and not design_matrix is None:
                # glm = GeneralLinearModel(design_matrix)
                # glm.fit(session_data, model=self.glm_model)
                self.glm_.append(self.memory.cache(_fit_glm)(
                    design_matrix, session_data, self.glm_model))

    def _contrast(self, contrast_id, contrast):
        contrast_ = None
        if not isinstance(contrast[0], list) and contrast[0] is not None:
            contrast = [contrast]

        for i, (glm, base_con) in enumerate(zip(self.glm_, contrast)):
            con_val = np.zeros(glm.X.shape[1])

            if base_con is not None:
                base_con = np.array(base_con)
                if 'derivative' in self.hrf_model:
                    base_con = np.insert(
                        base_con, np.arange(base_con.size) + 1, 0)
                con_val[:base_con.size] = base_con[:con_val.size]

            if np.all(con_val == 0):
                pass
                # print 'Contrast for session %d is null' % i
            elif contrast_ is None:
                contrast_ = glm.contrast(
                    con_val, contrast_type=self.contrast_type)
            else:
                contrast_ = contrast_ + glm.contrast(
                    con_val, contrast_type=self.contrast_type)

        if contrast_ is None:
            return dict()

        loader = get_loader(self.masker)
        mask_array = loader.mask_img_.get_data().astype('bool')
        affine = loader.mask_img_.get_affine()

        if self.output_z or self.output_stat:
            # compute the contrast and stat
            contrast_.z_score()

        do_outputs = [self.output_z, self.output_stat,
                      self.output_effects, self.output_variance]
        estimates = ['z_score_', 'stat_', 'effect', 'variance']
        descrips = ['z statistic', 'Statistical value',
                    'Estimated effect', 'Estimated variance']
        folders = ['z_maps', '%s_maps' % self.contrast_type,
                   'effect_maps', 'var_maps']
        outputs = {}
        for (do_output, estimate, descrip, folder) in zip(
            do_outputs, estimates, descrips, folders):
            if do_output:
                result_map = np.zeros(mask_array.shape)
                result_map[mask_array] = getattr(contrast_, estimate).squeeze()
                niimg = nb.Nifti1Image(result_map, affine=affine)
                reporter = Reporter(
                    report_dir=os.path.join(self.reporter.report_dir, folder),
                    plot_map_params=self.reporter.plot_map_params,
                    save_params=self.reporter.save_params)
                path = reporter.plot_map(niimg, contrast_id)
                outputs[folder] = path
        return outputs

    def contrast(self, contrasts):
        # niimgs = []
        outputs = {}
        for contrast_id in sorted(contrasts.keys()):
            out = self._contrast(contrast_id, contrasts[contrast_id])
            # if 'z_maps' in out:
            #     niimgs.append(nb.load(out['z_maps']))
            # else:
            #     key = sorted(out.keys())[0]
            #     niimgs.append(nb.load(out[key]))

            for key in out:
                outputs.setdefault(key, {}).setdefault(contrast_id, out[key])
        # labels = sorted(contrasts.keys())
        # self.reporter.plot_contours(niimgs, labels)
        # self.reporter.plot_labels(niimgs, labels)
        self.reporter.plot_map(get_loader(self.masker).mask_img_, 'mask')
        return outputs


def _fit_glm(X, Y, glm_model):
    glm = GeneralLinearModel(X)
    glm.fit(Y, model=glm_model)
    return glm
