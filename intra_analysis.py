import os
import tempfile

import numpy as np
import nibabel as nb

from nipy.modalities.fmri.glm import GeneralLinearModel
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import resample_img
from joblib import Memory, Parallel, delayed

from _globals import target_affine, target_shape


class IntraLinearModel(object):

    def __init__(self, masker=MultiNiftiMasker(),
                 output_dir=tempfile.gettempdir(),
                 glm_model='ols', contrast_type='t', output_z=True,
                 output_stat=False, output_effects=False,
                 output_variance=False, memory=Memory(cachedir=None),
                 target_affine=None, target_shape=None,
                 n_jobs=1):
        self.masker = masker
        self.output_dir = output_dir
        self.glm_model = glm_model
        self.contrast_type = contrast_type
        self.output_z = output_z
        self.output_stat = output_stat
        self.output_effects = output_effects
        self.output_variance = output_variance
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.memory = memory
        self.n_jobs = n_jobs

    def fit(self, niimgs, design_matrices):
        data = self.masker.fit_transform(niimgs)
        self.glm_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self.memory.cache(_fit_glm))(
                design_matrix, session_data, self.glm_model)
            for design_matrix, session_data in zip(design_matrices, data)
            if not session_data is None and not design_matrix is None)
        return self

    def _contrast(self, contrast_id, contrast_values):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        contrast = None

        n_regressors = [glm.X.shape[1] for glm in self.glm_]
        contrast_values = check_contrast(contrast_values, n_regressors)

        for i, (glm, con_val) in enumerate(zip(self.glm_, contrast_values)):
            if con_val is None or np.all(con_val == 0):
                pass  # print 'Contrast for session %d is null' % i
            elif contrast is None:
                contrast = glm.contrast(
                    con_val, contrast_type=self.contrast_type)
            else:
                contrast = contrast + glm.contrast(
                    con_val, contrast_type=self.contrast_type)

        if contrast is None:
            return dict()

        mask_array = self.masker.mask_img_.get_data().astype('bool')
        affine = self.masker.mask_img_.get_affine()

        if self.output_z or self.output_stat:
            # compute the contrast and stat
            contrast.z_score()

        do_outputs = [self.output_z, self.output_stat,
                      self.output_effects, self.output_variance]
        estimates = ['z_score_', 'stat_', 'effect', 'variance']
        descrips = ['z statistic', 'Statistical value',
                    'Estimated effect', 'Estimated variance']
        outputs = []
        for (do_output, estimate, descrip) in zip(
                do_outputs, estimates, descrips):

            if do_output:
                result_map = np.zeros(mask_array.shape)
                result_map[mask_array] = getattr(contrast, estimate).squeeze()
                niimg = nb.Nifti1Image(result_map, affine=affine)
                if (self.target_affine is not None
                        or self.target_shape is not None):
                    niimg = resample_img(
                        niimg,
                        target_affine=self.target_affine,
                        target_shape=self.target_shape)

                niimg_path = os.path.join(
                    self.output_dir, '%s_map.nii.gz' % contrast_id)
                niimg.to_filename(niimg_path)
                outputs.append(niimg_path)
        return outputs

    def contrast(self, contrasts):
        outputs = {}
        for contrast_id in sorted(contrasts.keys()):
            outputs[contrast_id] = self._contrast(
                contrast_id, contrasts[contrast_id])
        return outputs


def _fit_glm(X, Y, glm_model):
    glm = GeneralLinearModel(X)
    glm.fit(Y, model=glm_model)
    return glm


def check_contrast(con_val, n_regressors):
    contrast_values = []
    if not isinstance(con_val[0], list) and con_val[0] is not None:
        con_val = [con_val]

    for i, (n_reg, con_spec) in enumerate(zip(n_regressors, con_val)):
        session_con = np.zeros(n_reg)

        if con_spec is not None:
            con_spec = np.array(con_spec)
            session_con[:con_spec.size] = con_spec[:session_con.size]

        contrast_values.append(session_con)

    return contrast_values


if __name__ == '__main__':
    from nignore.openfmri import Loader, glob_subjects_dirs
    from nignore.spm import IntraEncoder

    n_jobs = 1

    root_dir = '/media/ys218403/mobile/brainpedia/preproc'
    result_dir = '/home/ys218403/Data/intra_stats'

    loader = Loader(model_id='model001')
    encoder = IntraEncoder()
    masker = MultiNiftiMasker(mask='mask.nii.gz', standardize=True,
                              smoothing_fwhm=6, n_jobs=n_jobs)

    def sanitize_contrast(contrast, insert_derivative=True):
        angry_contrasts = {}
        for contrast_id in contrasts:
            if ('house_vs_baseline' in contrast_id
                    or 'face_vs_baseline' in contrast_id
                    or 'face_vs_house' in contrast_id):
                contrast = []
                for session_con in contrasts[contrast_id]:
                    if session_con is not None:
                        session_con = np.array(session_con)
                        session_con = np.insert(
                            session_con,
                            np.arange(session_con.size) + 1, 0).tolist()
                    contrast.append(session_con)
                angry_contrasts[contrast_id] = contrast
        return angry_contrasts

    for study_id in ['ds105']:
        print study_id

        infos = glob_subjects_dirs('%s/%s/sub???' % (root_dir, study_id))
        docs = loader.fit_transform(infos['subjects_dirs'], infos['subjects'])
        subjects_niimgs = encoder.fit_transform(docs, infos['subjects'])

        for i, subject_id in enumerate(infos['subjects']):
            print subject_id
            output_dir = '%s/%s/%s' % (result_dir, study_id, subject_id)
            niimgs = subjects_niimgs[i]
            design_matrices = encoder.design_matrices_[i]
            contrasts = docs[i]['contrasts']

            # insert zeros in con_val because there is a derivative
            angry_contrasts = sanitize_contrast(contrasts)
            modeler = IntraLinearModel(
                masker,
                glm_model='ar1',
                output_dir=output_dir,
                target_affine=target_affine,
                target_shape=target_shape,
                n_jobs=n_jobs)
            modeler.fit(niimgs, design_matrices)
            modeler.contrast(angry_contrasts)
