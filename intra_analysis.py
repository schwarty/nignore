import os
import tempfile

import numpy as np
import nibabel as nb

from nipy.modalities.fmri.glm import GeneralLinearModel
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import resample_img
from joblib import Memory, Parallel, delayed


target_affine = nb.load('mask_3mm.nii.gz').get_affine()
target_shape = nb.load('mask_3mm.nii.gz').shape


class IntraLinearModel(object):

    def __init__(self, masker=MultiNiftiMasker(),
                 output_dir=tempfile.gettempdir(),
                 glm_model='ols', contrast_type='t', output_z=True,
                 output_stat=False, output_effects=False,
                 output_variance=False, memory=Memory(cachedir=None),
                 target_affine=None, target_shape=None,
                 model_tol=1e10,
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
        self.model_tol = model_tol
        self.memory = memory
        self.n_jobs = n_jobs

    def fit(self, niimgs, design_matrices):
        data = self.masker.fit_transform(niimgs)
        self.glm_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self.memory.cache(_fit_glm))(
                design_matrix, session_data, self.glm_model)
            for design_matrix, session_data in zip(design_matrices, data)
            if not session_data is None and not design_matrix is None)
        for i, design_matrix in enumerate(design_matrices):
            sv = np.linalg.svd(design_matrix)[1]
            if sv[0] / sv[-1] > self.model_tol:
                self.glm_[i] = None
        return self

    def _contrast(self, contrast_id, contrast_values):
        contrast = None

        n_regressors = [glm.X.shape[1] if glm is not None else 0
                        for glm in self.glm_]
        contrast_values = check_contrast(contrast_values, n_regressors)

        for i, (glm, con_val) in enumerate(zip(self.glm_, contrast_values)):
            if con_val is None or np.all(con_val == 0) or glm is None:
                # contrast null for session, or design_matrix ill conditioned
                pass
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
                output_dir = os.path.join(
                    self.output_dir, '%s_maps' % estimate.rsplit('_')[0])
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                map_path = os.path.join(output_dir, '%s.nii.gz' % contrast_id)
                niimg.to_filename(map_path)
                outputs.append(map_path)

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


def do_intra_analysis(masker, output_dir, niimgs, design_matrices, contrasts):
    modeler = IntraLinearModel(
        masker,
        glm_model='ar1',
        output_dir=output_dir,
        target_affine=target_affine,
        target_shape=target_shape,
        output_effects=True,
        output_variance=True,
        n_jobs=1)
    modeler.fit(niimgs, design_matrices)
    modeler.contrast(contrasts)


if __name__ == '__main__':
    from nignore.openfmri import Loader, glob_subjects_dirs
    from nignore.spm import IntraEncoder
    from nignore.utils import globing

    n_jobs = 48

    root_dir = '/storage/workspace/yschwart/new_brainpedia/preproc'
    result_dir = '/storage/workspace/yschwart/new_brainpedia/intra_stats_3mm'

    loader = Loader(model_id='model001')
    encoder = IntraEncoder()

    masker = MultiNiftiMasker(mask='mask_3mm.nii.gz', standardize=True,
                              smoothing_fwhm=6, n_jobs=1)

    def one_map_per_run(contrasts):
        angry_contrasts = {}
        for contrast_id in contrasts:
            n_sessions = len(contrasts[contrast_id])
            run_id = 1
            task_id, con_name = contrast_id.split('_', 1)
            for i, session_con in enumerate(contrasts[contrast_id]):
                is_null = np.all(np.array(session_con == 0))
                if session_con is not None and not is_null:
                    new_con_id = '%s_run%03i_%s' % (task_id, run_id, con_name)
                    angry_contrasts[new_con_id] = [None] * n_sessions
                    angry_contrasts[new_con_id][i] = session_con
                    run_id += 1
        return angry_contrasts

    def sanitize_contrast(contrasts, insert_derivative=True, one_per_run=True):
        angry_contrasts = {}
        for contrast_id in contrasts:
            contrast = []
            for session_con in contrasts[contrast_id]:
                if session_con is not None:
                    session_con = np.array(session_con)
                    session_con = np.insert(
                        session_con,
                        np.arange(session_con.size) + 1, 0).tolist()
                contrast.append(session_con)
            angry_contrasts[contrast_id] = contrast
        if one_per_run:
            return one_map_per_run(angry_contrasts)
        return angry_contrasts

    for study_dir in globing(root_dir, '*'):
        study_id = os.path.split(study_dir)[1]
    # for study_id in ['pinel2009twins']:
        print study_id

        infos = glob_subjects_dirs('%s/%s/sub???' % (root_dir, study_id))
        docs = loader.fit_transform(infos['subjects_dirs'], infos['subjects'])
        subjects_niimgs = encoder.fit_transform(docs, infos['subjects'])

        # for i, subject_id in enumerate(infos['subjects']):
        #     print subject_id
        #     output_dir = '%s/%s/%s/%s/%s' % (result_dir, study_id, subject_id,
        #                                      'model', 'model002')
        #     niimgs = subjects_niimgs[i]
        #     design_matrices = encoder.design_matrices_[i]
        #     contrasts = docs[i]['contrasts']

        #     # insert zeros in con_val because there is a derivative
        #     angry_contrasts = sanitize_contrast(contrasts)
        #     modeler = IntraLinearModel(
        #         masker,
        #         glm_model='ar1',
        #         output_dir=output_dir,
        #         target_affine=target_affine,
        #         target_shape=target_shape,
        #         output_effects=True,
        #         output_variance=True,
        #         n_jobs=n_jobs)
        #     modeler.fit(niimgs, design_matrices)
        #     modeler.contrast(angry_contrasts)
        stop
        Parallel(n_jobs=n_jobs)(delayed(do_intra_analysis)(
            masker=masker,
            output_dir='%s/%s/%s/%s/%s' % (
                result_dir, study_id, subject_id,
                'model', 'model002'),
            niimgs=subjects_niimgs[i],
            design_matrices=encoder.design_matrices_[i],
            contrasts=sanitize_contrast(docs[i]['contrasts']))
            for i, subject_id in enumerate(infos['subjects'])
        )
