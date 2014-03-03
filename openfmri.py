import os
import re
import glob
import csv
import copy
import shutil
import warnings

import nibabel as nb
import numpy as np

from sklearn.pipeline import Pipeline
from nilearn.input_data import MultiNiftiMasker
from nilearn.input_data import NiftiMasker
from nilearn.image.resampling import resample_img
# from nignore.spm import SPMIntraDataLoader, SPMIntraDesignLoader, load_preproc
from nignore.spm import check_experimental_conditions
from nignore.spm import check_tasks
from nignore.spm import check_timeseries
from nignore.spm import IntraEncoder
from nignore.linear_modeling import LinearModeler
from nignore.utils import make_dir, del_dir, save_table, get_table, safe_name
from nignore.utils import globing
from nignore.parsing_utils import parse_path
from nignore.parsing_utils import strip_prefix_filename
from nignore.reporting import Reporter
from joblib import Memory, Parallel, delayed
from StringIO import StringIO


# class SPMOpenfMRI(object):

#     def __init__(self, study_id, get_subject,
#                  masker=None, encoder=None, modeler=None,
#                  memory=Memory(cachedir=None), n_jobs=1):
#         self.study_id = study_id
#         self.get_subject = get_subject
#         self.masker = masker
#         self.encoder = encoder
#         self.memory = memory
#         self.n_jobs = n_jobs

#     def fit(self, mat_files):
#         pass


class Designer(object):

    def __init__(self, task_contrasts=None, condition_key=None,
                 run_key=None, task_key=None):

        self.task_contrasts = task_contrasts
        self.condition_key = condition_key
        self.run_key = run_key
        self.task_key = task_key

    def fit(self, catalog, subjects_id):
        self.task_contrasts_ = self.task_contrasts
        if self.task_contrasts is None and 'contrasts' in catalog[0]:
            self.task_contrasts_ = catalog[0]['contrasts']
        self.condition_key_ = self.condition_key
        self.run_key_ = self.run_key
        self.task_key_ = self.task_key
        if self.task_key is None and self.run_key_ is not None:
            self.task_key_ = check_tasks(self.run_key_)
        if subjects_id is not None:
            self.subject_key_ = dict(
                zip(subjects_id, [doc['subject_id'] for doc in catalog]))
        else:
            self.subject_key_ = [doc['subject_id'] for doc in catalog]
        return self

    def transform(self, catalog, subjects_id):
        catalog_ = copy.deepcopy(catalog)
        for doc in catalog_:
            doc['contrasts'] = self.task_contrasts_
            if self.condition_key_ is not None:
                doc['conditions'] = self.condition_key_
            if self.run_key_ is not None:
                doc['runs'] = self.run_key_
            if self.task_key_ is not None:
                doc['tasks'] = self.task_key_
        return catalog_

    def fit_transform(self, catalog, subjects_id=None):
        return self.fit(catalog, subjects_id).transform(catalog, subjects_id)


class Dumper(object):

    def __init__(self, data_dir, study_id, model_id, merge_tasks=False,
                 resample=False, target_affine=None, target_shape=None,
                 memory=Memory(cachedir=None), n_jobs=1):
        self.data_dir = data_dir
        self.study_id = study_id
        self.model_id = model_id
        self.merge_tasks = merge_tasks
        self.resample = resample
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.memory = memory
        self.n_jobs = n_jobs

    def fit(self, catalog, subjects_id):
        doc = catalog[0]
        self.task_contrasts_ = doc.get('contrasts')
        self.condition_key_ = doc.get('conditions')
        self.run_key_ = doc.get('runs')
        self.task_key_ = doc.get('tasks')
        if subjects_id is not None:
            self.subject_key_ = dict(
                zip(subjects_id, [doc['subject_id'] for doc in catalog]))
        else:
            self.subject_key_ = [doc['subject_id'] for doc in catalog]
        return self

    def transform(self, catalog, subjects_id):
        catalog_ = copy.deepcopy(catalog)

        study_dir = make_dir(self.data_dir, self.study_id, strict=False)
        if isinstance(self.subject_key_, dict):
            save_table(self.subject_key_,
                       os.path.join(study_dir, 'subject_key.txt'))
        save_table(self.task_key_, os.path.join(study_dir, 'task_key.txt'),
                    merge=self.merge_tasks)
        save_table({'TR': catalog[0]['tr']},
                   os.path.join(study_dir, 'scan_key.txt'))
        model_dir = make_dir(study_dir, 'models', self.model_id, strict=False)

        save_task_contrasts(model_dir, catalog_[0], merge=self.merge_tasks)
        save_condition_key(model_dir, catalog_[0], merge=self.merge_tasks)

        if subjects_id is None:
            subjects_id = [doc['subject_id'] for doc in catalog]

        Parallel(n_jobs=self.n_jobs)(delayed(save_maps)(
            os.path.join(study_dir, subject_id, 'model', self.model_id),
            doc, self.resample, self.target_affine, self.target_shape)
            for subject_id, doc in zip(subjects_id, catalog_))

        Parallel(n_jobs=self.n_jobs)(delayed(save_preproc)(
            os.path.join(study_dir, subject_id,
                         'model', self.model_id), doc)
            for subject_id, doc in zip(subjects_id, catalog_))

        Parallel(n_jobs=self.n_jobs)(delayed(save_raw)(
            os.path.join(study_dir, subject_id), doc)
            for subject_id, doc in zip(subjects_id, catalog_))

        Parallel(n_jobs=self.n_jobs)(delayed(save_onsets)(
            os.path.join(study_dir, subject_id,
                         'model', self.model_id, 'onsets'), doc)
            for subject_id, doc in zip(subjects_id, catalog_))

        return catalog_

    def fit_transform(self, catalog, subjects_id):
        return self.fit(catalog, subjects_id).transform(catalog, subjects_id)


class Loader(object):

    def __init__(self, model_id, ignore=None, get_baseline=False):
        self.model_id = model_id
        self.ignore = ignore
        self.get_baseline = get_baseline

    def fit(self, subjects_dir, target=None):
        study_dir = os.path.split(subjects_dir[0])[0]
        self.study_id_ = os.path.split(study_dir)[1]
        self.run_key_ = check_run_key(study_dir)
        self.task_contrasts_ = check_task_contrasts(
            study_dir, self.model_id, self.run_key_)
        self.condition_key_ = check_condition_key(study_dir, self.model_id)
        self.scan_key_ = check_scan_key(study_dir)
        self.task_key_ = get_table(os.path.join(study_dir, 'task_key.txt'))
        self.model_key_ = get_table(os.path.join(study_dir, 'model_key.txt'))
        self.orthogonalize_ = check_orthogonalize(study_dir, self.run_key_)
        return self

    def transform(self, subjects_dir, target=None):
        catalog = []
        for subject_dir in subjects_dir:
            doc = {}
            doc['subject_id'] = os.path.split(subject_dir)[1]
            doc['tr'] = self.scan_key_['TR']
            doc['conditions'] = self.condition_key_
            doc['tasks'] = self.task_key_
            bold_dir = os.path.join(
                subject_dir, 'BOLD')
            doc.update(check_bold(bold_dir, self.run_key_, 'bold'))
            bold_dir = os.path.join(
                subject_dir, 'model', self.model_id, 'BOLD')
            doc.update(check_bold(bold_dir, self.run_key_, 'swabold'))
            doc['anatomy'] = check_anatomy(
                os.path.join(subject_dir, 'anatomy'))
            doc['wmanatomy'] = check_anatomy(
                os.path.join(subject_dir, 'model', self.model_id, 'anatomy'))
            unvalid_sessions = doc.get('unvalid_sessions', [])
            doc['mask'] = os.path.join(subject_dir, 'model',
                                       self.model_id, 'mask.nii.gz')
            doc['contrasts'] = check_contrasts(
                self.task_contrasts_, unvalid_sessions)
            doc['runs'] = check_runs(self.run_key_, unvalid_sessions)
            doc['onsets'] = check_onsets(subject_dir, self.model_id,
                                         self.run_key_, self.condition_key_,
                                         unvalid_sessions)
            if self.get_baseline:
                tol = self.get_baseline \
                    if isinstance(self.get_baseline, float) else 1e-2
                baselines, self.condition_key_ = get_baseline_onsets(
                    subject_dir, doc['onsets'], doc['n_scans'],
                    doc['runs'], doc['conditions'], tol=tol)
                doc['baseline_onsets'] = baselines
                doc['conditions'] = self.condition_key_
                onsets_with_baseline = []
                for onsets, baseline in zip(doc['onsets'], baselines):
                    onsets.extend(baseline)
                    onsets_with_baseline.append(order_onsets(onsets))
                doc['onsets'] = onsets_with_baseline

            doc['orthogonalize'] = self.orthogonalize_
            for dtype in ['z_maps', 'c_maps', 'effect_maps', 'var_maps']:
                map_dir = os.path.join(
                    subject_dir, 'model', self.model_id, dtype)
                if os.path.exists(map_dir):
                    paths = sorted(
                        glob.glob(os.path.join(map_dir, '*.nii.gz')))
                    labels = [
                        os.path.split(p)[1].split('.nii.gz')[0] for p in paths]
                    doc[dtype] = dict(zip(labels, paths))

            if self.ignore is not None:
                for ignore_key in self.ignore:
                    if ignore_key in doc:
                        del doc[ignore_key]

            catalog.append(doc)

        return catalog

    def fit_transform(self, subjects_dir, target=None):
        return self.fit(subjects_dir, target).transform(subjects_dir, target)


class IntraStats(object):

    def __init__(self, data_dir, study_id, model_id,
                 masker=None,
                 hrf_model='canonical with derivative',
                 drift_model='cosine', glm_model='ar1',
                 contrast_type='t', output_z=True, output_stat=False,
                 output_effects=False, output_variance=False,
                 merge_tasks=False, resample=False, target_affine=None,
                 target_shape=None, memory=Memory(cachedir=None), n_jobs=1):
        self.data_dir = data_dir
        self.study_id = study_id
        self.model_id = model_id
        if masker is None:
            self.masker = MultiNiftiMasker()
        else:
            self.masker = masker
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.glm_model = glm_model
        self.contrast_type = contrast_type
        self.output_z = output_z
        self.output_stat = output_stat
        self.output_effects = output_effects
        self.output_variance = output_variance
        self.merge_tasks = merge_tasks
        self.resample = resample
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.memory = memory
        self.n_jobs = n_jobs

    def fit(self, catalog, subjects_id):
        doc = catalog[0]
        self.task_contrasts_ = doc.get('contrasts')
        self.condition_key_ = check_experimental_conditions(catalog)
        self.run_key_ = doc.get('runs')
        self.task_key_ = doc.get('tasks')
        if subjects_id is not None:
            self.subject_key_ = dict(
                zip(subjects_id, [doc['subject_id'] for doc in catalog]))
        else:
            self.subject_key_ = [doc['subject_id'] for doc in catalog]
        return self

    def transform(self, catalog, subjects_id):
        catalog_ = copy.deepcopy(catalog)
        study_dir = make_dir(self.data_dir, self.study_id, strict=False)
        if isinstance(self.subject_key_, dict):
            save_table(self.subject_key_,
                       os.path.join(study_dir, 'subject_key.txt'))
        save_table(self.task_key_, os.path.join(study_dir, 'task_key.txt'),
                   merge=self.merge_tasks)
        save_table({'TR': catalog_[0]['tr']},
                   os.path.join(study_dir, 'scan_key.txt'))

        model_dir = make_dir(study_dir, 'models', self.model_id, strict=False)
        save_task_contrasts(model_dir, catalog_[0], merge=self.merge_tasks)
        save_condition_key(model_dir, catalog_[0], merge=self.merge_tasks)

        n_jobs = -1 if self.n_jobs != 1 else 1

        self.encoder_ = IntraEncoder(hrf_model=self.hrf_model,
                                     drift_model=self.drift_model,
                                     memory=self.memory,
                                     n_jobs=n_jobs)

        all_niimgs = self.encoder_.fit_transform(catalog_, subjects_id)

        if subjects_id is None:
            subjects_id = [doc['subject_id'] for doc in catalog]

        outputs = Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_glm)(
                LinearModeler(masker=self.masker,
                              reporter=os.path.join(
                                  study_dir, subject_id,
                                  'model', self.model_id),
                              glm_model=self.glm_model,
                              hrf_model=self.hrf_model,
                              contrast_type=self.contrast_type,
                              output_z=self.output_z,
                              output_stat=self.output_stat,
                              output_effects=self.output_effects,
                              output_variance=self.output_variance),
                niimgs=niimgs,
                design_matrices=design_matrices,
                contrasts=doc['contrasts'])
                for subject_id, doc, niimgs, design_matrices in zip(
                    subjects_id,
                    catalog_,
                    all_niimgs,
                    self.encoder_.design_matrices_))

        if self.resample:
            Parallel(n_jobs=n_jobs)(
                delayed(_resample_img)(
                    doc[dtype][cid], self.target_affine, self.target_shape, )
                for doc in outputs for dtype in doc for cid in doc[dtype])

        return outputs

    def fit_transform(self, catalog, subjects_id):
        return self.fit(catalog, subjects_id).transform(catalog, subjects_id)


def _compute_glm(modeler, niimgs, design_matrices, contrasts):
    modeler.fit(niimgs, design_matrices)
    return modeler.contrast(contrasts)


class GroupStats(object):

    def __init__(self, data_dir, study_id, model_id, masker=NiftiMasker(),
                 output_mean=True, resample=False, target_affine=None,
                 target_shape=None, memory=Memory(cachedir=None), n_jobs=1):
        self.data_dir = data_dir
        self.study_id = study_id
        self.model_id = model_id
        self.masker = masker
        self.resample = resample
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.memory = memory
        self.n_jobs = n_jobs

    def fit(self, catalog, subjects_id=None):
        self.dtypes_ = set()
        self.labels_ = set()
        for doc in catalog:
            for dtype in ['z_maps', 'c_maps', 'effect_maps', 'var_maps']:
                if dtype in doc:
                    self.dtypes_.add(dtype)
                    self.labels_.update(doc[dtype].keys())

        self.dtypes_ = sorted(self.dtypes_)
        self.labels_ = sorted(self.labels_)

        return self

    def transform(self, catalog, subjects_id=None):
        groups = {}
        for dtype in self.dtypes_:
            for label in self.labels_:
                for doc in catalog:
                    if label in doc[dtype]:
                        groups.setdefault(label, {}).setdefault(
                            dtype, []).append(doc[dtype][label])

        Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_mean)(
                self.data_dir, self.study_id, self.model_id,
                self.masker, groups[label], label)
            for label in groups)

    def fit_transform(self, catalog, subjects_id=None):
        return self.fit(catalog, subjects_id).transform(catalog, subjects_id)


def _compute_mean(data_dir, study_id, model_id, masker, group, label):
    for dtype in group.keys():
        reporter = Reporter(
            os.path.join(data_dir, study_id, 'group', model_id, dtype),
            plot_map_params={'percentile': 96})
        data = [nb.load(niimg).get_data().ravel() for niimg in group[dtype]]
        affine = nb.load(niimg).get_affine()
        shape = nb.load(niimg).shape
        niimg = nb.Nifti1Image(np.mean(data, axis=0).reshape(shape), affine)
        reporter.plot_map(niimg, label)


def _resample_img(path, target_affine=None, target_shape=None,
                  interpolation='continuous'):
    img = resample_img(path, target_affine, target_shape,
                       interpolation, copy=False)
    nb.save(img, path)


def save_condition_key(model_dir, doc, merge=False):
    file_name = os.path.join(model_dir, 'condition_key.txt')
    condition_key = doc.get('conditions')
    mode = 'wb' if not merge else 'ab'
    if condition_key is not None:
        with open(file_name, mode) as f:
            writer = csv.writer(f, delimiter=' ', quotechar='"')
            for condition in condition_key:
                writer.writerow(condition.split('_', 2))


def save_task_contrasts(model_dir, doc, merge=False):
    file_name = os.path.join(model_dir, 'task_contrasts.txt')
    contrast_key = doc.get('contrasts')
    mode = 'wb' if not merge else 'ab'
    if contrast_key is not None:
        with open(file_name, mode) as f:
            writer = csv.writer(f, delimiter=' ', quotechar='"')
            for key in sorted(contrast_key.keys()):
                val = contrast_key[key]
                row = []
                if key.startswith('task'):
                    row += key.split('_', 1)
                else:
                    task_id = get_contrast_task(val, doc['runs'])
                    row += [task_id, key]
                row += get_contrast_value(val, doc['runs'])
                writer.writerow(row)


def get_contrast_task(contrast, run_key):
    task_id = set()
    for run_id, session in zip(run_key, contrast):
        if session is not None and np.any(session) != 0:
            task_id.add(run_id.split('_')[0])
    return '_'.join(task_id)


def get_contrast_value(contrast, run_key):
    task_id = set()
    value = []
    for run_id, session in zip(run_key, contrast):
        if session is not None and np.any(session) != 0:
            if not run_id.split('_')[0] in task_id:
                value.extend(session)
            task_id.add(run_id.split('_')[0])
    return value


def save_onsets(onsets_dir, doc, merge=False):
    run_key = doc.get('runs')
    if 'onsets' in doc:
        for session_id, session in zip(run_key, doc['onsets']):
            if not merge:
                del_dir(onsets_dir, session_id)
            session_dir = make_dir(onsets_dir, session_id, strict=False)
            for onset in session:
                cond_id = onset[0]
                values = [str(v) for v in onset[1:]]
                with open(os.path.join(session_dir,
                                       '%s.txt' % cond_id), 'a') as f:
                    writer = csv.writer(f, delimiter=' ', quotechar='"')
                    writer.writerow(values)


def save_maps(model_dir, doc, resample=False,
              target_affine=None, target_shape=None):
    for dtype in ['c_maps', 't_maps']:
        if dtype in doc:
            maps_dir = make_dir(model_dir, dtype, strict=False)
            for key in doc[dtype]:
                fname = '%s.nii.gz' % safe_name(key.lower())
                img = nb.load(doc[dtype][key])
                if resample:
                    img = resample_img(img, target_affine, target_shape)
                nb.save(img, os.path.join(maps_dir, fname))
    if 'beta_maps' in doc:
        maps_dir = make_dir(model_dir, 'beta_maps')
        for path in doc['beta_maps']:
            fname = '%s.nii.gz' % safe_name(os.path.split(
                path)[1].lower().split('.')[0])
            img = nb.load(path)
            if resample:
                img = resample_img(
                    img, target_affine, target_shape, copy=False)
            nb.save(img, os.path.join(maps_dir, fname))
    if 'mask' in doc:
        img = nb.load(doc['mask'])
        if resample:
            img = resample_img(img, target_affine, target_shape,
                               interpolation='nearest', copy=False)
        nb.save(img, os.path.join(model_dir, 'mask.nii.gz'))


def save_preproc(model_dir, doc):
    if 'swabold' in doc:
        run_key = doc['runs']
        for label, session_data, motion in zip(
                run_key, doc['swabold'], doc['motion']):
            if isinstance(session_data, (list, np.ndarray)):
                img = nb.concat_images(session_data)
            else:
                img = nb.load(session_data)
            session_dir = make_dir(model_dir, 'BOLD', label)
            nb.save(img, os.path.join(session_dir, 'bold.nii.gz'))
            if isinstance(motion, (str, unicode)):
                shutil.copyfile(
                    motion, os.path.join(session_dir, 'motion.txt'))
            else:
                np.savetxt(os.path.join(session_dir, 'motion.txt'), motion)
    if 'wmanatomy' in doc:
        anat_dir = make_dir(model_dir, 'anatomy')
        img = nb.load(doc['wmanatomy'])
        nb.save(img, os.path.join(anat_dir, 'highres001_brain.nii.gz'))


def save_raw(subject_dir, doc):
    if 'bold' in doc:
        run_key = doc['runs']
        for label, session_data in zip(run_key, doc['bold']):
            if isinstance(session_data, (list, np.ndarray)):
                img = nb.concat_images(session_data, check_affines=False)
            else:
                img = nb.load(session_data)
            session_dir = make_dir(subject_dir, 'BOLD', label, strict=False)
            nb.save(img, os.path.join(session_dir, 'bold.nii.gz'))
    if 'anatomy' in doc:
        anat_dir = make_dir(subject_dir, 'anatomy', strict=False)
        img = nb.load(doc['anatomy'])
        nb.save(img, os.path.join(anat_dir, 'highres001.nii.gz'))


def check_run_key(study_dir):
    run_key = set()
    for subject_dir in glob.glob(os.path.join(study_dir, 'sub???')):
        runs = [
            os.path.split(x)[1]
            for x in glob.glob(os.path.join(subject_dir, 'BOLD', '*'))]
        run_key.update(runs)
        runs = [
            os.path.split(x)[1]
            for x in glob.glob(os.path.join(subject_dir, 'model',
                                            'model001', 'BOLD', '*'))]
        run_key.update(runs)
        runs = [
            os.path.split(x)[1]
            for x in glob.glob(os.path.join(subject_dir, 'model',
                                            'model001', 'onsets', '*'))]
        run_key.update(runs)

    return sorted(run_key)


def check_task_contrasts(study_dir, model_id, run_key):
    model_dir = os.path.join(study_dir, 'models', model_id)
    task_contrasts = {}

    if not os.path.exists(os.path.join(model_dir, 'task_contrasts.txt')):
        return dict()

    with open(os.path.join(model_dir, 'task_contrasts.txt')) as f:
        f = StringIO(f.read().replace('\t', ' '))

    reader = csv.reader(f, delimiter=' ', quotechar='"')
    for row in reader:
        if not row[0].startswith('task'):
            row = row.insert(0, 'task001')
        task_id = row[0]
        contrast_id = row[1]
        if not contrast_id.startswith('task'):
            contrast_id = '%s_%s' % (task_id, contrast_id)
        try:
            con_val = np.array(row[2:], dtype='float').tolist()
        except:
            continue
        for run_id in run_key:
            if run_id.startswith(task_id):
                task_contrasts.setdefault(contrast_id, []).append(con_val)
            else:
                task_contrasts.setdefault(contrast_id, []).append(None)
    return task_contrasts


def check_condition_key(study_dir, model_id):
    model_dir = os.path.join(study_dir, 'models', model_id)
    condition_key = []

    with open(os.path.join(model_dir, 'condition_key.txt')) as f:
        f = StringIO(f.read().replace('\t', ' '))

    reader = csv.reader(f, delimiter=' ', quotechar='"')

    for row in reader:
        if not row[0].startswith('task'):
            row = row.insert(0, 'task001')
        if row[-1] == '':
            row = row[:-1]
        condition_key.append('_'.join(row))
    return condition_key


def check_scan_key(study_dir):
    """Parse scan_key file to get scanning information (currently only TR).
    """
    scan_key = get_table(os.path.join(study_dir, 'scan_key.txt'))
    scan_key['TR'] = float(scan_key['TR'])
    return scan_key


def check_onsets(subject_dir, model_id, run_key, condition_key,
                 unvalid_sessions=None):
    unvalid_sessions = [] if unvalid_sessions is None else unvalid_sessions
    onsets_dir = os.path.join(subject_dir, 'model', model_id, 'onsets')
    onsets = []
    for i, run_id in enumerate(run_key):
        if i in unvalid_sessions:
            continue
        run_dir = os.path.join(onsets_dir, run_id)
        if not os.path.exists(run_dir):
            onsets.append([])
        else:
            events = []
            labels = []
            for condition_id in condition_key:
                task_id, cond_id = condition_id.split('_')[:2]
                if run_id.startswith(task_id):
                    cond_file = os.path.join(run_dir, '%s.txt' % cond_id)
                    if os.path.exists(cond_file):
                        with open(cond_file) as f:
                            cond = f.read()
                            cond.replace('\t', ' ').replace('\r', ' ')
                            cond = re.sub('\s+', ' ', cond)
                            cond = np.fromstring(cond, dtype='float', sep=' ')
                            cond = cond.reshape(cond.shape[0] / 3, 3)
                            events.append(cond)
                            labels.extend([cond_id] * cond.shape[0])
                    else:
                        events.append([0., 0., 0.])
                        labels.append(cond_id)

            events = np.vstack(events)
            labels = np.array(labels)
            order = np.argsort(events[:, 0])
            onsets.append(zip(labels[order], *events[order].T))

    return onsets


def check_orthogonalize(study_dir, run_key):
    path = os.path.join(study_dir, 'models', 'model001', 'orthogonalize.txt')
    if not os.path.exists(path):
        return

    orthogonalize = []
    with open(path, 'rb') as f:
        reader = csv.reader(f, delimiter=' ')
        for run_id in run_key:
            session_orth = []
            for row in reader:
                task_id, x, y = row
                if run_id.startswith(task_id):
                    session_orth.append((x, y))
            orthogonalize.append(session_orth)
    return orthogonalize


def check_bold(bold_dir, run_key, bold_key):
    sessions = []
    doc = {}
    doc[bold_key] = []
    doc['motion'] = []
    n_scans = []
    unvalid_sessions = []

    for i, session_id in enumerate(run_key):
        session_dir = os.path.join(bold_dir, session_id)
        if os.path.exists(session_dir):
            sessions.append(os.path.split(session_dir)[1])
            bold = os.path.join(session_dir, 'bold.nii.gz')
            n_scan = nb.load(bold).shape[-1]
            doc[bold_key].append(bold)
            n_scans.append(n_scan)

            if os.path.exists(os.path.join(session_dir, 'motion.txt')):
                with open(os.path.join(session_dir, 'motion.txt')) as f:
                    regs = np.fromstring(f.read().replace('\t', ' '), sep=' ')
                    regs = regs.reshape(regs.shape[0] / 6, 6)
                    if regs.shape[0] != n_scan:
                        raise Exception('n_scans=%s differs from '
                                        'n_timepoints=%s in motion file' % (
                                            n_scan, regs.shape[0]))
                    doc['motion'].append(regs)
        else:
            unvalid_sessions.append(i)

    subject_id = bold_dir.split(os.path.sep)[-4]
    if n_scans != []:
        doc['n_scans'] = n_scans
    if doc[bold_key] == []:
        warnings.warn('Subject %s does not have %s bold.' % (
            subject_id, bold_key))
    else:
        doc['unvalid_sessions'] = unvalid_sessions
    if sessions != run_key:
        warnings.warn('Subject %s sessions -- %s -- differ '
                      'from specification -- %s --' % (
                          subject_id, sessions, run_key))
    return doc


def check_anatomy(anatomy_dir):
    if os.path.exists(os.path.join(anatomy_dir, 'highres001_brain.nii.gz')):
        return os.path.join(anatomy_dir, 'highres001_brain.nii.gz')
    elif os.path.exists(os.path.join(anatomy_dir, 'highres001.nii.gz')):
        return os.path.join(anatomy_dir, 'highres001.nii.gz')


def check_contrasts(contrasts, unvalid_sessions):
    curated_contrasts = {}
    for k in contrasts:
        for i, session in enumerate(contrasts[k]):
            if i not in unvalid_sessions:
                curated_contrasts.setdefault(k, []).append(session)
    contrasts = {}
    for k in curated_contrasts:
        if _is_valid_contrast(curated_contrasts[k]):
            contrasts[k] = curated_contrasts[k]
    return contrasts


def _is_valid_contrast(con_val):
    is_valid = False
    for session in con_val:
        if session is not None:
            if np.any(np.array(session) != 0):
                is_valid = True
    return is_valid


def check_runs(run_key, unvalid_sessions):
    curated_run_key = []
    for i, session in enumerate(run_key):
        if i not in unvalid_sessions:
            curated_run_key.append(session)
    return curated_run_key


def get_baseline_onsets(subject_dir, onsets, n_scans, run_key,
                        condition_key, tol=1e-2):
    study_dir = os.path.split(subject_dir)[0]
    tr = check_scan_key(study_dir)['TR']
    baseline_onsets = []
    condition_key = condition_key[:]
    labels = set()
    for i, session_onsets in enumerate(onsets):
        task_id, run_id = run_key[i].split('_')
        frametimes = np.linspace(0, (n_scans[i] - 1) * tr, n_scans[i])
        timing = [frametimes[0]]
        names = set()
        for o in session_onsets:
            timing.append(o[1])
            timing.append(o[1] + o[2])
            names.add(o[0])
        timing.append(frametimes[-1])
        baseline = []
        names = sorted(names)
        label = 'cond%03i' % (int(names[-1].split('cond')[1]) + 1)
        for t0, t1 in zip(timing[::2], timing[1::2]):
            if t1 - t0 > tol:
                baseline.append((label, t0, t1 - t0, 1.))
        baseline_onsets.append(baseline)
        labels.add('%s_%s_baseline' % (task_id, label))
    condition_key.extend(labels)
    return baseline_onsets, sorted(condition_key)


def glob_subjects_dirs(pattern, ignore=None, restrict=None):
    if ignore is None:
        ignore = []

    doc = {}
    for subject_dir in sorted(glob.glob(pattern)):
        sid = os.path.split(subject_dir)[1]
        if (restrict is None and sid not in ignore) or (
                restrict is not None and sid in restrict):
            doc.setdefault('subjects_dirs', []).append(subject_dir)
            doc.setdefault('subjects', []).append(sid)
        else:
            doc.setdefault('ignored_subjects', []).append(sid)
    return doc


def order_onsets(onsets):
    conditions_onsets = np.array([onset[1:] for onset in onsets])
    conditions_labels = np.array([onset[0] for onset in onsets])
    order = np.argsort(conditions_onsets[:, 0])
    onsets = []
    for e in np.hstack([conditions_labels[:, None][order],
                        conditions_onsets[order]]):
        onsets.append((e[0], float(e[1]), float(e[2]), float(e[3])))
    return onsets
