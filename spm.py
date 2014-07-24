import os
import gzip
import glob
import hashlib

import numpy as np
import nibabel as nb
import scipy.io as sio

from joblib import Memory, Parallel, delayed
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm

from parsing_utils import find_data_dir, makeup_path
from parsing_utils import strip_prefix_filename, prefix_filename, parse_path
from utils import safe_name, check_paths, check_path


def load_matfile(mat_file):
    if isinstance(mat_file, (str, unicode)):
        if mat_file.endswith('.gz'):
            return sio.loadmat(
                gzip.open(mat_file, 'rb'),
                squeeze_me=True,
                struct_as_record=False
                )
        return sio.loadmat(
            mat_file, squeeze_me=True, struct_as_record=False)
    else:
        return mat_file


def get_intra_infos(mat_file, memory=Memory(None)):
    mat = memory.cache(load_matfile)(mat_file)['SPM']
    infos = {}
    if hasattr(mat.nscan, '__iter__'):
        infos['n_scans'] = mat.nscan.tolist() \
            if isinstance(mat.nscan.tolist(), list) else [mat.nscan.tolist()]
        infos['n_sessions'] = mat.nscan.size
    else:
        infos['n_scans'] = mat.nscan
        infos['n_sessions'] = 1

    infos['tr'] = float(mat.xY.RT)    # xY: data
    return infos


def get_intra_preproc(mat_file, work_dir, n_scans, memory=Memory(None)):
    mat = memory.cache(load_matfile)(mat_file)['SPM']
    preproc = {}

    get_motion_file = False
    if len(n_scans) > 1:
        preproc['motion'] = []
        for session in mat.Sess:
            preproc['motion'].append(session.C.C.tolist())
            if session.C.C.size == 0:
                get_motion_file = True
    else:
        preproc['motion'] = [mat.Sess.C.C.tolist()]
        if mat.Sess.C.C.size == 0:
            get_motion_file = True

    swabold = check_paths(mat.xY.P)
    if len(nb.load(makeup_path(work_dir, swabold[0])).shape) == 4:
        swabold = np.unique(swabold)
    else:
        swabold = np.split(swabold, np.cumsum(n_scans)[:-1])

    if get_motion_file:
        preproc['motion'] = []

    for session in swabold:
        session_dir = find_data_dir(work_dir, check_path(session[0]))
        if get_motion_file:
            motion_file = glob.glob(os.path.join(session_dir, 'rp_*.txt'))[0]
            motion = np.fromfile(motion_file, sep=' ')
            motion = motion.reshape(motion.shape[0] / 6, 6)
            preproc['motion'].append(motion)

        if isinstance(session, (list, np.ndarray)):
            scans = [os.path.join(session_dir, os.path.split(scan)[1].strip())
                     for scan in session]
            preproc.setdefault('swabold', []).append(scans)
            preproc.setdefault('abold', []).append(
                [strip_prefix_filename(scan, 2) for scan in scans])
            preproc.setdefault('bold', []).append(
                [strip_prefix_filename(scan, 3) for scan in scans])
        else:
            preproc.setdefault('swabold', []).append(session)
            preproc.setdefault('abold', []).append(
                strip_prefix_filename(session, 2))
            preproc.setdefault('bold', []).append(
                strip_prefix_filename(session, 3))

    return preproc


def get_intra_onsets(mat_file, memory=Memory(None)):
    mat = memory.cache(load_matfile)(mat_file)['SPM']
    onsets = []
    conditions = []
    if hasattr(mat.Sess, '__iter__'):
        for session in mat.Sess:
            names = []
            events = []
            labels = []
            for i, condition in enumerate(session.U):
                condition_id = 'cond%03i' % (i + 1)
                condition_name = str(condition.name)
                time = condition.ons.tolist()
                duration = condition.dur.tolist()
                if not isinstance(time, list):
                    time = [time]
                    duration = [duration]
                n_events = len(time)
                amplitude = [1] * n_events
                events.append(zip(time, duration, amplitude))
                labels += [condition_id] * n_events
                names.append(condition_name)
            conditions.append(names)
            events = np.vstack(events)
            labels = np.array(labels)
            order = np.argsort(events[:, 0])
            onsets.append(zip(labels[order], *events[order].T))
            # onsets.append(zip(labels, *events.T))
    else:
        events = []
        labels = []
        for i, condition in enumerate(mat.Sess.U):
            condition_id = 'cond%03i' % (i + 1)
            condition_name = str(condition.name)
            time = condition.ons.tolist()
            duration = condition.dur.tolist()
            if not isinstance(time, list):
                time = [time]
                duration = [duration]
            n_events = len(time)
            amplitude = [1] * n_events
            events.append(zip(time, duration, amplitude))
            labels += [condition_id] * n_events
            conditions.append(condition_name)
        conditions = [conditions]
        events = np.vstack(events)
        labels = np.array(labels)
        order = np.argsort(events[:, 0])
        onsets.append(zip(labels[order], *events[order].T))
        # onsets.append(zip(labels, *events.T))
    return onsets, conditions


def get_intra_images(mat_file, work_dir, memory=Memory(None)):
    mat = memory.cache(load_matfile)(mat_file)['SPM']
    images = {}
    images['beta_maps'] = []
    images['c_maps'] = {}
    images['t_maps'] = {}
    images['contrasts'] = {}
    for c in mat.xCon:
        name = safe_name(str(c.name))
        try:
            images['c_maps'][name] = check_path(
                os.path.join(work_dir, str(c.Vcon.fname)))
            images['t_maps'][name] = check_path(
                os.path.join(work_dir, str(c.Vspm.fname)))
            images['contrasts'][name] = c.c.tolist()
        except:
            pass  # sometimes c.Vcon is an empty array
    for i, b in enumerate(mat.Vbeta):
        images['beta_maps'].append(
            check_path(os.path.join(work_dir, str(b.fname))))
    return images


def get_intra_design(mat_file, n_scans, contrasts, memory=Memory(None)):
    mat = memory.cache(load_matfile)(mat_file)['SPM']
    doc = {}

    design_matrix = mat.xX.X.tolist()           # xX: model
    conditions = [str(i) for i in mat.xX.name]

    n_sessions = len(n_scans)
    design_matrices = np.vsplit(design_matrix, np.cumsum(n_scans[:-1]))
    conditions = np.array(conditions)

    sessions_dm = []
    sessions_contrast = {}
    for i, dm in zip(range(n_sessions), design_matrices):
        mask = np.array(
            [cond.startswith('Sn(%s)' % (i + 1)) for cond in conditions])
        sessions_dm.append(dm[:, mask][:, :-1].tolist())

        for contrast_id in contrasts:
            sessions_contrast.setdefault(contrast_id, []).append(
                np.array(contrasts[contrast_id])[mask][:-1].tolist())

    doc['design_matrices'] = sessions_dm
    doc['contrasts'] = sessions_contrast
    return doc


def load_intra(mat_file, memory=Memory(None), **kwargs):
    doc = {}
    mat_file = os.path.realpath(mat_file)
    doc.update(parse_path(mat_file, **kwargs))

    work_dir = os.path.split(mat_file)[0]
    mat_file = memory.cache(load_matfile)(mat_file)
    mat = mat_file['SPM']

    doc.update(get_intra_infos(mat_file, memory))

    doc['mask'] = check_path(os.path.join(work_dir, str(mat.VM.fname)))
    doc['onsets'], doc['conditions'] = get_intra_onsets(mat_file, memory)
    doc.update(get_intra_preproc(mat_file, work_dir, doc['n_scans'], memory))
    doc.update(get_intra_images(mat_file, work_dir, memory))
    doc.update(get_intra_design(
        mat_file, doc['n_scans'], doc['contrasts'], memory))

    return doc


def load_preproc(mat_file, memory=Memory(None), **kwargs):
    doc = {}
    mat_file = os.path.realpath(mat_file)
    doc.update(parse_path(mat_file, **kwargs))

    work_dir = os.path.split(mat_file)[0]
    mat_file = memory.cache(load_matfile)(mat_file)
    if 'jobs' in mat_file:
        mat = mat_file['jobs']
    elif 'matlabbatch' in mat_file:
        mat = mat_file['matlabbatch']
    else:
        raise Exception("mat_file type not known.")

    if not hasattr(mat, '__iter__'):
        return doc

    for step in mat:
        if hasattr(step, 'spm'):
            step = step.spm
            doc.update(parse_spm8_preproc(work_dir, step))
        else:
            doc.update(parse_spm5_preproc(work_dir, step))
    return doc


def parse_spm8_preproc(work_dir, step):
    doc = {}

    if hasattr(step, 'spatial') and hasattr(step.spatial, 'preproc'):
        doc['anatomy'] = makeup_path(
            work_dir, check_path(step.spatial.preproc.data))
        doc['wmanatomy'] = prefix_filename(doc['anatomy'], 'wm')

    if hasattr(step, 'temporal'):
        doc['n_slices'] = int(step.temporal.st.nslices)
        doc['ref_slice'] = int(step.temporal.st.refslice)
        doc['slice_order'] = step.temporal.st.so.tolist()
        doc['ta'] = float(step.temporal.st.ta)
        doc['tr'] = float(step.temporal.st.tr)
        doc['bold'] = []
        doc['swabold'] = []
        if len(step.temporal.st.scans[0].shape) == 0:
            bold = [step.temporal.st.scans]
        else:
            bold = step.temporal.st.scans
        for session in bold:
            data_dir = find_data_dir(work_dir, str(session[0]))
            doc['bold'].append(check_paths(
                [os.path.join(data_dir, os.path.split(str(x))[1])
                 for x in session]))
            doc['swabold'].append(check_paths(
                [prefix_filename(os.path.join(
                    data_dir, os.path.split(str(x))[1]), 'swa')
                for x in session]))
        doc['n_scans'] = [len(s) for s in doc['bold']]
    return doc


def parse_spm5_preproc(work_dir, step):
    doc = {}
    if hasattr(step, 'spatial') and hasattr(step.spatial, 'realign'):
        realign = step.spatial.realign.estwrite
        motion = []
        if len(realign.data[0].shape) == 0:
            realign = [realign]
        else:
            realign = realign.data
            for session in realign:
                data_dir = find_data_dir(work_dir, check_path(session[0]))
                motion.append(glob.glob(os.path.join(data_dir, 'rp_*.txt'))[0])
            doc['motion'] = motion
    if hasattr(step, 'spatial') and isinstance(step.spatial, np.ndarray):
        doc['anatomy'] = makeup_path(
            work_dir, check_path(step.spatial[0].preproc.data))
        doc['wmanatomy'] = prefix_filename(makeup_path(
            work_dir,
            check_path(step.spatial[1].normalise.write.subj.resample)),
            'w')
    if hasattr(step, 'temporal'):
        doc['n_slices'] = int(step.temporal.st.nslices)
        doc['ref_slice'] = int(step.temporal.st.refslice)
        doc['slice_order'] = step.temporal.st.so.tolist()
        doc['ta'] = float(step.temporal.st.ta)
        doc['tr'] = float(step.temporal.st.tr)
        doc['bold'] = []
        doc['swabold'] = []
        if len(step.temporal.st.scans[0].shape) == 0:
            bold = [step.temporal.st.scans]
        else:
            bold = step.temporal.st.scans
        for session in bold:
            data_dir = find_data_dir(work_dir, str(session[0]))
            doc['bold'].append(check_paths(
                [os.path.join(data_dir, os.path.split(str(x))[1])
                 for x in session]))
            doc['swabold'].append(check_paths(
                [prefix_filename(os.path.join(
                    data_dir, os.path.split(str(x))[1]), 'swa')
                for x in session]))
        doc['n_scans'] = [len(s) for s in doc['bold']]
    return doc


def make_design_matrices(onsets, n_scans, tr, motion=None,
                         hrf_model='canonical with derivative',
                         drift_model='cosine', orthogonalize=None):
    design_matrices = []
    for i, onset in enumerate(onsets):
        if n_scans[i] == 0:
            design_matrices.append(None)
        onset = np.array(onset)
        labels = onset[:, 0]
        time = onset[:, 1].astype('float')
        duration = onset[:, 2].astype('float')
        amplitude = onset[:, 3].astype('float')

        if duration.sum() == 0:
            paradigm = EventRelatedParadigm(labels, time, amplitude)
        else:
            paradigm = BlockParadigm(labels, time, duration, amplitude)

        frametimes = np.linspace(0, (n_scans[i] - 1) * tr, n_scans[i])

        if motion is not None:
            add_regs = np.array(motion[i]).astype('float')
            add_reg_names = ['motion_%i' % r
                             for r in range(add_regs.shape[1])]
            design_matrix = make_dmtx(
                frametimes, paradigm, hrf_model=hrf_model,
                drift_model=drift_model,
                add_regs=add_regs, add_reg_names=add_reg_names)
        else:
            design_matrix = make_dmtx(
                frametimes, paradigm, hrf_model=hrf_model,
                drift_model=drift_model)

        if orthogonalize is not None:
            if 'derivative' in hrf_model:
                raise Exception(
                    'Orthogonalization not supported with hrf derivative.')
            orth = orthogonalize[i]
            if orth is not None:
                for x, y in orth:
                    x_ = design_matrix.matrix[:, x]
                    y_ = design_matrix.matrix[:, y]
                    z = orthogonalize_vectors(x_, y_)
                    design_matrix.matrix[:, x] = z

        design_matrices.append(design_matrix.matrix)

    return design_matrices


def check_experimental_conditions(catalog):
    has_sessions = False
    conditions_ = set()
    for doc in catalog:
        if 'conditions' in doc:
            if not isinstance(doc['conditions'][0], (str, unicode)):
                has_sessions = True
                conditions = [tuple(sess) for sess in doc['conditions']]
                conditions_.add(tuple(conditions))
            else:
                conditions_.add(tuple(doc['conditions']))
    if len(conditions_) > 1:
        print ('Warning: some mat_files do not'
               ' have the same conditions.')
    if has_sessions:
        return list([list(c) for c in conditions_])[0]
    else:
        return list(list(conditions_)[0])


def check_runs(conditions):
    task_table = {}
    run_table = {}
    runs = []
    for session in conditions:
        key = hashlib.md5(str(session)).hexdigest()
        if key in task_table:
            task_id = task_table[key]
        else:
            if task_table.values() != []:
                task_id = max(task_table.values()) + 1
            else:
                task_id = 1
        task_table.setdefault(key, task_id)
        run_table.setdefault(key, []).append('run')
        run_id = len(run_table[key])
        runs.append('task%03i_run%03i' % (task_id, run_id))
    return runs


def check_tasks(runs):
    tasks = [session.split('_')[0] for session in runs]
    return dict(zip(tasks, tasks))


def check_timeseries(catalog):
    n_scans_ = set()
    for doc in catalog:
        if 'bold' in doc:
            n_scans = tuple([len(sess_bold) for sess_bold in doc['bold']])
            n_scans_.add(n_scans)
    return list(n_scans_)


class IntraLoader(object):

    def __init__(self, subject_getter, ignore=None,
                 memory=Memory(cachedir=None), n_jobs=1):
        self.subject_getter = subject_getter
        self.ignore = [] if ignore is None else ignore
        self.memory = memory
        self.n_jobs = n_jobs

    def fit(self, mat_files, subjects_id):
        self.catalog_ = Parallel(n_jobs=self.n_jobs)(
            delayed(load_intra)(mat_file, memory=self.memory,
                                subject_id=self.subject_getter)
            for mat_file in mat_files)
        self.task_contrasts_ = self.catalog_[0]['contrasts']
        self.condition_key_ = check_experimental_conditions(self.catalog_)
        self.run_key_ = check_runs(self.condition_key_)
        self.task_key_ = check_tasks(self.run_key_)
        self.n_scans_ = check_timeseries(self.catalog_)
        for key in self.ignore:
            for doc in self.catalog_:
                if key in doc:
                    del doc[key]
        return self

    def transform(self, mat_files, subjects_id):
        return self.catalog_

    def fit_transform(self, mat_files, subjects_id):
        return self.fit(
            mat_files, subjects_id).transform(mat_files, subjects_id)


class PreprocLoader(object):

    def __init__(self, subject_getter, ignore=None,
                 memory=Memory(cachedir=None), n_jobs=1):
        self.subject_getter = subject_getter
        self.ignore = [] if ignore is None else ignore
        self.memory = memory
        self.n_jobs = n_jobs

    def fit(self, mat_files, subjects_id):
        self.catalog_ = Parallel(n_jobs=self.n_jobs)(
            delayed(load_preproc)(mat_file, memory=self.memory,
                                  subject_id=self.subject_getter)
            for mat_file in mat_files)
        self.n_scans_ = check_timeseries(self.catalog_)
        for key in self.ignore:
            for doc in self.catalog_:
                if key in doc:
                    del doc[key]
        return self

    def transform(self, mat_files, subjects_id):
        return self.catalog_

    def fit_transform(self, mat_files, subjects_id):
        return self.fit(
            mat_files, subjects_id).transform(mat_files, subjects_id)


class IntraEncoder(object):

    def __init__(self, hrf_model='canonical with derivative',
                 drift_model='cosine', compute_design=True,
                 orthogonalize=None,
                 memory=Memory(cachedir=None), n_jobs=1):
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.compute_design = compute_design
        self.orthogonalize = orthogonalize
        self.memory = memory
        self.n_jobs = n_jobs

    def fit(self, catalog, subjects_id):
        if self.compute_design:
            self.design_matrices_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self.memory.cache(make_design_matrices))(
                    x['onsets'], x['n_scans'], x['tr'],
                    x['motion'], self.hrf_model, self.drift_model,
                    self.orthogonalize)
                for x in catalog)
        else:
            self.design_matrices_ = [x['design_matrices'] for x in catalog]
        return self

    def transform(self, catalog, subjects_id):
        return [x['swabold'] for x in catalog]

    def fit_transform(self, catalog, subjects_id):
        return self.fit(
            catalog, subjects_id).transform(catalog, subjects_id)


def glob_matfiles(pattern, subject_getter, ignore=None, restrict=None):
    if ignore is None:
        ignore = []
    i = 1
    doc = {}
    for mat_file in sorted(glob.glob(pattern)):
        sid = parse_path(mat_file, subject_id=subject_getter)['subject_id']
        if (restrict is None and sid not in ignore) or (
                restrict is not None and sid in restrict):
            doc.setdefault('mat_files', []).append(mat_file)
            doc.setdefault('subjects', []).append('sub%03i' % i)
            doc.setdefault('original_subjects', []).append(sid)
            i += 1
        else:
            doc.setdefault('ignored_subjects', []).append(sid)

    return doc


def orthogonalize_vectors(x, y):
    x = np.array(x)
    y = np.array(y)

    s = np.dot(x, y) / np.sum(y ** 2)

    return (x - y * s)
