import os
import glob

import nibabel as nb
import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from joblib import Memory

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.grid_search import GridSearchCV

from nilearn.input_data import NiftiMasker

from nignore.decoding import Decoder
from nignore.reporting import Reporter


def get_labels():
    labels = {}
    for l in open('rolling_terms.csv').readlines():
        l = l.replace('"', "").replace('\n', '').split(',')
        labels[l[1]] = tuple(l[2:])
    return labels


mask_array = nb.load(
    '/volatile/tpm/grey_resampled.nii').get_data() > .3
affine = nb.load('/volatile/tpm/grey_resampled.nii').get_affine()
mask = nb.Nifti1Image(mask_array.astype('float'), affine=affine)
nb.save(mask, '/tmp/mask.nii.gz')

data_dir = '/volatile/brainpedia/neurospin/pinel2009twins'

names = [
    'computation_vs_rest',
    'saccade_vs_rest',
    'digit_vs_rest',
    'scramble_vs_rest',
    'words_vs_rest',
    'face_vs_rest',
    'French_vs_rest',
    'Korean_vs_rest',
    'house_vs_rest',
    'sound_vs_rest', ]

images = glob.glob(os.path.join(
    data_dir, 'sub???', 'model', 'model002', 't_maps'))

labels = get_labels()

niimgs = []
target = []
for image_dir in images:
    for name in names:
        path = os.path.join(image_dir, '%s.nii.gz' % name)
        niimgs.append(path)
        key = name.split('_vs_rest')[0].lower()
        target.append(labels[key])
        # target.append(name)

if __name__ == '__main__':
    memory = Memory('/havoc/cache', mmap_mode='r+')

    le = LabelEncoder()
    lb = LabelBinarizer()
    loader = NiftiMasker(mask='/tmp/mask.nii.gz',
                         memory=memory, memory_level=1)
    reporter = Reporter(report_dir='/tmp/reporter')

    cv = ShuffleSplit(len(target), n_iter=5)
    Cs = [1e-3, 1e-2, 1e-1, 1., 10, 1e2, 1e3]

    scaler = StandardScaler()
    n_x, n_y, n_z = mask.shape
    connectivity = grid_to_graph(n_x, n_y, n_z, mask=mask_array)
    ward = WardAgglomeration(n_clusters=2000,
                             connectivity=connectivity, memory=memory)
    svc = LinearSVC(penalty='l1', dual=False)
    # rand_svc = RandomizedWardClassifier(mask_array, n_iter=16,
    #                                     memory=memory, n_jobs=-1)

    pipe = Pipeline([('scaler', scaler), ('clf', svc)])
    grid = GridSearchCV(pipe, param_grid={'clf__C': Cs},
                        cv=cv, n_jobs=1)
    grid.best_estimator_ = grid.estimator
    ovr = OneVsRestClassifier(grid, n_jobs=1)

    # decoder = Decoder(ovr, loader, lb, reporter)
    # decoder.fit(niimgs, target).score(niimgs, target)

    # pipeline = Pipeline([('scaler', scaler), ('clf', clf)])
    decoder = Decoder(ovr, loader, lb, reporter)
    decoder.fit(niimgs, target).score(niimgs, target)
