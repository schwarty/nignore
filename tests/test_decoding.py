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
from nilearn.input_data import NiftiMasker

from nignore.decoding_utils import Decoder
from nignore.reporting_utils import Reporter


def get_labels():
    labels = {}
    for l in open('rolling_terms.csv').readlines():
        l = l.replace('"', "").replace('\n', '').split(',')
        labels[l[1]] = tuple(l[2:])
    return labels


mask_array = nb.load(
    '/home/ys218403/Data/tpm/grey_resampled.nii').get_data() > .3
affine = nb.load('/home/ys218403/Data/tpm/grey_resampled.nii').get_affine()
mask = nb.Nifti1Image(mask_array.astype('float'), affine=affine)
nb.save(mask, '/tmp/mask.nii.gz')

data_dir = '/home/ys218403/Data/brainpedia/neurospin/pinel2009twins'
# names = [
#     'visual_calculation_vs_rest',
#     'vertical_checkerboard_vs_rest',
#     'horizontal_checkerboard_vs_rest',
#     'visual_sentences_vs_rest',
#     'visual_left_motor_vs_rest']

# names = [
#     'computation_vs_math',
#     'computation_vs_saccade',
#     'digit_vs_scramble',
#     'digit_vs_words',
#     'face_vs_house',
#     'French_vs_Korean',
#     'words_vs_house',
#     'French_vs_sound',
#     'Korean_vs_sound']

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
    memory = Memory('/havoc/cache')

    le = LabelEncoder()
    lb = LabelBinarizer()
    loader = NiftiMasker(mask='/tmp/mask.nii.gz',
                         memory=memory, memory_level=1)
    reporter = Reporter(report_dir='/tmp/reporter')
    scaler = StandardScaler()
    clf = LinearSVC()

    # cv = ShuffleSplit(target.size)
    # pipeline = Pipeline([('scaler', scaler), ('clf', clf)])
    # decoder = Decoder(pipeline, loader, le, reporter)
    # decoder.fit(niimgs, target).score(niimgs, target)

    ovr = OneVsRestClassifier(clf, n_jobs=-1)
    pipeline = Pipeline([('scaler', scaler), ('clf', ovr)])
    decoder = Decoder(pipeline, loader, lb, reporter)
    decoder.fit(niimgs, target).score(niimgs, target)
