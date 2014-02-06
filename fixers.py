import copy

import numpy as np

from nilearn.image.resampling import resample_img


class SPMOnsetFixer(object):

    def __init__(self, true_contrasts):
        self.true_contrasts = true_contrasts

    def fit(self, catalog, target):
        return self

    def transform(self, catalog, target):
        catalog_ = copy.deepcopy(catalog)
        for doc in catalog_:
            if not 'contrasts' in doc:
                continue
            for contrast_id in self.true_contrasts:
                if not contrast_id in doc['contrasts']:
                    continue
                for i, (con, con_ref) in enumerate(
                        zip(doc['contrasts'][contrast_id],
                            self.true_contrasts[contrast_id])):
                    con = np.array(con)
                    con_ref = np.array(con_ref)
                    if np.all(con <= 0) and not np.all(con_ref <= 0):
                        con_run = np.zeros(con_ref.size)
                        con_run[:con_ref.size] = con[:con_ref.size]
                        index = np.where((con_ref != con_run))[0][0]
                        # print i, contrast_id, index
                        new_onsets = [('cond%03i' % (index + 1), 0, 0, 0)]
                        for onset in doc['onsets'][i]:
                            onset_id, timing = onset[0], onset[1:]
                            if int(onset_id.split('cond')[1]) > index:
                                new_id = 'cond%03i' % (
                                    int(onset_id.split('cond')[1]) + 1)
                                onset = (new_id, ) + timing
                            new_onsets.append(onset)
                        doc['onsets'][i] = new_onsets
        return catalog_

    def fit_transform(self, catalog, target):
        return self.fit(catalog, target).transform(catalog, target)
