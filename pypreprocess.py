import os
import glob

import nibabel as nb

from nignore.utils import make_dir, copy_file, globing, copy_dir

data_dir = "/volatile/drop/data"
intra_dir = "/lotta/brainpedia/intra_stats"
preproc_dir = "/rosie/preproc"
out_dir = "/volatile/drop/preproc"

studies = globing(preproc_dir, 'pinel2007fast')

for study_dir in studies:
    study_id = os.path.split(study_dir)[1]
    if '_' in study_id:
        data_study_id = study_id.split('_')[0]
    else:
        data_study_id = study_id

    out_study_dir = make_dir(out_dir, study_id)

    copy_dir(os.path.join(intra_dir, data_study_id, 'models', 'model002'),
             os.path.join(out_study_dir, 'models', 'model001'), safe=False)

    for txt_file in globing(data_dir, data_study_id, '*.txt'):
        copy_file(txt_file, os.path.join(
            out_study_dir,
            os.path.split(txt_file)[1]), safe=False)

    for subject_dir in glob.glob(os.path.join(study_dir, 'sub???')):
        subject_id = os.path.split(subject_dir)[1]
        print '', subject_id

        data_subject_dir = os.path.join(data_dir, data_study_id, subject_id)
        onsets_dir = os.path.join(data_subject_dir,
                                  'model', 'model001', 'onsets')

        out_model = make_dir(
            out_study_dir, subject_id, 'model', 'model001')
        copy_dir(onsets_dir, os.path.join(out_model, 'onsets'), safe=False)

        out_anat = make_dir(out_model, 'anatomy')
        out_bold = make_dir(out_model, 'BOLD')
        anat_path = os.path.join(subject_dir, 'whighres001_brain.nii')
        if not os.path.exists(anat_path):
            anat_path = os.path.join(subject_dir, 'whighres001.nii')
        anat = nb.load(anat_path)
        nb.save(anat, os.path.join(out_anat, 'highres001.nii.gz'))

        has_bold = False
        for run_dir in glob.glob(os.path.join(subject_dir, 'task???_run???')):
            has_bold = True
            run_id = os.path.split(run_dir)[1]

            out_run = make_dir(out_bold, run_id)

            bold = nb.load(glob.glob(
                os.path.join(run_dir, 'wrbold*.nii'))[0])
            nb.save(bold, os.path.join(out_run, 'bold.nii.gz'))

            copy_file(glob.glob(os.path.join(
                run_dir, 'rp_bold*.txt'))[0],
                os.path.join(out_run, 'motion.txt'), safe=False)
        if not has_bold:
            run_id = 'task001_run001'
            bold = nb.load(
                os.path.join(subject_dir, 'wrbold.nii'))
            out_run = make_dir(out_bold, run_id)
            nb.save(bold, os.path.join(out_run, 'bold.nii.gz'))
            copy_file(glob.glob(os.path.join(
                subject_dir, 'rp_bold*.txt'))[0],
                os.path.join(out_run, 'motion.txt'), safe=False)
