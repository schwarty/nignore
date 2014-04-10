import os
import glob

module_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = '/storage/workspace/brainpedia/data'
base_out_dir = '/storage/workspace/yschwart/keeporient_preproc'
exec_path = '/storage/workspace/yschwart/python/pypreprocess/examples/openfmri_preproc.py'

for input_dir in glob.glob(data_dir + '/*'):
    print 'input_dir:', input_dir
    job = {}

    study_id = os.path.split(input_dir)[1]
    if study_id == 'ds113' or study_id == 'fBIRN':
        print 'skip'
        continue

    output_dir = base_out_dir + '/%s' % study_id
    print 'output_dir:', output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    job_dir = output_dir

    job['input_dir'] = input_dir
    job['output_dir'] = output_dir
    job['job_dir'] = job_dir
    job['job_name'] = 'preproc_%s' % study_id
    job['script'] = exec_path
    job['options'] = '-n 48'

    with open(os.path.join(module_dir, 'job_template.qsub')) as f:
        qsub = f.read() % job
    with open(os.path.join(job_dir, 'job.qsub'), 'wb') as f:
        f.write(qsub)
    os.chmod(os.path.join(job_dir, 'job.qsub'), 0777)
    os.system('. %s' % (os.path.join(job_dir, 'job.qsub')))
