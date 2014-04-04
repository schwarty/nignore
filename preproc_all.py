import os
import glob

module_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = '/neurospin/tmp/brainpedia/data'
base_out_dir = '/neurospin/tmp/brainpedia/new_preproc'
exec_path = '/home/ys218403/Python/lib/pypreprocess/examples/openfmri_preproc.py'

for input_dir in glob.glob(data_dir + '/*'):
    print 'input_dir:', input_dir
    job = {}

    study_id = os.path.split(input_dir)[1]
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
    job['options'] = '-O -n 12'

    with open(os.path.join(module_dir, 'job_template.qsub')) as f:
        qsub = f.read() % job
    with open(os.path.join(job_dir, 'job.qsub'), 'wb') as f:
        f.write(qsub)
    os.chmod(os.path.join(job_dir, 'job.qsub'), 0777)
    os.system('qsub %s' % (os.path.join(job_dir, 'job.qsub')))
