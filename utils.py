import os
import re
import csv
import copy
import glob
import shutil
import warnings

import numpy as np


def make_dir(dir_path, *optional, **kwargs):
    safe = kwargs.get('safe', True)
    strict = kwargs.get('strict', True)

    dir_path = os.path.join(dir_path, *optional)
    try:
        os.makedirs(dir_path)
    except Exception, e:
        if os.path.exists(dir_path) and not safe:
            del_dir(dir_path, safe, strict)
        elif os.path.exists(dir_path) and safe and strict:
            raise e

    return dir_path


def del_dir(dir_path, *optional):
    dir_path = os.path.join(dir_path, *optional)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def copy_dir(src_dir, dest_dir, safe=True, strict=True):
    if not os.path.exists(src_dir) and not strict:
        warnings.warn('Source directory %s does not exist.' % src_dir)
        return
    if os.path.exists(dest_dir) and not safe:
        del_dir(dest_dir)
    shutil.copytree(src_dir, dest_dir)


def copy_file(src_file, dest_file, safe=True, strict=True):
    if not os.path.exists(src_file) and not strict:
        warnings.warn('Source file %s does not exist.' % src_file)
        return
    if os.path.exists(dest_file) and safe:
        raise Exception('Destination file %s already exists.' % dest_file)
    shutil.copyfile(src_file, dest_file)


def globing(data_dir, *args, **kwargs):
    if kwargs.get('generator', False):
        glober = glob.iglob
    else:
        glober = glob.glob
    return glober(os.path.join(data_dir, *args))


def save_table(dict_obj, file_name, merge=False):
    if dict_obj is None:
        return
    mode = 'wb' if not merge else 'ab'
    with open(file_name, mode) as f:
        writer = csv.writer(f, delimiter=' ', quotechar='"')
        for key in sorted(dict_obj.keys()):
            if isinstance(dict_obj[key], list):
                writer.writerow([key] + dict_obj[key])
            else:
                writer.writerow([key, dict_obj[key]])


def get_table(file_name):
    if not os.path.exists(file_name):
        return dict()
    with open(file_name) as f:
        reader = csv.reader(f, delimiter=' ', quotechar='"')
        keys = []
        values = []
        for row in reader:
            keys.append(row[0])
            values.append(row[1])
    return dict(zip(keys, values))


def safe_name(name):
    name = re.sub('[/ \'\"!*?;(){}.]', '_', name)
    return re.sub('_+', '_', name)


def check_path(path):
    path = str(path)
    path = path.strip()
    # test separator
    if '\\' in path:
        parts = path.split('\\')
        if parts[0] == '' or ':' in parts[0]:
            parts = parts[1:]
        return os.path.join(*parts)
    # SPM8 adds the volume number at the end of the path
    if re.match('.*,\d+', path):
        return path.split(',')[0]
    return path


def check_paths(paths):
    return [check_path(path) for path in paths]


def contrasts_spec(contrasts, sessions_spec):
    new_contrasts = {}
    for k in contrasts:
        contrast = copy.deepcopy(contrasts[k])
        for i, session_spec in enumerate(sessions_spec):
            con = np.array(contrast, copy=True)
            selection = np.ones(len(con), dtype='bool')
            selection[session_spec] = False
            con[selection] = 0

            if not k.startswith('task'):
                new_k = 'task001_%s' % k
            else:
                new_k = k
            task_id, con_name = new_k.split('_', 1)
            new_k = '%s_run%03i_%s' % (task_id, i + 1, con_name)
            new_contrasts[new_k] = con.tolist()
    return new_contrasts
