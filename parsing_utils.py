import os
import re


def find_data_dir(wd, fpath):
    fpath = fpath.strip()

    def right_splits(p):
        yield p
        while p not in ['', None]:
            p = p.rsplit(os.path.sep, 1)[0]
            yield p

    def left_splits(p):
        yield p
        while len(p.split(os.path.sep, 1)) > 1:
            p = p.split(os.path.sep, 1)[1]
            yield p

    if not os.path.isfile(fpath):
        for rs in right_splits(wd):
            if not os.path.exists(rs):
                continue
            for ls in left_splits(fpath):
                p = os.path.join(rs, *ls.split(os.path.sep))
                if os.path.isfile(p):
                    return os.path.dirname(p)
    else:
        return os.path.dirname(fpath)
    return ''


def makeup_path(work_dir, path):
    data_dir = find_data_dir(work_dir, str(path))
    return os.path.join(data_dir, os.path.split(str(path))[1])


def prefix_filename(path, prefix):
    path, filename = os.path.split(str(path))
    return os.path.join(path, '%s%s' % (prefix, filename))


def strip_prefix_filename(path, len_strip):
    path, filename = os.path.split(str(path))
    return os.path.join(path, filename[len_strip:])


def remove_special(name):
    return re.sub("[^0-9a-zA-Z\-]+", '_', name)


def parse_path(path, **kwargs):
    doc = {}
    for k in kwargs:
        if hasattr(kwargs[k], '__call__'):
            doc[k] = kwargs[k](path)
        elif isinstance(kwargs[k], int):
            doc[k] = path.split(os.path.sep)[kwargs[k]]
        else:
            doc[k] = kwargs[k]
    return doc
