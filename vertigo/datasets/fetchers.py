# -*- coding: utf-8 -*-
"""
Functions for fetching datasets from the internet
"""

from collections import namedtuple
import os.path as op

from .osf import _get_data_dir, _get_dataset_info
from .utils import _fetch_files

ANNOT = namedtuple('Surface', ('lh', 'rh'))


def fetch_fsaverage(version='fsaverage', data_dir=None, url=None, resume=True,
                    verbose=1):
    """
    Downloads files for fsaverage FreeSurfer template

    Parameters
    ----------
    version : str, optional
        One of {'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5',
        'fsaverage6'}. Default: 'fsaverage'
    data_dir : str or os.PathLike, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'NNT_DATA'; if that is not set, will use
        `~/nnt-data` instead. Default: None
    url : str, optional
        URL from which to download data. Default: None
    resume : bool, optional
        Whether to attempt to resume partial download, if possible. Default:
        True
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    filenames : dict
        Dictionary with keys ['surf'] where corresponding values are length-2
        lists of downloaded template files (each list composed of files for the
        left and right hemisphere).

    References
    ----------

    """

    from ..freesurfer import _check_fs_subjid  # avoid circular import

    versions = [
        'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6'
    ]
    if version not in versions:
        raise ValueError('The version of fsaverage requested "{}" does not '
                         'exist. Must be one of {}'.format(version, versions))

    dataset_name = 'tpl-fsaverage'
    keys = ['orig', 'white', 'smoothwm', 'pial', 'inflated', 'sphere']

    data_dir = _get_data_dir(data_dir=data_dir)
    info = _get_dataset_info(dataset_name)[version]
    if url is None:
        url = info['url']

    opts = {
        'uncompress': True,
        'md5sum': info['md5'],
        'move': '{}.tar.gz'.format(dataset_name)
    }

    filenames = [
        op.join(version, 'surf', '{}.{}'.format(hemi, surf))
        for surf in keys for hemi in ['lh', 'rh']
    ]

    try:
        data_dir = _check_fs_subjid(version)[1]
        data = [op.join(data_dir, f) for f in filenames]
    except FileNotFoundError:
        data = _fetch_files(data_dir, resume=resume, verbose=verbose,
                            files=[(op.join(dataset_name, f), url, opts)
                                   for f in filenames])

    data = [ANNOT(*data[i:i + 2]) for i in range(0, len(keys) * 2, 2)]

    return dict(zip(keys, data))
