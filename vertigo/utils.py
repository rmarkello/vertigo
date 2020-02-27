# -*- coding: utf-8 -*-
"""
General utility functions
"""

import os
from pathlib import Path
import subprocess

import tqdm

try:
    from joblib import Parallel, delayed
    joblib_avail = True
except ImportError:
    joblib_avail = False


def pathify(path):
    """
    Convenience function for coercing a potential pathlike to a Path object

    Parameter
    ---------
    path : str or os.PathLike
        Path to be checked for coercion to `pathlib.Path` object

    Returns
    -------
    path : pathlib.Path
        Input `path` as a resolved `pathlib.Path` object
    """

    if isinstance(path, (str, os.PathLike)):
        path = Path(path)
    return path


def trange(n_iter, verbose=True, **kwargs):
    """
    Wrapper for `tqdm.trange` with some default options set

    Parameters
    ----------
    n_iter : int
        Number of iterations for progress bar
    verbose : bool, optional
        Whether to return an `tqdm.tqdm` progress bar instead of a range
        generator. Default: True
    kwargs
        Key-value arguments provided to `tqdm.trange`

    Returns
    -------
    progbar : tqdm.tqdm
        Progress bar instance to be used as you would `tqdm.tqdm`
    """

    form = ('{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
            ' | {elapsed}<{remaining}')
    defaults = dict(ascii=True, leave=False, bar_format=form)
    defaults.update(kwargs)

    return tqdm.trange(n_iter, disable=not verbose, **defaults)


def run(cmd, env=None, return_proc=False, quiet=False):
    """
    Runs `cmd` via shell subprocess with provided environment `env`

    Parameters
    ----------
    cmd : str
        Command to be run as single string
    env : dict, optional
        If provided, dictionary of key-value pairs to be added to base
        environment when running `cmd`. Default: None
    return_proc : bool, optional
        Whether to return CompletedProcess object. Default: false
    quiet : bool, optional
        Whether to suppress stdout/stderr from subprocess. Default: False

    Returns
    -------
    proc : subprocess.CompletedProcess
        Process output

    Raises
    ------
    subprocess.CalledProcessError
        If subprocess does not exit cleanly

    Examples
    --------
    >>> from vertigo import utils
    >>> p = utils.run('echo "hello world"', return_proc=True, quiet=True)
    >>> p.returncode
    0
    >>> p.stdout
    'hello world\\n'
    """

    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)

    opts = {}
    if quiet:
        opts = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    proc = subprocess.run(cmd, env=merged_env, shell=True, check=True,
                          universal_newlines=True, **opts)

    if return_proc:
        return proc


def parallelize(n_jobs, func, **kwargs):
    """
    Creates joblib-style parallelization function if joblib is available

    Parameters
    ----------
    n_jobs : int
        Number of processors (i.e., jobs) to use for parallelization
    func : function
        Function to parallelize
    kwargs : key-value pairs
        Passed to `joblib.Parallel`

    Returns
    -------
    parallel : `joblib.Parallel` object
        Object to parallelize over `func`
    func : `joblib.delayed`
        Provided `func` wrapped in `joblib.delayed`
    """

    def _unravel(x):
        return [f for f in x]

    defaults = dict(max_nbytes=None, mmap_mode='r+')
    defaults.update(kwargs)

    if joblib_avail:
        func = delayed(func)
        parallel = Parallel(n_jobs=n_jobs, **defaults)
    else:
        parallel = _unravel

    return parallel, func
