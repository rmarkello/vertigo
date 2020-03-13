# -*- coding: utf-8 -*-
"""
Functions for working with FreeSurfer data and parcellations
"""

import os
import warnings

from nibabel.freesurfer import read_annot, read_geometry
import numpy as np
import scipy.sparse as ssp
from scipy.ndimage.measurements import _stats, labeled_comprehension
from scipy.spatial.distance import cdist

from . import mesh
from .datasets import fetch_fsaverage
from .spins import gen_spinsamples
from .utils import pathify

FSEXT = ['.orig', '.white', '.smoothwm', '.pial', '.inflated', '.sphere']
FSIGNORE = [
    'unknown', 'corpuscallosum', 'Background+FreeSurfer_Defined_Medial_Wall',
    '???'
]


def _decode_list(vals):
    """ List decoder
    """

    return [l.decode() if hasattr(l, 'decode') else l for l in vals]


def _check_fs_subjid(subject_id, subjects_dir=None):
    """
    Checks that `subject_id` exists in provided FreeSurfer `subjects_dir`

    Parameters
    ----------
    subject_id : str
        FreeSurfer subject ID
    subjects_dir : str or os.PathLike, optional
        Path to FreeSurfer subject directory. If not set, will inherit from
        the environmental variable $SUBJECTS_DIR. Default: None

    Returns
    -------
    subject_id : pathlib.Path
        FreeSurfer subject ID, as provided
    subjects_dir : pathlib.Path
        Full filepath to `subjects_dir`

    Raises
    ------
    FileNotFoundError
    """

    # check inputs for subjects_dir and subject_id
    if subjects_dir is None or not os.path.isdir(subjects_dir):
        try:
            subjects_dir = os.environ['SUBJECTS_DIR']
        except KeyError:
            subjects_dir = os.getcwd()
    else:
        subjects_dir = os.path.abspath(subjects_dir)

    subjects_dir = pathify(subjects_dir)
    subjdir = subjects_dir / subject_id
    if not subjdir.is_dir():
        raise FileNotFoundError('Cannot find specified subject id {} in '
                                'provided subject directory {}.'
                                .format(subject_id, subjects_dir))

    return subject_id, subjects_dir


def find_parcel_centroids(*, lhannot, rhannot, method='surface',
                          version='fsaverage', surf='sphere',
                          drop_labels=None):
    """
    Returns vertex coords corresponding to centroids of parcels in annotations

    Note that using any other `surf` besides the default of 'sphere' may result
    in centroids that are not directly within the parcels themselves due to
    sulcal folding patterns.

    Parameters
    ----------
    {lh,rh}annot : str
        Path to .annot file containing labels of parcels on the {left,right}
        hemisphere. These must be specified as keyword arguments to avoid
        accidental order switching.
    method : {'average', 'surface', 'geodesic'}, optional
        Method for calculation of parcel centroid. See Notes for more
        information. Default: 'surface'
    version : str, optional
        Specifies which version of `fsaverage` provided annotation files
        correspond to. Must be one of {'fsaverage', 'fsaverage3', 'fsaverage4',
        'fsaverage5', 'fsaverage6'}. Default: 'fsaverage'
    surf : str, optional
        Specifies which surface projection of fsaverage to use for finding
        parcel centroids. Default: 'sphere'
    drop_labels : list, optional
        Specifies regions in {lh,rh}annot for which the parcel centroid should
        not be calculated. If not specified, centroids for parcels defined in
        `netneurotools.freesurfer.FSIGNORE` are not calculated. Default: None

    Returns
    -------
    centroids : (N, 3) numpy.ndarray
        xyz coordinates of vertices closest to the centroid of each parcel
        defined in `lhannot` and `rhannot`
    hemiid : (N,) numpy.ndarray
        Array denoting hemisphere designation of coordinates in `centroids`,
        where `hemiid=0` denotes the left and `hemiid=1` the right hemisphere

    Notes
    -----
    The following methods can be used for finding parcel centroids:

    1. ``method='average'``

       Uses the arithmetic mean of the coordinates for the vertices in each
       parcel. Note that in this case the calculated centroids will not act
       actually fall on the surface of `surf`.

    2. ``method='surface'``

       Calculates the 'average' coordinates and then finds the closest vertex
       on `surf`, where closest is defined as the vertex with the minimum
       Euclidean distance.

    3. ``method='geodesic'``

       Uses the coordinates of the vertex with the minimum average geodesic
       distance to all other vertices in the parcel. Note that this is slightly
       more time-consuming than the other two methods, especially for
       high-resolution meshes.
    """

    methods = ['average', 'surface', 'geodesic']
    if method not in methods:
        raise ValueError('Provided method for centroid calculation {} is '
                         'invalid. Must be one of {}'.format(methods, methods))

    if drop_labels is None:
        drop_labels = FSIGNORE
    drop_labels = _decode_list(drop_labels)

    surfaces = fetch_fsaverage(version)[surf]

    centroids, hemiid = [], []
    for n, (annot, surf) in enumerate(zip([lhannot, rhannot], surfaces)):
        vertices, faces = read_geometry(surf)
        labels, ctab, names = read_annot(annot)
        names = _decode_list(names)

        for lab in np.unique(labels):
            if names[lab] in drop_labels:
                continue
            if method in ['average', 'surface']:
                roi = np.atleast_2d(vertices[labels == lab].mean(axis=0))
                if method == 'surface':  # find closest vertex on the sphere
                    roi = vertices[np.argmin(cdist(vertices, roi), axis=0)[0]]
            elif method == 'geodesic':
                inds, = np.where(labels == lab)
                roi = _geodesic_parcel_centroid(vertices, faces, inds)
            centroids.append(roi)
            hemiid.append(n)

    return np.row_stack(centroids), np.asarray(hemiid)


def _geodesic_parcel_centroid(vertices, faces, inds):
    """
    Calculates parcel centroids based on surface distance

    Parameters
    ----------
    vertices : (N, 3)
        Coordinates of vertices defining surface
    faces : (F, 3)
        Triangular faces defining surface
    inds : (R,)
        Indices of `vertices` that belong to parcel

    Returns
    --------
    roi : (3,) numpy.ndarray
        Vertex corresponding to centroid of parcel
    """

    # only retain the faces with at least one vertex in `inds`
    keep = np.sum(np.isin(faces, inds), axis=1) > 1
    mat = mesh.make_surf_graph(vertices, faces[keep])
    paths = ssp.csgraph.dijkstra(mat, directed=False, indices=inds)[:, inds]

    # the selected vertex is the one with the minimum average shortest path
    # to the other vertices in the parcel
    roi = vertices[inds[paths.mean(axis=1).argmin()]]

    return roi


def parcels_to_vertices(data, *, lhannot, rhannot, drop_labels=None):
    """
    Projects parcellated `data` to vertices defined in annotation files

    Assigns np.nan to all ROIs in `drop_labels`

    Parameters
    ----------
    data : (N,) numpy.ndarray
        Parcellated data to be projected to vertices. Parcels should be ordered
        by [left, right] hemisphere; ordering within hemisphere should
        correspond to the provided annotation files.
    {lh,rh}annot : str
        Path to .annot file containing labels of parcels on the {left,right}
        hemisphere. These must be specified as keyword arguments to avoid
        accidental order switching.
    drop_labels : list, optional
        Specifies regions in {lh,rh}annot that are not present in `data`. NaNs
        will be inserted in place of the these regions in the returned data. If
        not specified, parcels defined in `netneurotools.freesurfer.FSIGNORE`
        are assumed to not be present. Default: None

    Returns
    ------
    projected : numpy.ndarray
        Vertex-level data
    """

    if drop_labels is None:
        drop_labels = FSIGNORE
    drop_labels = _decode_list(drop_labels)

    data = np.vstack(data)

    # check this so we're not unduly surprised by anything...
    n_vert = expected = 0
    for a in [lhannot, rhannot]:
        vn, _, names = read_annot(a)
        n_vert += len(vn)
        names = _decode_list(names)
        expected += len(names) - len(set(drop_labels) & set(names))
    if expected != len(data):
        raise ValueError('Number of parcels in provided annotation files '
                         'differs from size of parcellated data array.\n'
                         '    EXPECTED: {} parcels\n'
                         '    RECEIVED: {} parcels'
                         .format(expected, len(data)))

    projected = np.zeros((n_vert, data.shape[-1]), dtype=data.dtype)
    start = end = n_vert = 0
    for annot in [lhannot, rhannot]:
        # read files and update end index for `data`
        labels, ctab, names = read_annot(annot)
        names = _decode_list(names)
        todrop = set(names) & set(drop_labels)
        end += len(names) - len(todrop)  # unknown and corpuscallosum

        # get indices of unknown and corpuscallosum and insert NaN values
        inds = sorted([names.index(f) for f in todrop])
        inds = [f - n for n, f in enumerate(inds)]
        currdata = np.insert(data[start:end], inds, np.nan, axis=0)

        # project to vertices and store
        projected[n_vert:n_vert + len(labels), :] = currdata[labels]
        start = end
        n_vert += len(labels)

    return np.squeeze(projected)


def vertices_to_parcels(data, *, lhannot, rhannot, drop_labels=None):
    """
    Reduces vertex-level `data` to parcels defined in annotation files

    Takes average of vertices within each parcel, excluding np.nan values
    (i.e., np.nanmean). Assigns np.nan to parcels for which all vertices are
    np.nan.

    Parameters
    ----------
    data : (N,) numpy.ndarray
        Vertex-level data to be reduced to parcels
    {lh,rh}annot : str
        Path to .annot file containing labels to parcels on the {left,right}
        hemisphere
    drop_labels : list, optional
        Specifies regions in {lh,rh}annot that should be removed from the
        parcellated version of `data`. If not specified, vertices corresponding
        to parcels defined in `netneurotools.freesurfer.FSIGNORE` will be
        removed. Default: None

    Reurns
    ------
    reduced : numpy.ndarray
        Parcellated `data`, without regions specified in `drop`
    """

    if drop_labels is None:
        drop_labels = FSIGNORE
    drop_labels = _decode_list(drop_labels)

    data = np.vstack(data)

    n_parc = expected = 0
    for a in [lhannot, rhannot]:
        vn, _, names = read_annot(a)
        expected += len(vn)
        names = _decode_list(names)
        n_parc += len(names) - len(set(drop_labels) & set(names))
    if expected != len(data):
        raise ValueError('Number of vertices in provided annotation files '
                         'differs from size of vertex-level data array.\n'
                         '    EXPECTED: {} vertices\n'
                         '    RECEIVED: {} vertices'
                         .format(expected, len(data)))

    reduced = np.zeros((n_parc, data.shape[-1]), dtype=data.dtype)
    start = end = n_parc = 0
    for annot in [lhannot, rhannot]:
        # read files and update end index for `data`
        labels, ctab, names = read_annot(annot)
        names = _decode_list(names)

        indices = np.unique(labels)
        end += len(labels)

        for idx in range(data.shape[-1]):
            # get average of vertex-level data within parcels
            # set all NaN values to 0 before calling `_stats` because we are
            # returning sums, so the 0 values won't impact the sums (if we left
            # the NaNs then all parcels with even one NaN entry would be NaN)
            currdata = np.squeeze(data[start:end, idx])
            isna = np.isnan(currdata)
            counts, sums = _stats(np.nan_to_num(currdata), labels, indices)

            # however, we do need to account for the NaN values in the counts
            # so that our means are similar to what we'd get from e.g.,
            # np.nanmean here, our "sums" are the counts of NaN values in our
            # parcels
            _, nacounts = _stats(isna, labels, indices)
            counts = (np.asanyarray(counts, dtype=float)
                      - np.asanyarray(nacounts, dtype=float))

            with np.errstate(divide='ignore', invalid='ignore'):
                currdata = sums / counts

            # get indices of unkown and corpuscallosum and delete from parcels
            inds = sorted([
                names.index(f) for f in set(drop_labels) & set(names)
            ])
            currdata = np.delete(currdata, inds)

            # store parcellated data
            reduced[n_parc:n_parc + len(names) - len(inds), idx] = currdata
        start = end
        n_parc += len(names) - len(inds)

    return np.squeeze(reduced)


def _get_fsaverage_coords(version='fsaverage', surface='sphere'):
    """
    Gets vertex coordinates for specified `surface` of fsaverage `version`

    Parameters
    ----------
    version : str, optional
        One of {'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5',
        'fsaverage6'}. Default: 'fsaverage'
    surface : str, optional
        Surface for which to return vertex coordinates. Default: 'sphere'

    Returns
    -------
    coords : (N, 3) numpy.ndarray
        xyz coordinates of vertices for {left,right} hemisphere
    hemiid : (N,) numpy.ndarray
        Array denoting hemisphere designation of entries in `coords`, where
        `hemiid=0` denotes the left and `hemiid=1` the right hemisphere
    """

    # get coordinates and hemisphere designation for spin generation
    lhsphere, rhsphere = fetch_fsaverage(version)[surface]
    coords, hemi = [], []
    for n, sphere in enumerate([lhsphere, rhsphere]):
        coords.append(read_geometry(sphere)[0])
        hemi.append(np.ones(len(coords[-1])) * n)

    return np.row_stack(coords), np.hstack(hemi)


def _get_fsaverage_spins(version='fsaverage', spins=None, n_rotate=1000,
                         **kwargs):
    """
    Generates spatial permutation resamples for fsaverage `version`

    If `spins` are provided then performs checks to confirm they are valid

    Parameters
    ----------
    version : str, optional
        Specifies which version of `fsaverage` for which to generate spins.
        Must be one of {'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5',
        'fsaverage6'}. Default: 'fsaverage'
    spins : array_like, optional
        Pre-computed spins to use instead of generating them on the fly. If not
        provided will use other provided parameters to create them. Default:
        None
    n_rotate : int, optional
        Number of rotations to generate. Default: 1000
    return_cost : bool, optional
        Whether to return cost array (specified as Euclidean distance) for each
        coordinate for each rotation. Currently this option is not supported if
        pre-computed `spins` are provided. Default: True
    kwargs : key-value pairs
        Keyword arguments passed to `netneurotools.stats.gen_spinsamples`

    Returns
    --------
    spins : (N, S) numpy.ndarray
        Resampling array
    """

    if spins is None:
        coords, hemiid = _get_fsaverage_coords(version, 'sphere')
        spins = gen_spinsamples(coords, hemiid, n_rotate=n_rotate,
                                **kwargs)
        if kwargs.get('return_cost', False):
            return spins

    spins = np.asarray(spins, dtype='int32')
    if spins.shape[-1] != n_rotate:
        warnings.warn('Shape of provided `spins` array does not match '
                      'number of rotations requested with `n_rotate`. '
                      'Ignoring specified `n_rotate` parameter and using '
                      'all provided `spins`.')
        n_rotate = spins.shape[-1]

    return spins, None


def spin_data(data, *, lhannot, rhannot, version='fsaverage', n_rotate=1000,
              spins=None, drop_labels=None, verbose=False, **kwargs):
    """
    Projects parcellated `data` to surface, rotates, and re-parcellates

    Projection to the surface uses `{lh,rh}annot` files. Rotation uses vertex
    coordinates from the specified fsaverage `version` and relies on
    :func:`netneurotools.stats.gen_spinsamples`. Re-parcellated data will not
    be exactly identical to original values due to re-averaging process.
    Parcels subsumed by regions in `drop_labels` will be listed as NaN.

    Parameters
    ----------
    data : (N,) numpy.ndarray
        Parcellated data to be rotated. Parcels should be ordered by [left,
        right] hemisphere; ordering within hemisphere should correspond to the
        provided `{lh,rh}annot` annotation files.
    {lh,rh}annot : str
        Path to .annot file containing labels to parcels on the {left,right}
        hemisphere
    version : str, optional
        Specifies which version of `fsaverage` provided annotation files
        correspond to. Must be one of {'fsaverage', 'fsaverage3', 'fsaverage4',
        'fsaverage5', 'fsaverage6'}. Default: 'fsaverage'
    n_rotate : int, optional
        Number of rotations to generate. Default: 1000
    spins : array_like, optional
        Pre-computed spins to use instead of generating them on the fly. If not
        provided will use other provided parameters to create them. Default:
        None
    drop_labels : list, optional
        Specifies regions in {lh,rh}annot that are not present in `data`. NaNs
        will be inserted in place of the these regions in the returned data. If
        not specified, parcels defined in `netneurotools.freesurfer.FSIGNORE`
        are assumed to not be present. Default: None
    verbose : bool, optional
        Whether to print occasional status messages. Default: False
    kwargs : key-value pairs
        Keyword arguments passed to `netneurotools.stats.gen_spinsamples`

    Returns
    -------
    rotated : (N, `n_rotate`) numpy.ndarray
        Rotated `data
    cost : (N, `n_rotate`,) numpy.ndarray
        Cost (specified as Euclidean distance) of re-assigning each coordinate
        for every rotation in `spinsamples`. Only provided if `return_cost` is
        True.
    """

    if drop_labels is None:
        drop_labels = FSIGNORE

    # get coordinates and hemisphere designation for spin generation
    vertices = parcels_to_vertices(data, lhannot=lhannot, rhannot=rhannot,
                                   drop_labels=drop_labels)

    # get spins + cost (if requested)
    spins, cost = _get_fsaverage_spins(version=version, spins=spins,
                                       n_rotate=n_rotate,
                                       verbose=verbose, **kwargs)
    if len(vertices) != len(spins):
        raise ValueError('Provided annotation files have a different '
                         'number of vertices than the specified fsaverage '
                         'surface.\n    ANNOTATION: {} vertices\n     '
                         'FSAVERAGE:  {} vertices'
                         .format(len(vertices), len(spins)))

    spun = np.zeros(data.shape + (n_rotate,))
    for n in range(n_rotate):
        if verbose:
            msg = f'Reducing vertices to parcels: {n:>5}/{n_rotate}'
            print(msg, end='\b' * len(msg), flush=True)
        spun[..., n] = vertices_to_parcels(vertices[spins[:, n]],
                                           lhannot=lhannot, rhannot=rhannot,
                                           drop_labels=drop_labels)

    if verbose:
        print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)

    if kwargs.get('return_cost', False):
        return spun, cost

    return spun


def spin_parcels(*, lhannot, rhannot, version='fsaverage', n_rotate=1000,
                 spins=None, drop_labels=None, verbose=False, **kwargs):
    """
    Rotates parcels in `{lh,rh}annot` and re-assigns based on maximum overlap

    Vertex labels are rotated with :func:`netneurotools.stats.gen_spinsamples`
    and a new label is assigned to each *parcel* based on the region maximally
    overlapping with its boundaries.

    Parameters
    ----------
    {lh,rh}annot : str
        Path to .annot file containing labels to parcels on the {left,right}
        hemisphere
    version : str, optional
        Specifies which version of `fsaverage` provided annotation files
        correspond to. Must be one of {'fsaverage', 'fsaverage3', 'fsaverage4',
        'fsaverage5', 'fsaverage6'}. Default: 'fsaverage'
    n_rotate : int, optional
        Number of rotations to generate. Default: 1000
    spins : array_like, optional
        Pre-computed spins to use instead of generating them on the fly. If not
        provided will use other provided parameters to create them. Default:
        None
    drop_labels : list, optional
        Specifies regions in {lh,rh}annot that are not present in `data`. NaNs
        will be inserted in place of the these regions in the returned data. If
        not specified, parcels defined in `netneurotools.freesurfer.FSIGNORE`
        are assumed to not be present. Default: None
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None
    verbose : bool, optional
        Whether to print occasional status messages. Default: False
    return_cost : bool, optional
        Whether to return cost array (specified as Euclidean distance) for each
        coordinate for each rotation. Default: True
    kwargs : key-value pairs
        Keyword arguments passed to `netneurotools.stats.gen_spinsamples`

    Returns
    -------
    spinsamples : (N, `n_rotate`) numpy.ndarray
        Resampling matrix to use in permuting data parcellated with labels from
        {lh,rh}annot, where `N` is the number of parcels. Indices of -1
        indicate that the parcel was completely encompassed by regions in
        `drop` and should be ignored.
    cost : (N, `n_rotate`,) numpy.ndarray
        Cost (specified as Euclidean distance) of re-assigning each coordinate
        for every rotation in `spinsamples`. Only provided if `return_cost` is
        True.
    """

    def overlap(vals):
        """ Returns most common non-negative value in `vals`; -1 if all neg
        """
        vals = np.asarray(vals)
        vals, counts = np.unique(vals[vals > 0], return_counts=True)
        try:
            return vals[counts.argmax()]
        except ValueError:
            return -1

    if drop_labels is None:
        drop_labels = FSIGNORE
    drop_labels = _decode_list(drop_labels)

    # get vertex-level labels (set drop labels to - values)
    vertices, end = [], 0
    for n, annot in enumerate([lhannot, rhannot]):
        labels, ctab, names = read_annot(annot)
        names = _decode_list(names)
        todrop = set(names) & set(drop_labels)
        inds = [names.index(f) - n for n, f in enumerate(todrop)]
        labs = np.arange(len(names) - len(inds)) + (end - (len(inds) * n))
        insert = np.arange(-1, -(len(inds) + 1), -1)
        vertices.append(np.insert(labs, inds, insert)[labels])
        end += len(names)
    vertices = np.hstack(vertices)
    labels = np.unique(vertices)
    mask = labels > -1

    # get spins + cost (if requested)
    spins, cost = _get_fsaverage_spins(version=version, spins=spins,
                                       n_rotate=n_rotate, verbose=verbose,
                                       **kwargs)
    if len(vertices) != len(spins):
        raise ValueError('Provided annotation files have a different '
                         'number of vertices than the specified fsaverage '
                         'surface.\n    ANNOTATION: {} vertices\n     '
                         'FSAVERAGE:  {} vertices'
                         .format(len(vertices), len(spins)))

    # spin and assign regions based on max overlap
    regions = np.zeros((len(labels[mask]), n_rotate), dtype='int32')
    for n in range(n_rotate):
        if verbose:
            msg = f'Calculating parcel overlap: {n:>5}/{n_rotate}'
            print(msg, end='\b' * len(msg), flush=True)
        regions[:, n] = labeled_comprehension(vertices[spins[:, n]], vertices,
                                              labels, overlap, int, -1)[mask]

    if kwargs.get('return_cost', False):
        return regions, cost

    return regions
