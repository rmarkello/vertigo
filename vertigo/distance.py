# -*- coding: utf-8 -*-
"""
Functions for calculating geodesic parcel distance
"""

import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage, sparse

from . import io, mesh
from .freesurfer import FSIGNORE
from .utils import pathify, parallelize, trange, run


def _get_workbench_distance(vertex, surf, labels=None):
    """
    Gets surface distance of `vertex` to all other vertices in `surf`

    Parameters
    ----------
    vertex : int
        Index of vertex for which to calculate surface distance
    surf : str or os.PathLike
        Path to surface file on which to calculate distance
    labels : array_like, optional
        Labels indicating parcel to which each vertex belongs. If provided,
        distances will be averaged within distinct labels

    Returns
    -------
    dist : (N,) numpy.ndarray
        Distance of `vertex` to all other vertices in `graph` (or to all
        parcels in `labels`, if provided)
    """

    distcmd = 'wb_command -surface-geodesic-distance {surf} {vertex} {out}'

    # run the geodesic distance command with wb_command
    with tempfile.NamedTemporaryFile(suffix='.func.gii') as out:
        run(distcmd.format(surf=surf, vertex=vertex, out=out.name), quiet=True)
        dist = nib.load(out.name).agg_data()

    return _get_parcel_distance(vertex, dist, labels)


def _get_graph_distance(vertex, graph, labels=None):
    """
    Gets surface distance of `vertex` to all other vertices in `graph`

    Parameters
    ----------
    vertex : int
        Index of vertex for which to calculate surface distance
    graph : array_like
        Graph along which to calculate shortest path distances
    labels : array_like, optional
        Labels indicating parcel to which each vertex belongs. If provided,
        distances will be averaged within distinct labels

    Returns
    -------
    dist : (N,) numpy.ndarray
        Distance of `vertex` to all other vertices in `graph` (or to all
        parcels in `labels`, if provided)
    """

    # this involves an up-cast to float64; we're gonne get some rounding diff
    # here when compared to the wb_command subprocess call
    dist = sparse.csgraph.dijkstra(graph, directed=False, indices=[vertex])

    return _get_parcel_distance(vertex, dist, labels)


def _get_parcel_distance(vertex, dist, labels=None):
    """
    Averages `dist` within `labels`, if provided

    Parameters
    ----------
    vertex : int
        Index of vertex used to calculate `dist`
    dist : (N,) array_like
        Distance of `vertex` to all other vertices
    labels : (N,) array_like, optional
        Labels indicating parcel to which each vertex belongs. If provided,
        `dist` will be average within distinct labels.

    Returns
    -------
    dist : numpy.ndarray
        Distance (vertex or parcels), converted to float32
    """

    if labels is not None:
        dist = ndimage.mean(input=np.delete(dist, vertex),
                            labels=np.delete(labels, vertex),
                            index=np.unique(labels))

    return dist.astype(np.float32)


def get_surface_distance(surf, dlabel=None, medial=None, drop_labels=None,
                         use_wb=False, n_proc=1, verbose=False):
    """
    Calculates surface distance for vertices in `surf`

    Parameters
    ----------
    surf : str or os.PathLike
        Path to surface file on which to calculate distance
    dlabel : str or os.PathLike, optional
        Path to file with parcel labels for provided `surf`. If provided will
        calculate parcel-parcel distances instead of vertex distances. Default:
        None
    medial : str or os.PathLike, optional
        Path to file containing labels for vertices corresponding to medial
        wall. If provided (and `use_wb=False`), will disallow calculation of
        surface distance along the medial wall. Distances for vertices on the
        medial wall will be set to inf. Default: None
    drop_labels : list of str, optional
        List of parcel names that should be dropped from the final distance
        matrix (if `dlabel` is provided). If not specified, will ignore all
        parcels commonly used to reference the medial wall (e.g., 'unknown',
        'corpuscallosum', '???', 'Background+FreeSurfer_Defined_Medial_Wall').
        Default: None
    use_wb : bool, optional
        Whether to use calls to `wb_command -surface-geodesic-distance` for
        computation of the distance matrix; this will involve significant disk
        I/O. If False, all computations will be done in memory using the
        `scipy.sparse.csgraph.dijkstra` function. Default: False
    n_proc : int, optional
        Number of processors to use for parallelizing distance calculation. If
        negative, will use max available processors plus 1 minus the specified
        number. Only available if `joblib` is installed. Default: 1 (no
        parallelization)
    verbose : bool, optional
        Whether to print progress bar while distances are calculated. Default:
        True

    Returns
    -------
    distance : (N, N) numpy.ndarray
        Surface distance between vertices/parcels on `surf`

    Notes
    -----
    The distance matrix computed with `use_wb=False` will have slightly lower
    values than when `use_wb=True` due to known estimation errors. These will
    be fixed at a later date.
    """

    if drop_labels is None:
        drop_labels = FSIGNORE

    # convert to paths, if necessary
    surf, dlabel, medial = pathify(surf), pathify(dlabel), pathify(medial)

    # wb_command requires gifti files so convert if we receive e.g., a FS file
    # also return a "remove" flag that will be used to delete the temporary
    # gifti file at the end of this process
    surf, remove_surf = io.surf_to_gii(surf)
    n_vert = len(nib.load(surf).agg_data()[0])

    # check if dlabel / medial wall files were provided
    labels, mask = None, np.zeros(n_vert, dtype=bool)
    dlabel, remove_dlabel = io.labels_to_gii(dlabel)
    medial, remove_medial = io.labels_to_gii(medial)

    # get data from dlabel / medial wall files if they provided
    if dlabel is not None:
        labels = nib.load(dlabel).agg_data().astype(np.int32)
    if medial is not None:
        mask = nib.load(medial).agg_data().astype(bool)

    # determine which parcels should be ignored (if they exist)
    delete = []
    if len(drop_labels) > 0 and labels is not None:
        # get vertex labels
        uniq_labels = np.unique(labels)
        n_labels = len(uniq_labels)

        # get parcel labels and reverse dict to (name : label)
        table = nib.load(dlabel).labeltable.get_labels_as_dict()
        table = {v: k for k, v in table.items()}

        # generate dict mapping label to array indices (since labels don't
        # necessarily start at 0 / aren't contiguous)
        idx = dict(zip(uniq_labels, np.arange(n_labels)))

        # get indices of parcel distance matrix to be deleted
        for lab in set(table) & set(drop_labels):
            lab = table.get(lab)
            delete.append(idx.get(lab))

    # calculate distance from each vertex to all other parcels
    if use_wb:
        parallel, parfunc = parallelize(n_proc, _get_workbench_distance)
        graph = surf
    else:
        parallel, parfunc = parallelize(n_proc, _get_graph_distance)
        graph = mesh.make_surf_graph(*nib.load(surf).agg_data(), mask=mask)
    bar = trange(n_vert, verbose=verbose, desc='Calculating distances')
    dist = np.row_stack(parallel(parfunc(n, graph, labels) for n in bar))

    # average distance for all vertices within a parcel + set diagonal to 0
    if labels is not None:
        dist = np.row_stack([
            dist[labels == lab].mean(axis=0) for lab in uniq_labels
        ])
        dist[np.diag_indices_from(dist)] = 0

    # remove distances for parcels that we aren't interested in
    if len(delete) > 0:
        for axis in range(2):
            dist = np.delete(dist, delete, axis=axis)

    # if we created gifti files then remove them
    if remove_surf:
        surf.unlink()
    if remove_dlabel:
        dlabel.unlink()
    if remove_medial:
        medial.unlink()

    return dist
