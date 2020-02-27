# -*- coding: utf-8 -*-
"""
Functions for I/O and file format conversion
"""

import tempfile

import nibabel as nib

from .utils import pathify

CIFTITOGIFTI = 'wb_command -cifti-separate {cifti} COLUMN -label {hemi} {gii}'


def _decode(x):
    """ Decodes `x` if it has the `.decode()` method
    """
    return x.decode() if hasattr(x, 'decode') else x


def _annot_to_gii(annot, out=None):
    """
    Converts FreeSurfer-style annotation file to gifti format

    Parameters
    ----------
    annot : str or os.PathLike
        Path to FreeSurfer annotation file
    out : str or os.PathLike
        Path to desired output file location. If not specified will use same
        directory as `annot` and replace file extension with `.dlabel.gii`

    Returns
    -------
    gifti : pathlib.Path
        Path to generated gifti file
    """

    # path handling
    annot = pathify(annot)
    if out is None:
        out = annot.with_suffix('.label.gii')

    # convert annotation to gifti internally, with nibabel
    labels, ctab, names = nib.freesurfer.read_annot(annot)
    darray = nib.gifti.GiftiDataArray(labels, 'NIFTI_INTENT_LABEL',
                                      datatype='NIFTI_TYPE_INT32')
    # create label table used ctab and names from annotation file
    labtab = nib.gifti.GiftiLabelTable()
    for n, (r, g, b, a) in enumerate(ctab[:, :-1]):
        lab = nib.gifti.GiftiLabel(n, r, g, b, a)
        lab.label = _decode(names[n])
        labtab.labels.append(lab)
    gii = nib.gifti.GiftiImage(labeltable=labtab, darrays=[darray])

    nib.save(gii, out)

    return out


def _dlabel_to_gii(dlabel, out=None, use_wb=False):
    """
    Converts CIFTI-2 surface file to gifti format

    Parameters
    ----------
    dlabel : str or os.PathLike
        Path to CIFTI dlabel file
    out : str or os.PathLike
        Path to desired output file location. If not specified will use same
        directory as `surf` and replace file extension with `.surf.gii`

    Returns
    -------
    gifti : pathlib.Path
        Path to generated gifti file
    """

    raise NotImplementedError


def _fsurf_to_gii(surf, out=None):
    """
    Converts FreeSurfer-style surface file to gifti format

    Parameters
    ----------
    surf : str or os.PathLike
        Path to FreeSurfer surface file
    out : str or os.PathLike
        Path to desired output file location. If not specified will use same
        directory as `surf` and replace file extension with `.surf.gii`

    Returns
    -------
    gifti : pathlib.Path
        Path to generated gifti file
    """

    # path handling
    surf = pathify(surf)
    if out is None:
        out = surf.with_suffix('.surf.gii')

    vertices, faces = nib.freesurfer.read_geometry(surf)
    vertices = nib.gifti.GiftiDataArray(vertices, 'NIFTI_INTENT_POINTSET',
                                        datatype='NIFTI_TYPE_FLOAT32')
    faces = nib.gifti.GiftiDataArray(faces, 'NIFTI_INTENT_TRIANGLE',
                                     datatype='NIFTI_TYPE_INT32')
    gii = nib.gifti.GiftiImage(darrays=[vertices, faces])

    nib.save(gii, out)

    return out


def labels_to_gii(labels):
    """
    Checks whether input `labels` is a gifti file and converts it, if needed

    Can currently only Convert FreeSurfer annotation and CIFTI-2 dlabel files

    Parameters
    ----------
    labels : str or os.PathLike
        Input labels file to be converted to gifti format, as necessary

    Returns
    -------
    labels : os.PathLike
        Gifti format of provided `labels`
    remove_labels : bool
        Whether a temporary gifti file was created from `labels`
    """

    labels, remove_labels = pathify(labels), False

    if labels is None:
        return labels, remove_labels

    suffixes = labels.suffixes
    if suffixes[-1] == '.annot':
        labels = _annot_to_gii(labels, out=tempfile.mkstemp('.label.gii')[1])
        remove_labels = True
    elif len(suffixes) >= 2 and ''.join(suffixes[-2:]) == '.dlabel.nii':
        labels = _dlabel_to_gii(labels, out=tempfile.mkstemp('.label.gii')[1])
        remove_labels = True

    return labels, remove_labels


def surf_to_gii(surf):
    """'
    Checks whether input `surf` is a gifti file and converts it, if needed

    Can currently only convert FreeSurfer surface files

    Parameters
    ----------
    surf : str or os.PathLike
        Input surface file to be converted to gifti format, as necessary

    Returns
    -------
    surf : os.PathLike
        Gifti format of provided `surf`
    remove_surf : bool
        Whether a temporary gifti file was created from `surf`
    """

    from .freesurfer import FSEXT

    surf, remove_surf = pathify(surf), False

    if surf is None:
        return surf, remove_surf

    if surf.suffix in FSEXT:
        surf = surf_to_gii(surf, out=tempfile.mkstemp('.surf.gii')[1])
        remove_surf = True

    return surf, remove_surf
