# -*- coding: utf-8 -*-
"""
Functions for generating surrogate brain maps as in Burtt et al., 2018, Nature
Neuroscience.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import boxcox


def _make_weight_matrix(dist, d0):
    """
    Constructs weight matrix from distance matrix + autocorrelation estimate

    Parameters
    ----------
    dist : array_like
        Distance matrix
    d0 : float
        Estimate of spatial scale of autocorrelation

    Returns
    -------
    W : numpy.ndarray
        Weight matrix
    """

    # "W is the row-normalized weight matrix with zero diagonal and"
    # "off-diagonal elements proportional to W[ij] = z[i]^-1 exp(-D[ij]/d0),"
    # "where D[ij] is the surface-based geodesic distance between cortical"
    # "areas i and j, and z[i] is a row-wise normalization factor."
    # z[i] = row sum exp(-D[ij]/d0)
    weight = np.exp(-dist / d0) * np.logical_not(np.eye(len(dist), dtype=bool))

    return weight / np.sum(weight, axis=1)


def estimate_rho_d0(dist, neuro, rho=1.0, d0=1.0):
    """
    Uses a least-squares fit to estimate `rho` and `d0`

    Parameters
    ----------
    dist : array_like
        Distance matrix
    neuro : array_like
        Dependent brain-imaging variable; all values must be positive in order
        for successful Box-Cox transformation
    rho : float, optional
        Initial guess for rho parameter. Default: 1.0
    d0 : float, optional
        Initial guess for d0 (spatial scale of autocorrelation) parameter.
        Default: 1.0

    Returns
    -------
    rho_hat : float
        Estimate of `rho` based on least-squares fit between `dist` and `neuro`
    d0_hat : float
        Estimate of `d0` based on least-squares fit between `dist` and `neuro`
    """

    # "two free parameters, rho and d0, are estimated by minimizing the "
    # "residual sum-of-squares"
    def _estimate(parameters, dist, neuro):
        rho, d0 = parameters
        y_hat = rho * (_make_weight_matrix(dist, d0) @ neuro)
        return neuro - y_hat

    # "y is a vector of first Bob-Cox transformed and then mean-subtracted
    # map values"
    neuro, *_ = boxcox(neuro)
    neuro -= neuro.mean()

    return least_squares(_estimate, [rho, d0],
                         args=(dist, neuro), method='lm').x


def make_surrogate(dist, neuro, rho=None, d0=None, seed=None,
                   return_order=False):
    """
    Generates surrogate map of `neuro`, retaining spatial features

    Parameters
    ----------
    dist : array_like
        Distance matrix
    neuro : array_like
        Dependent brain-imaging variable; all values must be positive
    rho : float, optional
        Estimate for rho parameter. If not provided will be estimated from
        input data. Default: None
    d0 : float, optional
        Estimate for d0 parameter. If not provided will be estimated from input
        data. Default: None
    return_order : bool, optional
        Whether to return rank order of generated `surrogate` before values
        were replaced with `neuro`

    Returns
    -------
    surrogate : array_like
        Input `neuro` matrix, permuted according to surrogate map with similar
        spatial autocorrelation factor
    order : array_like
        Rank-order of `surrogate` before values were replaced with `neuro`
    """

    # new random seed
    rs = np.random.default_rng(seed)

    if rho is None or d0 is None:
        rho, d0 = estimate_rho_d0(dist, neuro, rho=rho, d0=d0)

    # "using best-fit parameters rho_hat and d0_hat, surrogate maps y_surr"
    # "are generated according to y_surr = (I - rho_hat * W[d0_hat])^-1 * u"
    # "where u ~ normal(0, 1)"
    w = _make_weight_matrix(dist, d0)
    u = rs.standard_normal(len(dist))
    i = np.identity(len(dist))
    surrogate = np.linalg.inv(i - rho * w) @ u

    # "to match surrogate map value distributions to the distributon of values"
    # "in the corresponding empirical map, rank-ordered surrogate map values"
    # "were re-assigned the corresponding rank-ordered values in the empirical"
    # "data"
    order = surrogate.argsort()
    surrogate[order] = np.sort(neuro)

    if return_order:
        return surrogate, order

    return surrogate
