#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 07:18:55 2023

@author: tommy
"""
import warnings

import numpy as np
import scipy as sp
from numpy.linalg import LinAlgError

try:
    from sksparse.cholmod import cholesky as spcholesky
    from sksparse.test_cholmod import CholmodNotPositiveDefiniteError

    SKSPIMPORT = True
except ImportError:
    CholmodNotPositiveDefiniteError = ValueError
    SKSPIMPORT = False


from pygam.log import setup_custom_logger

logger = setup_custom_logger(__name__)


def _cholesky(A, constraint_l2, constraint_l2_max, sparse=True):
    """
    method to handle potential problems with the cholesky decomposition.

    will try to increase L2 regularization of the penalty matrix to
    do away with non-positive-definite errors

    Parameters
    ----------
    A : np.array

    Returns
    -------
    np.array
    """
    # create appropriate-size diagonal matrix
    if sp.sparse.issparse(A):
        diag = sp.sparse.eye(A.shape[0])
    else:
        diag = np.eye(A.shape[0])

    loading = constraint_l2
    while loading <= constraint_l2_max:
        try:

            L = cholesky(A, sparse=sparse)
            return L
        except (CholmodNotPositiveDefiniteError, LinAlgError):
            warnings.warn("Matrix is not positive definite. Increasing l2 reg by factor of 10.", stacklevel=2)
            A -= loading * diag
            loading *= 10
            A += loading * diag

    raise LinAlgError("Matrix is not positive definite.")


def cholesky(A, sparse=True):
    """
    Choose the best possible cholesky factorizor.

    if possible, import the Scikit-Sparse sparse Cholesky method.
    Permutes the output L to ensure A = L.H . L

    otherwise defaults to numpy's non-sparse version

    Parameters
    ----------
    A : array-like
        array to decompose
    sparse : boolean, default: True
        whether to return a sparse array
    """
    if SKSPIMPORT:
        logger.info("Cholesky decomposition with scikit-sparse")
        A = sp.sparse.csc_matrix(A)
        try:
            F = spcholesky(A)

            # permutation matrix P
            P = sp.sparse.lil_matrix(A.shape)
            p = F.P()
            P[np.arange(len(p)), p] = 1

            # permute
            L = F.L()
            L = P.T.dot(L)
        except CholmodNotPositiveDefiniteError as error:
            raise error

        if sparse:
            return L.T  # upper triangular factorization
        return L.T.A  # upper triangular factorization

    else:
        msg = (
            "Could not import Scikit-Sparse or Suite-Sparse.\n"
            "This will slow down optimization for models with "
            "monotonicity/convexity penalties and many splines.\n"
            "See installation instructions for installing "
            "Scikit-Sparse and Suite-Sparse via Conda."
        )
        warnings.warn(msg)

        if sp.sparse.issparse(A):
            A = A.A

        try:
            L = sp.linalg.cholesky(A, lower=False)
        except LinAlgError as error:
            raise error

        if sparse:
            return sp.sparse.csc_matrix(L)
        return L
