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
from copy import deepcopy

try:
    from sksparse.cholmod import cholesky as spcholesky
    from sksparse.test_cholmod import CholmodNotPositiveDefiniteError

    SKSPIMPORT = True
except ImportError:
    CholmodNotPositiveDefiniteError = ValueError
    SKSPIMPORT = False


EPS = np.finfo(np.float64).eps  # machine epsilon

from pygam.log import setup_custom_logger

logger = setup_custom_logger(__name__)


class BetaOptimizer:

    """Optimization routines for beta, given lambda."""

    def __init__(self, gam):
        """Code that is common to all routines."""
        self.gam = gam

        if not all(hasattr(gam, name) for name in ("X_", "y_", "weights_")):
            raise ValueError("Fit model...")

        # Get data
        X, Y = gam.X_, gam.y_

        # Builds a model matrix out of the spline basis for each feature
        self.modelmat = gam._modelmat(X)

        # initialize GLM coefficients if model is not yet fitted
        if not gam._is_fitted or len(gam.coef_) != gam.terms.n_coefs or not np.isfinite(gam.coef_).all():
            gam.coef_ = gam._initial_estimate(Y, self.modelmat)

        if not np.isfinite(gam.coef_).all():
            raise ValueError("Coefficient should be well-behaved.")

    def stable_pirls_negative_weights(self):
        """Stable PIRLS (Penalized Iteratively Re-Weighted Least Squares).

        See sections
          - 6.1.1 Estimating beta given lambda
          - 6.1.3 Stable least squares with negative weights
        in 'Generalized Additive Models' by Simon N. Wood (2nd edition)

        """
        num_observations, num_splines = self.modelmat.shape
        modelmat = self.modelmat
        gam = self.gam
        Y, weights = gam.y_, gam.weights_

        # Penalty matrix of size (num_splines, num_splines)
        P = gam._P()
        S = sp.sparse.diags(np.ones(num_splines) * np.sqrt(EPS))  # improve condition

        # if we dont have any constraints, then do cholesky now
        if not gam.terms.hasconstraint:

            E = P.A
            print(E.shape)

        min_n_m = min(num_splines, num_observations)

        for iteration in range(1, gam.max_iter + 1):
            logger.info(f"PIRLS iteration {iteration}")

            # Recompute cholesky if needed. The constraint matrix C is a
            # function of the current parameter estimates
            if gam.terms.hasconstraint:

                # TODO: Why is sparse flag always false?
                E = _cholesky(
                    S + P.T.dot(P) + gam._C(),
                    constraint_l2=gam._constraint_l2,
                    constraint_l2_max=gam._constraint_l2_max,
                    sparse=False,
                )

            print(P.A.round(2))

            # forward pass
            y = deepcopy(Y)  # Because we mask it later
            lp = gam._linear_predictor(modelmat=modelmat)
            mu = gam.link.mu(lp, gam.distribution)

            # Create weight vector
            # w = (gam.link.gradient(mu, gam.distribution) ** 2 * gam.distribution.V(mu=mu) * weights**-1) ** -0.5
            # w = np.sqrt(weights) / (np.sqrt(gam.distribution.V(mu=mu)) * gam.link.gradient(mu, gam.distribution))
            w = gam._W(mu, weights, y=y)
            mask = (np.abs(w) >= np.sqrt(EPS)) * np.isfinite(w)

            # Mask variables
            y, lp, mu, w = y[mask], lp[mask], mu[mask], w[mask]

            pseudo_data = gam._pseudo_data(y, lp, mu) * w
            logger.debug(f"Mean value of pseudo data: {pseudo_data.mean()}")

            # log on-loop-start stats
            gam._on_loop_start(vars())

            # WB = (modelmat[mask, :].A.T * w).T
            WB = (sp.sparse.diags(w) * modelmat[mask, :]).A
            Q, R = np.linalg.qr(WB)  # 'A' transforms scipy.sparse._csr.csr_matrix -> numpy.ndarray

            if not np.isfinite(Q).all() or not np.isfinite(R).all():
                raise ValueError("QR decomposition produced NaN or Inf. Check X data.")

            # need to recompute the number of singular values
            min_n_m = min(num_splines, num_observations, mask.sum())
            Dinv = np.zeros((num_splines, min_n_m))

            # SVD
            print(f"SV-tacking R with shape {R.shape} over E with shape {E.shape}")
            U, d, Vt = np.linalg.svd(np.vstack([R, E]))
            # svd_mask = d <= (d.max() * np.sqrt(EPS))  # mask out small singular values

            np.fill_diagonal(Dinv, 1 / d)  # invert the singular values
            U1 = U[:min_n_m, :min_n_m]  # keep only top corner of U

            # update coefficients
            # print(Dinv.shape)
            # print(Vt.T.shape)

            B = Vt.T.dot(Dinv).dot(U1.T).dot(Q.T)
            # B2 = (Dinv2 * Vt.T).dot(U1.T).dot(Q.T)

            # assert np.allclose(B, B2)

            coef_new = B.dot(pseudo_data).flatten()
            diff = np.linalg.norm(gam.coef_ - coef_new) / np.linalg.norm(coef_new)
            gam.coef_ = coef_new  # update
            logger.debug(f"Model coefficients: {gam.coef_.round(3)}")

            # log on-loop-end stats
            gam._on_loop_end(vars())
            logger.info(f"End of iteration {iteration}. Deviance: {gam.logs_['deviance'][-1]}")

            # check convergence
            if diff < gam.tol:
                logger.info(f"PIRLS converged {diff} < {gam.tol}")
                break
        else:
            logger.info(f"PIRLS stopped after {iteration} iterations (no convergence)")

        # Update model statistics
        lp = gam._linear_predictor(modelmat=modelmat)
        mu = gam.link.mu(lp, gam.distribution)

        gam.statistics_["edof_per_coef"] = np.sum(U1**2, axis=1)
        gam.statistics_["edof"] = gam.statistics_["edof_per_coef"].sum()

        if not gam.distribution._known_scale:
            gam.distribution.scale = gam.distribution.phi(y=y, mu=mu, edof=gam.statistics_["edof"], weights=weights)

        # With
        # X_1 = modelmat.A
        # W_1 = np.diag(w**2)
        # S_1 = (S + P + gam._C()).A
        # we have
        #             (equation 6.3 in wood)
        # edof = np.diagonal(np.linalg.inv(X_1.T @ W_1 @ X_1 +S_1) @ X_1.T @ W_1 @ X_1)
        # edof = np.diagonal(U1.dot(U1.T))
        # edof = np.sum(U1**2, axis=1)
        gam.statistics_["cov"] = (
            B.dot(B.T)
        ) * gam.distribution.scale  # parameter covariances. no need to remove a W because we are using W^2. Wood pg 184

        # estimate statistics even if not converged
        gam._estimate_model_statistics(B=B, U1=U1)


def pirls_naive(gam):
    """
    Performs naive PIRLS iterations to estimate GAM coefficients

    Parameters
    ---------
    X : array-like of shape (n_samples, m_features)
        containing input data
    y : array-like of shape (n,)
        containing target data

    Returns
    -------
    None
    """
    if not all(hasattr(gam, name) for name in ("X_", "y_", "weights_")):
        raise ValueError("Fit model...")

    # Get data
    X, y, weights = gam.X_, gam.y_, gam.weights_

    modelmat = gam._modelmat(X)  # build a basis matrix for the GLM
    m = modelmat.shape[1]

    # initialize GLM coefficients
    if not gam._is_fitted or len(gam.coef_) != sum(gam._n_coeffs):
        gam.coef_ = np.ones(m) * np.sqrt(EPS)  # allow more training

    P = gam._P()  # create penalty matrix
    P += sp.sparse.diags(np.ones(m) * np.sqrt(EPS))  # improve condition

    # Fisher weights
    alpha = 1

    # Step 1: Initialize mu and eta
    beta = 0
    mu = y + np.sqrt(EPS)
    eta = gam.link.link(mu, dist=gam.distribution)
    assert np.isfinite(eta).all()

    for iteration in range(1, gam.max_iter + 1):
        logger.info(f"PIRLS iteration {iteration}")

        # Step 2: Compute pseudodata z and iterative weights w
        # z = lp + (y - mu) * gam.link.gradient(mu, gam.distribution)
        z = mu + (y - mu) * gam.link.gradient(mu, gam.distribution) / alpha
        g_prime = gam.link.gradient(mu, gam.distribution)
        w = alpha / (g_prime**2 * gam.distribution.V(mu=mu))
        assert np.all(w > 0)

        # Step 3: Find beta, the minimizer of the least squares objective
        # |z - X * beta|^2_W + |beta|^2_P
        modelmat_w = (modelmat.A.T * np.sqrt(w)).T

        P_squared = sp.linalg.cholesky(P.A)
        lhs_stacked = np.vstack((modelmat_w, P_squared))
        rhs_stacked = np.hstack((np.sqrt(w) * z, np.zeros(P.shape[0])))

        beta_new, residuals, rank, singular_values = sp.linalg.lstsq(lhs_stacked, rhs_stacked)

        # Update eta and mu
        eta = modelmat.A @ beta_new
        mu = gam.link.mu(eta, dist=gam.distribution)

        relative_error = np.linalg.norm(beta - beta_new) / np.linalg.norm(beta_new)

        # Convergence
        print("Convergence", relative_error)

        beta = beta_new
        gam.coef_ = beta  # update

        if relative_error < gam.tol:
            return

    raise Exception("Did not converge")


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


if __name__ == "__main__":

    np.random.seed(4)
    X = np.random.randn(100, 2)
    y = np.sin(X[:, 0] * 2.5) + X[:, 1] ** 2 + np.random.randn(100) / 5 + 100

    from pygam import LinearGAM, s, l

    gam = LinearGAM(s(0, n_splines=4, lam=1) + s(1, n_splines=4, lam=1)).fit(X, y)
