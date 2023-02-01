#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 07:18:55 2023

@author: tommy
"""

import numpy as np
import scipy as sp

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
        logger.info("--------------------------------------------------")
        logger.info("Starting algorithm 'stable_pirls_negative_weights'")

        num_observations, num_coefficients = self.modelmat.shape
        modelmat = self.modelmat
        gam = self.gam
        Y, weights = gam.y_, gam.weights_

        # Penalty matrix of size (num_splines, num_splines)
        # S = sp.sparse.diags(np.ones(num_splines) * np.sqrt(EPS)).A  # improve condition

        P = gam._P().A  # Spline penalties (typically second derivative)
        ID = gam._identifiability_constraints()  # Identifiability constraints (soft sum-to-zero)

        # Since constraints are function of the coefficients, a GAM with
        # constraints can cycle back and forth as constraints 'kick-in'
        # and deactivate. A more gentle update approach helps:
        step_size = 0.5
        if not gam.terms.hasconstraint:
            E = np.vstack((P, ID))
            logger.debug(f"No constraints. Creating penalty of size {E.shape} before PIRLS loop.")

            # Since constraints are function of the coefficients, a GAM with
            # constraints can cycle back and forth as constraints 'kick-in'
            # and deactivate. A more gentle update approach helps:
            step_size = 0.99

        min_n_m = min(num_coefficients, num_observations)

        for iteration in range(1, gam.max_iter + 1):
            logger.info(f"--------------- PIRLS iteration {iteration} ---------------")

            # If the model has constraints (e.g. monotonic increasing),
            # then these constraints are function of the current beta values
            # and must be updated in each inner loop
            if gam.terms.hasconstraint:
                E = np.vstack((P + gam._C().A, ID))

            # forward pass
            y = Y.copy()  # Because we mask it later
            lp = gam._linear_predictor(modelmat=modelmat)
            mu = gam.link.mu(lp, gam.distribution)

            # Create (square root of) the weight vector
            w = gam._W(mu, weights, y=y)
            assert np.all((w >= 0) * np.isfinite(w)), "Weights must be >= 0 and finite"
            # mask = (np.abs(w) >= np.sqrt(EPS)) * np.isfinite(w)
            # logger.debug(f"Strange results on {(~mask).sum()} weights.")

            # Mask variables
            # y, lp, mu, w = y[mask], lp[mask], mu[mask], w[mask]

            pseudo_data = gam._pseudo_data(y, lp, mu) * w
            logger.debug(f"Mean value of pseudo data: {pseudo_data.mean()}")

            # log on-loop-start stats
            gam._on_loop_start(vars())

            # We want to solve the minimization problem
            # ||X \beta - z||^2_{W^2} + ||\beta||^2_{E^T E} =
            # ||W X \beta - W z ||^2 + ||E \beta||^2
            # Differentiating and setting equal to zero yields the normal eqns:
            # (X^T W^T W X + E^T E) \beta = X^T W z
            # The steps below follow page 274 in Wood (2nd edition)

            # Compute Q R = W X
            WB = (modelmat.A.T * w).T
            Q, R = np.linalg.qr(WB)

            if not np.isfinite(Q).all() or not np.isfinite(R).all():
                raise ValueError("QR decomposition produced NaN or Inf. Check X data.")

            # Compute the SVD of R stacked over E
            U, D, Vt = np.linalg.svd(np.vstack([R, E]))
            logger.debug(f"SVD shapes: {U.shape} {D.shape} {Vt.shape}")

            # Only keep the sub-matrices such that R = U_1 D V^T
            U1 = U[:min_n_m, :min_n_m]
            logger.debug(f"Clipping U (shape: {U.shape}) to U1 (shape: {U1.shape})")
            Vt = Vt[:min_n_m, :]
            D_inv = (1 / D)[:min_n_m]

            # At this point WX = Q R = Q (U1 D V^T)
            # and (X^T W^T W X + E^T E) = R^T R + E^T E = (U D V^T)^T (U D V^T) = V D^2 V^T
            # Hence the solution is \beta = (V D^2 V^T)^{-1} (W X)^T z =
            # (V D^{-2} V.T) (Q (U1 D V^T))^T z =
            # V D^{-1} U1^T Q^T z

            # Compute updated coefficients
            inv_vt = D_inv * Vt.T
            coef_new = np.linalg.multi_dot((inv_vt, U1.T, Q.T, pseudo_data))

            # Since constraints are function of the coefficients, a GAM with
            # constraints can cycle back and forth as constraints 'kick-in'
            # and deactivate. A more gentle update approach helps:
            coef_new = step_size * coef_new + (1 - step_size) * gam.coef_

            # Stopping criterion
            relative_change = np.linalg.norm(gam.coef_ - coef_new) / np.linalg.norm(coef_new)

            gam.coef_ = coef_new

            # If deviance decreased, decrease the step size
            if len(gam.logs_["deviance"]) > 2 and gam.logs_["deviance"][-1] > gam.logs_["deviance"][-2]:
                step_size = step_size * 0.99
                logger.info(f"Deviance increased, setting step size: {step_size:.4f}")

            # log on-loop-end stats
            gam._on_loop_end(vars())
            logger.info(f"End of iteration {iteration}. Deviance: {gam.logs_['deviance'][-1]}")

            # check convergence
            if relative_change < gam.tol:
                logger.info(f"PIRLS converged {relative_change} < {gam.tol}")
                break
        else:
            logger.info(f"PIRLS stopped after {iteration} iterations (no convergence)")

        # Update model statistics
        lp = gam._linear_predictor(modelmat=modelmat)
        mu = gam.link.mu(lp, gam.distribution)

        # The effective degrees of freedom is given by the trace of the hat matrix:
        # A general source here is chapter 3.4 Shrinkage Methods in Elements, 2nd ed
        # Below is Equation (6.3) on page 251 in Wood, 2nd ed
        # trace( (X^T W^T W X + E^T E)^{-1} (W X)^T (W X) )
        # trace( (W X) (X^T W^T W X + E^T E)^{-1} (W X)^T ) [by cyclic trace property]
        # trace( Q (U1 D V^T) (V D^{-2} V^T) V D^T U1^T Q^T )
        # trace( Q U1 D D^{-2} D^T U1^T Q^T )
        # trace( Q U1 U1^T Q^T )
        # trace( U1 U1^T ) = trace( U1^T U1 )               [by cyclic trace property]
        edof_per_coef = np.sum(U1**2, axis=1)

        # The covariance is given by ...
        # TODO
        covariance = np.linalg.multi_dot((inv_vt, U1.T, U1, inv_vt.T))

        # The line below is equal to trace( U1 U1^T ) = trace( U1^T U1 )
        gam.statistics_["edof_per_coef"] = edof_per_coef
        gam.statistics_["edof"] = edof_per_coef.sum()

        if not gam.distribution._known_scale:
            gam.distribution.scale = gam.distribution.phi(y=y, mu=mu, edof=gam.statistics_["edof"], weights=weights)

        gam.statistics_["cov"] = covariance * gam.distribution.scale

        # estimate statistics even if not converged
        gam._estimate_model_statistics()


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
        w = alpha / (g_prime**2 * gam.distribution.V(mu=mu)) * weights  # TODO: check
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


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "-k test_convex"])

    np.random.seed(4)
    X = np.random.rand(100_000, 1)
    X = np.sort(X, axis=0)
    y = np.sin(X[:, 0] * 0.99 * np.pi) + np.random.randn(X.shape[0]) / 5 + 100

    from pygam import LinearGAM, s

    gam = LinearGAM(s(0, n_splines=80 * 4, lam=1, constraints="monotonic_inc"), max_iter=100).fit(X, y)

    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], y)
    plt.plot(X[:, 0], gam.predict(X), color="black", lw=5, alpha=0.66)
