# -*- coding: utf-8 -*-

import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.utils import check_array, check_X_y

from pygam.callbacks import CALLBACKS, CallBack, validate_callback
from pygam.core import Core
from pygam.distributions import (
    DISTRIBUTIONS,
    Distribution,
    GammaDist,
    InvGaussDist,
    NormalDist,
)
from pygam.links import LINKS, Link
from pygam.optimization import BetaOptimizer
from pygam.terms import (
    Intercept,
    MetaTermMixin,
    SplineTerm,
    Term,
    TermList,
)
from pygam.utils import (
    TablePrinter,
    check_lengths,
    check_param,
    check_X,
    check_y,
    flatten,
    isiterable,
    load_diagonal,
    make_2d,
    sig_code,
    space_row,
)


EPS = np.finfo(np.float64).eps  # machine epsilon


# TODO: Constraints
# https://arxiv.org/pdf/1812.07696.pdf

from pygam.log import setup_custom_logger

logger = setup_custom_logger(__name__)


class GAM(Core, MetaTermMixin):
    """Generalized Additive Model

    Parameters
    ----------
    terms : expression specifying terms to model, optional.

        By default a univariate spline term will be allocated for each feature.
        Will fit a spline term on feature 0, a linear term on feature 1,
        a factor term on feature 2, and a tensor term on features 3 and 4.

    callbacks : list of str or list of CallBack objects, optional
        Names of callback objects to call during the optimization loop.

    distribution : str or Distribution object, optional
        Distribution to use in the model.

    link : str or Link object, optional
        Link function to use in the model.

    fit_intercept : bool, optional
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
        Note: the intercept receives no smoothing penalty.

    max_iter : int, optional
        Maximum number of iterations allowed for the solver to converge.

    tol : float, optional
        Tolerance for stopping criteria.

    verbose : bool, optional
        whether to show pyGAM warnings.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as ``{callback: [...]}``

    Examples
    --------
    >>> from pygam import GAM, s, l, f, te
    >>> from pygam.datasets import wage
    >>> X, y = wage(return_X_y=True)
    >>> # The features are: 'year', 'age', 'education'
    >>> gam = GAM(l(0) + s(1) + f(2), link="identity", distribution="normal")
    >>> gam = gam.fit(X, y)
    >>> gam.predict(X)
    array([ 52.68627245,  99.57814217, 113.04814967, ...,  71.25893898,
            94.71905301, 105.0052182 ])
    >>> gam.score(X, y)
    0.294...

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """

    def __init__(
        self,
        terms="auto",
        max_iter=100,
        tol=1e-4,
        distribution="normal",
        link="identity",
        callbacks=["deviance", "diffs"],
        fit_intercept=True,
        verbose=False,
        **kwargs,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.distribution = distribution
        self.link = link
        self.callbacks = callbacks
        self.verbose = verbose
        self.terms = TermList(terms) if isinstance(terms, Term) else terms
        self.fit_intercept = fit_intercept

        for k, v in kwargs.items():
            if k not in self._plural:
                raise TypeError("__init__() got an unexpected keyword argument {}".format(k))
            setattr(self, k, v)

        # internal settings
        self._constraint_lam = 1e8  # regularization intensity for constraints
        # self._constraint_l2 = 1e-3  # diagononal loading to improve conditioning
        # self._constraint_l2_max = 1e-1  # maximum loading
        # self._opt = 0 # use 0 for numerically stable optimizer, 1 for naive
        self._term_location = "terms"  # for locating sub terms
        # self._include = ['lam']

        # call super and exclude any variables
        super().__init__()

        logger.info(f"Created GAM instance:\n{self}")

    def yield_terms_and_betas(self):
        assert self._is_fitted
        start_idx = 0
        for term in self.terms:
            yield term, self.coef_[start_idx : start_idx + term.n_coefs]
            start_idx += term.n_coefs

    @property
    def _is_fitted(self):
        """Check whether the GAM is fitted."""
        return hasattr(self, "coef_")

    def _validate_params(self):
        """Validate input parameters."""

        # fit_intercep
        if not isinstance(self.fit_intercept, bool):
            raise ValueError("fit_intercept must be type bool, but found {}".format(self.fit_intercept.__class__))

        # terms
        if (self.terms != "auto") and not (isinstance(self.terms, (TermList, Term, type(None)))):
            raise ValueError("terms must be a TermList, but found " "terms = {}".format(self.terms))

        # max_iter
        self.max_iter = check_param(self.max_iter, param_name="max_iter", dtype="int", constraint=">=1", iterable=False)

        # distribution
        if not ((self.distribution in DISTRIBUTIONS) or isinstance(self.distribution, Distribution)):
            raise ValueError("unsupported distribution {}".format(self.distribution))
        if self.distribution in DISTRIBUTIONS:
            self.distribution = DISTRIBUTIONS[self.distribution]()

        # link
        if not ((self.link in LINKS) or isinstance(self.link, Link)):
            raise ValueError("unsupported link {}".format(self.link))
        if self.link in LINKS:
            self.link = LINKS[self.link]()

        # callbacks
        if not isiterable(self.callbacks):
            raise ValueError("Callbacks must be iterable, but found {}".format(self.callbacks))

        if not all(c in CALLBACKS or isinstance(c, CallBack) for c in self.callbacks):
            raise ValueError("unsupported callback(s) {}".format(self.callbacks))
        callbacks = list(self.callbacks)
        for i, c in enumerate(self.callbacks):
            if c in CALLBACKS:
                callbacks[i] = CALLBACKS[c]()
        self.callbacks = [validate_callback(c) for c in callbacks]

    def _validate_data_dep_params(self, X):
        """Validate and prepare data dependent parameters."""

        n_samples, m_features = X.shape

        # terms
        if self.terms == "auto":
            # one numerical spline per feature
            self.terms = TermList(*[SplineTerm(feat, verbose=self.verbose) for feat in range(m_features)])

        elif self.terms is None:
            # no terms
            self.terms = TermList()

        else:
            # user-specified
            self.terms = TermList(self.terms, verbose=self.verbose)

        # add intercept
        if self.fit_intercept:
            self.terms = self.terms + Intercept()

        if len(self.terms) == 0:
            raise ValueError("At least 1 term must be specified")

        # copy over things from plural
        remove = []
        for k, v in self.__dict__.items():
            if k in self._plural:
                setattr(self.terms, k, v)
                remove.append(k)
        for k in remove:
            delattr(self, k)

        self.terms.compile(X)

    def loglikelihood(self, X, y, weights=None):
        """
        compute the log-likelihood of the dataset using the current model

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing the input dataset
        y : array-like of shape (n,)
            containing target values
        weights : array-like of shape (n,), optional
            containing sample weights

        Returns
        -------
        log-likelihood : np.array of shape (n,)
            containing log-likelihood scores
        """
        y = check_y(y, self.link, self.distribution)
        mu = self.predict_mu(X)

        if weights is not None:
            weights = np.array(weights).astype("f").ravel()
            weights = check_array(weights, input_name="sample weights", ensure_2d=False)
            assert len(y) == len(weights)
        else:
            weights = np.ones_like(y).astype("float64")

        return self._loglikelihood(y, mu, weights=weights)

    def _loglikelihood(self, y, mu, weights=None):
        """
        compute the log-likelihood of the dataset using the current model

        Parameters
        ---------
        y : array-like of shape (n,)
            containing target values
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs
        weights : array-like of shape (n,), optional
            containing sample weights

        Returns
        -------
        log-likelihood : np.array of shape (n,)
            containing log-likelihood scores
        """
        return self.distribution.log_pdf(y=y, mu=mu, weights=weights).sum()

    def _linear_predictor(self, X=None, modelmat=None, b=None, term=-1):
        """linear predictor
        compute the linear predictor portion of the model
        ie multiply the model matrix by the spline basis coefficients

        Parameters
        ---------
        at least 1 of (X, modelmat)
            and
        at least 1 of (b, feature)

        X : array-like of shape (n_samples, m_features) or None, optional
            containing the input dataset
            if None, will attempt to use modelmat

        modelmat : array-like or None, optional
            contains the spline basis for each feature evaluated at the input
            values for each feature, ie model matrix
            if None, will attempt to construct the model matrix from X

        b : array-like or None, optional
            contains the spline coefficients
            if None, will use current model coefficients

        feature : int, optional
                  feature for which to compute the linear prediction
                  if -1, will compute for all features

        Returns
        -------
        lp : np.array of shape (n_samples,)
        """
        if modelmat is None:
            modelmat = self._modelmat(X, term=term)
        if b is None:
            b = self.coef_[self.terms.get_coef_indices(term)]
        return modelmat.dot(b).flatten()

    def predict_mu(self, X):
        """
        preduct expected value of target given model and input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features),
            containing the input dataset

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing expected values under the model
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        X = check_X(
            X,
            n_feats=self.statistics_["m_features"],
            edge_knots=self.edge_knots_,
            dtypes=self.dtype,
            features=self.feature,
        )

        lp = self._linear_predictor(X)
        return self.link.mu(lp, self.distribution)

    def predict(self, X):
        """
        preduct expected value of target given model and input X
        often this is done via expected value of GAM given input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing the input dataset

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing predicted values under the model
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")
        X = check_array(X, force_all_finite=True, input_name="X", ensure_2d=True, estimator=self)
        return self.predict_mu(X)

    def _modelmat(self, X, term=-1):
        """
        Builds a model matrix, B, out of the spline basis for each feature

        B = [B_0, B_1, ..., B_p]

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing the input dataset
        term : int, optional
            term index for which to compute the model matrix
            if -1, will create the model matrix for all features

        Returns
        -------
        modelmat : sparse matrix of len n_samples
            containing model matrix of the spline basis for selected features
        """
        X = check_X(
            X,
            n_feats=self.statistics_["m_features"],
            edge_knots=self.edge_knots_,
            dtypes=self.dtype,
            features=self.feature,
        )

        return self.terms.build_columns(X, term=term)

    def _identifiability_constraints(self):
        """Create a matrix C such that |C beta| imposes identifiability constraints."""
        C = np.zeros(shape=(len(self.terms), sum([t.n_coefs for t in self.terms])), dtype=float)
        start_idx = 0
        for i, term in enumerate(self.terms):
            if not term.isintercept:
                C[i, start_idx : start_idx + term.n_coefs] = 1.0 / np.sqrt(len(term))
            start_idx += term.n_coefs

        return C

    def _P(self):
        return self.terms.build_penalties()

    def _C(self):
        return self.terms.build_constraints(self.coef_, self._constraint_lam)

    def _pseudo_data(self, y, lp, mu):
        """
        compute the pseudo data for a PIRLS iterations

        Parameters
        ---------
        y : array-like of shape (n,)
            containing target data
        lp : array-like of shape (n,)
            containing linear predictions by the model
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs

        Returns
        -------
        pseudo_data : np.array of shape (n,)
        """
        return lp + (y - mu) * self.link.gradient(mu, self.distribution)

    def _W(self, mu, weights, y=None):
        """
        compute the PIRLS weights for model predictions.

        TODO lets verify the formula for this.
        if we use the square root of the mu with the stable opt,
        we get the same results as when we use non-sqrt mu with naive opt.

        this makes me think that they are equivalent.

        also, using non-sqrt mu with stable opt gives very small edofs for even lam=0.001
        and the parameter variance is huge. this seems strange to me.

        computed [V * d(link)/d(mu)] ^(-1/2) by hand and the math checks out as hoped.

        ive since moved the square to the naive pirls method to make the code modular.

        Parameters
        ---------
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs
        weights : array-like of shape (n_samples,)
            containing sample weights
        y = array-like of shape (n_samples,) or None, optional
            does nothing. just for compatibility with ExpectileGAM

        Returns
        -------
        weights : sp..sparse array of shape (n_samples, n_samples)
        """
        # Section 6.1.1, list entry (2) in Wood
        return (self.link.gradient(mu, self.distribution) ** 2 * self.distribution.V(mu=mu) * weights**-1) ** -0.5

    def _initial_estimate(self, y, modelmat):
        """
        Makes an inital estimate for the model coefficients.

        For a LinearGAM we simply initialize to small coefficients.

        For other GAMs we transform the problem to the linear space
        and solve an unpenalized version.

        Parameters
        ---------
        y : array-like of shape (n,)
            containing target data
        modelmat : sparse matrix of shape (n, m)
            containing model matrix of the spline basis

        Returns
        -------
        coef : array of shape (m,) containing the initial estimate for the model
            coefficients

        Notes
        -----
            This method implements the suggestions in
            Wood (2nd ed), section 3.2.2 Geometry and IRLS convergence, pg 124
        """
        logger.info("Calling `_initial_estimate`")

        # do a simple initialization for LinearGAMs
        if isinstance(self, LinearGAM):
            n, m = modelmat.shape
            return np.ones(m) * np.sqrt(EPS)

        # transform the problem to the linear scale
        y = deepcopy(y).astype("float64")
        y[y == 0] += 0.01  # edge case for log link, inverse link, and logit link
        y[y == 1] -= 0.01  # edge case for logit link

        y_ = self.link.link(y, self.distribution)
        y_ = make_2d(y_)
        assert np.isfinite(y_).all(), "transformed response values should be well-behaved."

        # Solve ||X \beta - y||^2_W, where W is a diagonal loading matrix
        # This is the Ridge problem, and sklearn solves it like this:
        # https://github.com/scikit-learn/scikit-learn/blob/4db04923a754b6a2defa1b172f55d492b85d165e/sklearn/linear_model/_ridge.py#L204
        lhs = load_diagonal(modelmat.T.dot(modelmat).A, load=1e-4)
        rhs = modelmat.T.dot(y_)
        ans = sp.linalg.solve(lhs, rhs, assume_a="sym")
        return ans.ravel()

    def _on_loop_start(self, variables):
        """
        performs on-loop-start actions like callbacks

        variables contains local namespace variables.

        Parameters
        ---------
        variables : dict of available variables

        Returns
        -------
        None
        """
        for callback in self.callbacks:
            if hasattr(callback, "on_loop_start"):
                self.logs_[str(callback)].append(callback.on_loop_start(**variables))

    def _on_loop_end(self, variables):
        """
        performs on-loop-end actions like callbacks

        variables contains local namespace variables.

        Parameters
        ---------
        variables : dict of available variables

        Returns
        -------
        None
        """
        for callback in self.callbacks:
            if hasattr(callback, "on_loop_end"):
                self.logs_[str(callback)].append(callback.on_loop_end(**variables))

    def fit(self, X, y, weights=None):
        """Fit the generalized additive model.

        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training vectors.
        y : array-like, shape (n_samples,)
            Target values,
            ie integers in classification, real numbers in
            regression)
        weights : array-like shape (n_samples,) or None, optional
            Sample weights.
            if None, defaults to array of ones

        Returns
        -------
        self : object
            Returns fitted GAM object
        """
        logger.info(f"Fitting GAM\n{self}")

        # validate parameters
        self._validate_params()

        # validate data
        y = check_y(y, self.link, self.distribution)
        X = check_X(X)
        check_X_y(X, y)

        if weights is not None:
            weights = np.array(weights).astype("f").ravel()
            weights = check_array(weights, input_name="sample weights", ensure_2d=False)
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype("float64")

        # validate data-dependent parameters
        self._validate_data_dep_params(X)

        self.X_ = X
        self.y_ = y
        self.weights_ = weights

        # set up logging
        if not hasattr(self, "logs_"):
            self.logs_ = defaultdict(list)

        # begin capturing statistics
        self.statistics_ = {"n_samples": len(y), "m_features": X.shape[1]}

        # optimize
        BetaOptimizer(self).stable_pirls_negative_weights()

        return self

    def score(self, X, y, weights=None):
        """
        method to compute the explained deviance for a trained model for a given X data and y labels

        Parameters
        ----------
        X : array-like
            Input data array of shape (n_samples, m_features)
        y : array-like
            Output data vector of shape (n_samples,)
        weights : array-like shape (n_samples,) or None, optional
            Sample weights.
            if None, defaults to array of ones

        Returns
        -------
        explained deviancce score: np.array() (n_samples,)

        """
        r2 = self._estimate_r2(X=X, y=y, mu=None, weights=weights)

        return r2["explained_deviance"]

    def deviance_residuals(self, X, y, weights=None, scaled=False):
        """
        method to compute the deviance residuals of the model

        these are analogous to the residuals of an OLS.

        Parameters
        ----------
        X : array-like
            Input data array of shape (n_samples, m_features)
        y : array-like
            Output data vector of shape (n_samples,)
        weights : array-like shape (n_samples,) or None, optional
            Sample weights.
            if None, defaults to array of ones
        scaled : bool, optional
            whether to scale the deviance by the (estimated) distribution scale

        Returns
        -------
        deviance_residuals : np.array
            with shape (n_samples,)
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        y = check_y(y, self.link, self.distribution)
        X = check_X(
            X,
            n_feats=self.statistics_["m_features"],
            edge_knots=self.edge_knots_,
            dtypes=self.dtype,
            features=self.feature,
        )
        check_X_y(X, y)

        if weights is not None:
            weights = np.array(weights).astype("f").ravel()
            weights = check_array(weights, input_name="sample weights", ensure_2d=False)
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype("float64")

        mu = self.predict_mu(X)
        sign = np.sign(y - mu)
        return sign * self.distribution.deviance(y, mu, weights=weights, scaled=scaled) ** 0.5

    def _estimate_model_statistics(self):
        """
        method to compute all of the model statistics

        results are stored in the 'statistics_' attribute of the model, as a
        dictionary keyed by:

        - edof: estimated degrees freedom
        - scale: distribution scale, if applicable
        - cov: coefficient covariances
        - se: standarrd errors
        - AIC: Akaike Information Criterion
        - AICc: corrected Akaike Information Criterion
        - pseudo_r2: dict of Pseudo R-squared metrics
        - GCV: generailized cross-validation
            or
        - UBRE: Un-Biased Risk Estimator
        - n_samples: number of samples used in estimation

        Parameters
        ----------
        y : array-like
          output data vector of shape (n_samples,)
        modelmat : array-like, default: None
            contains the spline basis for each feature evaluated at the input
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
        U1 : cropped U matrix from SVD.

        Returns
        -------
        None
        """
        modelmat = self._modelmat(self.X_)
        y = self.y_
        weights = self.weights_

        lp = self._linear_predictor(modelmat=modelmat)
        mu = self.link.mu(lp, self.distribution)

        self.statistics_["scale"] = self.distribution.scale
        self.statistics_["se"] = self.statistics_["cov"].diagonal() ** 0.5
        self.statistics_["AIC"] = self._estimate_AIC(y=y, mu=mu, weights=weights)
        self.statistics_["AICc"] = self._estimate_AICc(y=y, mu=mu, weights=weights)
        self.statistics_["pseudo_r2"] = self._estimate_r2(y=y, mu=mu, weights=weights)
        self.statistics_["GCV"], self.statistics_["UBRE"] = self._estimate_GCV_UBRE(
            modelmat=modelmat, y=y, weights=weights
        )
        self.statistics_["loglikelihood"] = self._loglikelihood(y, mu, weights=weights)
        self.statistics_["deviance"] = self.distribution.deviance(y=y, mu=mu, weights=weights).sum()
        self.statistics_["p_values"] = self._estimate_p_values()

    def _estimate_AIC(self, y, mu, weights=None):
        """
        estimate the Akaike Information Criterion

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            output data vector
        mu : array-like of shape (n_samples,),
            expected value of the targets given the model and inputs
        weights : array-like shape (n_samples,) or None, optional
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        None
        """
        estimated_scale = not (self.distribution._known_scale)  # if we estimate the scale, that adds 2 dof
        return (
            -2 * self._loglikelihood(y=y, mu=mu, weights=weights) + 2 * self.statistics_["edof"] + 2 * estimated_scale
        )

    def _estimate_AICc(self, y, mu, weights=None):
        """
        estimate the corrected Akaike Information Criterion

        relies on the estimated degrees of freedom, which must be computed
        before.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            output data vector
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs
        weights : array-like shape (n_samples,) or None, optional
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        None
        """
        edof = self.statistics_["edof"]
        if self.statistics_["AIC"] is None:
            self.statistics_["AIC"] = self._estimate_AIC(y, mu, weights)
        return self.statistics_["AIC"] + 2 * (edof + 1) * (edof + 2) / (y.shape[0] - edof - 2)

    def _estimate_r2(self, X=None, y=None, mu=None, weights=None):
        """
        estimate some pseudo R^2 values

        currently only computes explained deviance.
        results are stored

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            output data vector
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs
        weights : array-like shape (n_samples,) or None, optional
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        None
        """
        if mu is None:
            mu = self.predict_mu(X=X)

        if weights is None:
            weights = np.ones_like(y).astype("float64")

        null_mu = y.mean() * np.ones_like(y).astype("float64")

        null_d = self.distribution.deviance(y=y, mu=null_mu, weights=weights)
        full_d = self.distribution.deviance(y=y, mu=mu, weights=weights)

        null_ll = self._loglikelihood(y=y, mu=null_mu, weights=weights)
        full_ll = self._loglikelihood(y=y, mu=mu, weights=weights)

        r2 = OrderedDict()
        r2["explained_deviance"] = 1.0 - full_d.sum() / null_d.sum()
        r2["McFadden"] = full_ll / null_ll
        r2["McFadden_adj"] = 1.0 - (full_ll - self.statistics_["edof"]) / null_ll

        return r2

    def _estimate_GCV_UBRE(self, X=None, y=None, modelmat=None, gamma=1.4, add_scale=True, weights=None):
        """
        Generalized Cross Validation and Un-Biased Risk Estimator.

        UBRE is used when the scale parameter is known,
        like Poisson and Binomial families.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            output data vector
        modelmat : array-like, default: None
            contains the spline basis for each feature evaluated at the input
        gamma : float, default: 1.4
            serves as a weighting to increase the impact of the influence matrix
            on the score
        add_scale : boolean, default: True
            UBRE score can be negative because the distribution scale
            is subtracted. to keep things positive we can add the scale back.
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        score : float
            Either GCV or UBRE, depending on if the scale parameter is known.

        Notes
        -----
        Sometimes the GCV or UBRE selected model is deemed to be too wiggly,
        and a smoother model is desired. One way to achieve this, in a
        systematic way, is to increase the amount that each model effective
        degree of freedom counts, in the GCV or UBRE score, by a factor ?? ??? 1

        see Wood 2006 pg. 177-182, 220 for more details.
        """
        if gamma < 1:
            raise ValueError("gamma scaling should be greater than 1, " "but found gamma = {}", format(gamma))

        if modelmat is None:
            modelmat = self._modelmat(X)

        if weights is None:
            weights = np.ones_like(y).astype("float64")

        lp = self._linear_predictor(modelmat=modelmat)
        mu = self.link.mu(lp, self.distribution)
        n = y.shape[0]
        edof = self.statistics_["edof"]

        GCV = None
        UBRE = None

        dev = self.distribution.deviance(mu=mu, y=y, scaled=False, weights=weights).sum()

        if self.distribution._known_scale:
            # scale is known, use UBRE
            scale = self.distribution.scale
            UBRE = 1.0 / n * dev - (~add_scale) * (scale) + 2.0 * gamma / n * edof * scale
        else:
            # scale unkown, use GCV
            GCV = (n * dev) / (n - gamma * edof) ** 2
        return (GCV, UBRE)

    def _estimate_p_values(self):
        """estimate the p-values for all features"""
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        p_values = []
        for term_i in range(len(self.terms)):
            p_values.append(self._compute_p_value(term_i))

        return p_values

    def _compute_p_value(self, term_i):
        """compute the p-value of the desired feature

        Arguments
        ---------
        term_i : int
            term to select from the data

        Returns
        -------
        p_value : float

        Notes
        -----
        Wood 2006, section 4.8.5:
            The p-values, calculated in this manner, behave correctly for un-penalized models,
            or models with known smoothing parameters, but when smoothing parameters have
            been estimated, the p-values are typically lower than they should be, meaning that
            the tests reject the null too readily.

                (...)

            In practical terms, if these p-values suggest that a term is not needed in a model,
            then this is probably true, but if a term is deemed ???significant??? it is important to be
            aware that this significance may be overstated.

        based on equations from Wood 2006 section 4.8.5 page 191
        and errata https://people.maths.bris.ac.uk/~sw15190/igam/iGAMerrata-12.pdf

        the errata shows a correction for the f-statisitc.
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        idxs = self.terms.get_coef_indices(term_i)
        cov = self.statistics_["cov"][idxs][:, idxs]
        coef = self.coef_[idxs]

        # center non-intercept term functions
        if isinstance(self.terms[term_i], SplineTerm):
            coef -= coef.mean()

        inv_cov, rank = sp.linalg.pinv(cov, return_rank=True)
        score = coef.T.dot(inv_cov).dot(coef)

        # compute p-values
        if self.distribution._known_scale:
            # for known scale use chi-squared statistic
            return 1 - sp.stats.chi2.cdf(x=score, df=rank)
        else:
            # if scale has been estimated, prefer to use f-statisitc
            score = score / rank
            return 1 - sp.stats.f.cdf(score, rank, self.statistics_["n_samples"] - self.statistics_["edof"])

    def confidence_intervals(self, X, width=0.95, quantiles=None):
        """estimate confidence intervals for the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, m_features)
            Input data matrix
        width : float on [0,1], optional
        quantiles : array-like of floats in (0, 1), optional
            Instead of specifying the prediciton width, one can specify the
            quantiles. So ``width=.95`` is equivalent to ``quantiles=[.025, .975]``

        Returns
        -------
        intervals: np.array of shape (n_samples, 2 or len(quantiles))


        Notes
        -----
        Wood 2006, section 4.9
            Confidence intervals based on section 4.8 rely on large sample results to deal with
            non-Gaussian distributions, and treat the smoothing parameters as fixed, when in
            reality they are estimated from the data.
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        X = check_X(
            X,
            n_feats=self.statistics_["m_features"],
            edge_knots=self.edge_knots_,
            dtypes=self.dtype,
            features=self.feature,
        )

        return self._get_quantiles(X, width, quantiles, prediction=False)

    def _get_quantiles(self, X, width, quantiles, modelmat=None, lp=None, prediction=False, xform=True, term=-1):
        """
        estimate prediction intervals for LinearGAM

        Parameters
        ----------
        X : array
            input data of shape (n_samples, m_features)
        width : float on (0, 1)
        quantiles : array-like of floats on (0, 1)
            instead of specifying the prediciton width, one can specify the
            quantiles. so width=.95 is equivalent to quantiles=[.025, .975]
        modelmat : array of shape or None, default: None
        lp : array or None, default: None
        prediction : bool, default: True.
            whether to compute prediction intervals (True)
            or confidence intervals (False)
        xform : bool, default: True,
            whether to apply the inverse link function and return values
            on the scale of the distribution mean (True),
            or to keep on the linear predictor scale (False)
        term : int, default: -1

        Returns
        -------
        intervals: np.array of shape (n_samples, 2 or len(quantiles))

        Notes
        -----
        when the scale parameter is known, then we can proceed with a large
        sample approximation to the distribution of the model coefficients
        where B_hat ~ Normal(B, cov)

        when the scale parameter is unknown, then we have to account for
        the distribution of the estimated scale parameter, which is Chi-squared.
        since we scale our estimate of B_hat by the sqrt of estimated scale,
        we get a t distribution: Normal / sqrt(Chi-squared) ~ t

        see Simon Wood section 1.3.2, 1.3.3, 1.5.5, 2.1.5
        """
        if quantiles is not None:
            quantiles = np.atleast_1d(quantiles)
        else:
            alpha = (1 - width) / 2.0
            quantiles = [alpha, 1 - alpha]
        for quantile in quantiles:
            if (quantile >= 1) or (quantile <= 0):
                raise ValueError("quantiles must be in (0, 1), but found {}".format(quantiles))

        if modelmat is None:
            modelmat = self._modelmat(X, term=term)
        if lp is None:
            lp = self._linear_predictor(modelmat=modelmat, term=term)

        idxs = self.terms.get_coef_indices(term)
        cov = self.statistics_["cov"][idxs][:, idxs]

        var = (modelmat.dot(cov) * modelmat.A).sum(axis=1)
        if prediction:
            var += self.distribution.scale

        lines = []
        for quantile in quantiles:
            if self.distribution._known_scale:
                q = sp.stats.norm.ppf(quantile)
            else:
                q = sp.stats.t.ppf(quantile, df=self.statistics_["n_samples"] - self.statistics_["edof"])

            lines.append(lp + q * var**0.5)
        lines = np.vstack(lines).T

        if xform:
            lines = self.link.mu(lines, self.distribution)
        return lines

    def _flatten_mesh(self, Xs, term):
        """flatten the mesh and distribute into a feature matrix"""
        n = Xs[0].size

        if self.terms[term].istensor:
            terms = self.terms[term]
        else:
            terms = [self.terms[term]]

        X = np.zeros((n, self.statistics_["m_features"]))
        for term_, x in zip(terms, Xs):
            X[:, term_.feature] = x.ravel()
        return X

    def generate_X_grid(self, term, n=100, meshgrid=False):
        """create a nice grid of X data

        array is sorted by feature and uniformly spaced,
        so the marginal and joint distributions are likely wrong

        if term is >= 0, we generate n samples per feature,
        which results in n^deg samples,
        where deg is the degree of the interaction of the term

        Parameters
        ----------
        term : int,
            Which term to process.

        n : int, optional
            number of data points to create

        meshgrid : bool, optional
            Whether to return a meshgrid (useful for 3d plotting)
            or a feature matrix (useful for inference like partial predictions)

        Returns
        -------
        if meshgrid is False:
            np.array of shape (m, n_features)
            where m is the number of
            (sub)terms in the requested (tensor)term.
        else:
            tuple of len m,
            where m is the number of (sub)terms in the requested
            (tensor)term.

            each element in the tuple contains a np.ndarray of size (n)^m

        Raises
        ------
        ValueError :
            If the term requested is an intercept
            since it does not make sense to process the intercept term.

        Examples
        --------
        >>> import numpy as np
        >>> from pygam import LinearGAM, s, te
        >>> x = [np.linspace(0, i, num=2) for i in range(1, 4)]
        >>> X = np.vstack(x).T
        >>> y = X @ np.array([1, 2, 3])
        >>> gam = LinearGAM(s(0) + te(1, 2)).fit(X, y)
        >>> gam.generate_X_grid(0, n=10, meshgrid=False)
        array([[0.        , 0.        , 0.        ],
               [0.11111111, 0.        , 0.        ],
               [0.22222222, 0.        , 0.        ],
               [0.33333333, 0.        , 0.        ],
               [0.44444444, 0.        , 0.        ],
               [0.55555556, 0.        , 0.        ],
               [0.66666667, 0.        , 0.        ],
               [0.77777778, 0.        , 0.        ],
               [0.88888889, 0.        , 0.        ],
               [1.        , 0.        , 0.        ]])
        >>> gam.generate_X_grid(0, n=10, meshgrid=True)
        (array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
               0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ]),)
        >>> gam.generate_X_grid(1, n=3, meshgrid=False)
        array([[0. , 0. , 0. ],
               [0. , 0. , 1.5],
               [0. , 0. , 3. ],
               [0. , 1. , 0. ],
               [0. , 1. , 1.5],
               [0. , 1. , 3. ],
               [0. , 2. , 0. ],
               [0. , 2. , 1.5],
               [0. , 2. , 3. ]])
        >>> gam.generate_X_grid(1, n=3, meshgrid=True)
        (array([[0., 0., 0.],
               [1., 1., 1.],
               [2., 2., 2.]]), array([[0. , 1.5, 3. ],
               [0. , 1.5, 3. ],
               [0. , 1.5, 3. ]]))

        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        # cant do Intercept
        if self.terms[term].isintercept:
            raise ValueError("cannot create grid for intercept term")

        # process each subterm in a TensorTerm
        if self.terms[term].istensor:
            Xs = [np.linspace(*term_.edge_knots_, num=n) for term_ in self.terms[term]]
            Xs = np.meshgrid(*Xs, indexing="ij")
            if meshgrid:
                return tuple(Xs)
            else:
                return self._flatten_mesh(Xs, term=term)

        # all other Terms
        elif hasattr(self.terms[term], "edge_knots_"):
            x = np.linspace(*self.terms[term].edge_knots_, num=n)

            if meshgrid:
                return (x,)

            # fill in feature matrix with only relevant features for this term
            X = np.zeros((n, self.statistics_["m_features"]))
            X[:, self.terms[term].feature] = x
            if getattr(self.terms[term], "by", None) is not None:
                X[:, self.terms[term].by] = 1.0

            return X

        # dont know what to do here
        else:
            raise TypeError(f"Unexpected term type: {self.terms[term]}")

    def partial_dependence(self, term, X=None, width=None, quantiles=None, meshgrid=False):
        """
        Computes the term functions for the GAM
        and possibly their confidence intervals.

        if both width=None and quantiles=None,
        then no confidence intervals are computed

        Parameters
        ----------
        term : int, optional
            Term for which to compute the partial dependence functions.

        X : array-like with input data, optional

            if `meshgrid=False`, then `X` should be an array-like
            of shape (n_samples, m_features).

            if `meshgrid=True`, then `X` should be a tuple containing
            an array for each feature in the term.

            if None, an equally spaced grid of points is generated.

        width : float on (0, 1), optional
            Width of the confidence interval.

        quantiles : array-like of floats on (0, 1), optional
            instead of specifying the prediciton width, one can specify the
            quantiles. so width=.95 is equivalent to quantiles=[.025, .975].
            if None, defaults to width.

        meshgrid : bool, whether to return and accept meshgrids.

            Useful for creating outputs that are suitable for
            3D plotting.

            Note, for simple terms with no interactions, the output
            of this function will be the same for ``meshgrid=True`` and
            ``meshgrid=False``, but the inputs will need to be different.

        Returns
        -------
        pdeps : np.array of shape (n_samples,)
        conf_intervals : list of length len(term)
            containing np.arrays of shape (n_samples, 2 or len(quantiles))

        Raises
        ------
        ValueError :
            If the term requested is an intercept
            since it does not make sense to process the intercept term.

        See Also
        --------
        generate_X_grid : for help creating meshgrids.
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        if not isinstance(term, int):
            raise ValueError(f"term must be an integer, but found term: {term}.")

        # ensure term exists
        if (term >= len(self.terms)) or (term < -1):
            raise ValueError(f"Term {term} out of range for model with {len(self.terms)} terms.")

        # cant do Intercept
        if self.terms[term].isintercept:
            raise ValueError("Cannot create grid for intercept term.")

        if X is None:
            X = self.generate_X_grid(term=term, meshgrid=meshgrid)

        if meshgrid:
            if not isinstance(X, tuple):
                raise ValueError(f"X must be a tuple of grids if `meshgrid=True`, but found X: {X}")
            shape = X[0].shape

            X = self._flatten_mesh(X, term=term)
            X = check_X(
                X,
                n_feats=self.statistics_["m_features"],
                edge_knots=self.edge_knots_,
                dtypes=self.dtype,
                features=self.feature,
            )

        modelmat = self._modelmat(X, term=term)
        pdep = self._linear_predictor(modelmat=modelmat, term=term)
        out = [pdep]

        compute_quantiles = (width is not None) or (quantiles is not None)
        if compute_quantiles:
            conf_intervals = self._get_quantiles(
                X, width=width, quantiles=quantiles, modelmat=modelmat, lp=pdep, term=term, xform=False
            )

            out += [conf_intervals]

        if meshgrid:
            for i, array in enumerate(out):
                print(i, array.shape)
                # add extra dimensions arising from multiple confidence intervals
                if array.ndim > 1:
                    depth = array.shape[-1]
                    shape += (depth,)

                print(f"reshape from {array.shape} to {shape}")
                out[i] = np.reshape(array, shape)

        if compute_quantiles:
            return out

        return out[0]

    def summary(self):
        """produce a summary of the model statistics

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        # high-level model summary
        width_details = 47
        width_results = 58

        model_fmt = [(self.__class__.__name__, "model_details", width_details), ("", "model_results", width_results)]

        model_details = []

        objective = "UBRE" if self.distribution._known_scale else "GCV"

        model_details.append(
            {
                "model_details": space_row(
                    "Distribution:", self.distribution.__class__.__name__, total_width=width_details
                ),
                "model_results": space_row(
                    "Effective DoF:", str(np.round(self.statistics_["edof"], 4)), total_width=width_results
                ),
            }
        )
        model_details.append(
            {
                "model_details": space_row("Link Function:", self.link.__class__.__name__, total_width=width_details),
                "model_results": space_row(
                    "Log Likelihood:", str(np.round(self.statistics_["loglikelihood"], 4)), total_width=width_results
                ),
            }
        )
        model_details.append(
            {
                "model_details": space_row(
                    "Number of Samples:", str(self.statistics_["n_samples"]), total_width=width_details
                ),
                "model_results": space_row(
                    "AIC: ", str(np.round(self.statistics_["AIC"], 4)), total_width=width_results
                ),
            }
        )
        model_details.append(
            {
                "model_results": space_row(
                    "AICc: ", str(np.round(self.statistics_["AICc"], 4)), total_width=width_results
                )
            }
        )
        model_details.append(
            {
                "model_results": space_row(
                    objective + ":", str(np.round(self.statistics_[objective], 4)), total_width=width_results
                )
            }
        )
        model_details.append(
            {
                "model_results": space_row(
                    "Scale:", str(np.round(self.statistics_["scale"], 4)), total_width=width_results
                )
            }
        )
        model_details.append(
            {
                "model_results": space_row(
                    "Pseudo R-Squared:",
                    str(np.round(self.statistics_["pseudo_r2"]["explained_deviance"], 4)),
                    total_width=width_results,
                )
            }
        )

        # term summary
        data = []

        for i, term in enumerate(self.terms):
            # TODO bug: if the number of samples is less than the number of coefficients
            # we cant get the edof per term
            if len(self.statistics_["edof_per_coef"]) == len(self.coef_):
                idx = self.terms.get_coef_indices(i)
                edof = np.round(self.statistics_["edof_per_coef"][idx].sum(), 1)
            else:
                edof = ""

            term_data = {
                "feature_func": repr(term),
                "lam": "" if term.isintercept else np.round(flatten(term.lam), 4),
                "rank": "{}".format(term.n_coefs),
                "edof": "{}".format(edof),
                "p_value": "%.2e" % (self.statistics_["p_values"][i]),
                "sig_code": sig_code(self.statistics_["p_values"][i]),
            }

            data.append(term_data)

        fmt = [
            ("Feature Function", "feature_func", 33),
            ("Lambda", "lam", 20),
            ("Rank", "rank", 12),
            ("EDoF", "edof", 12),
            ("P > x", "p_value", 12),
            ("Sig. Code", "sig_code", 12),
        ]

        print(TablePrinter(model_fmt, ul="=", sep=" ")(model_details))
        print("=" * 106)
        print(TablePrinter(fmt, ul="=")(data))
        print("=" * 106)
        print("Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print()
        print(
            "WARNING: Fitting splines and a linear function to a feature introduces a model identifiability problem\n"
            "         which can cause p-values to appear significant when they are not."
        )
        print()
        print(
            "WARNING: p-values calculated in this manner behave correctly for un-penalized models or models with\n"
            "         known smoothing parameters, but when smoothing parameters have been estimated, the p-values\n"
            "         are typically lower than they should be, meaning that the tests reject the null too readily."
        )

        # P-VALUE BUG
        warnings.warn(
            "KNOWN BUG: p-values computed in this summary are likely "
            "much smaller than they should be. \n \n"
            "Please do not make inferences based on these values! \n\n"
            "Collaborate on a solution, and stay up to date at: \n"
            "github.com/dswah/pyGAM/issues/163 \n",
            stacklevel=2,
        )

    def gridsearch(self, X, y, weights=None, return_scores=False, keep_best=True, objective="auto", **param_grids):
        """
        Performs a grid search over a space of parameters for a given
        objective

        Warnings
        --------
        ``gridsearch`` is lazy and will not remove useless combinations
        from the search space.

        Also, it is not recommended to search over a grid that alternates
        between known scales and unknown scales, as the scores of the
        candidate models will not be comparable.

        Parameters
        ----------
        X : array-like
          input data of shape (n_samples, m_features)

        y : array-like
          label data of shape (n_samples,)

        weights : array-like shape (n_samples,), optional
            sample weights

        return_scores : boolean, optional
            whether to return the hyperpamaters and score for each element
            in the grid

        keep_best : boolean, optional
            whether to keep the best GAM as self.

        objective : {'auto', 'AIC', 'AICc', 'GCV', 'UBRE'}, optional
            Metric to optimize.
            If `auto`, then grid search will optimize `GCV` for models with unknown
            scale and `UBRE` for models with known scale.

        **kwargs
            pairs of parameters and iterables of floats, or
            parameters and iterables of iterables of floats.

            If no parameter are specified, ``lam=np.logspace(-3, 3, 11)`` is used.
            This results in a 11 points, placed diagonally across lam space.

            If grid is iterable of iterables of floats,
            the outer iterable must have length ``m_features``.
            the cartesian product of the subgrids in the grid will be tested.

            If grid is a 2d numpy array,
            each row of the array will be tested.

            The method will make a grid of all the combinations of the parameters
            and fit a GAM to each combination.


        Returns
        -------
        if ``return_scores=True``:
            model_scores: dict containing each fitted model as keys and corresponding
            objective scores as values
        else:
            self: ie possibly the newly fitted model

        Examples
        --------
        For a model with 4 terms, and where we expect 4 lam values,
        our search space for lam must have 4 dimensions.

        We can search the space in 3 ways:

        1. via cartesian product by specifying the grid as a list.
        our grid search will consider ``11 ** 4`` points:

        > lam = np.logspace(-3, 3, 11)
        > lams = [lam] * 4
        > gam.gridsearch(X, y, lam=lams)

        2. directly by specifying the grid as a np.ndarray.
        This is useful for when the dimensionality of the search space
        is very large, and we would prefer to execute a randomized search:

        > lams = np.exp(np.random.random(50, 4) * 6 - 3)
        > gam.gridsearch(X, y, lam=lams)

        3. copying grids for parameters with multiple dimensions.
        if we specify a 1D np.ndarray for lam, we are implicitly testing the
        space where all points have the same value

        > gam.gridsearch(lam=np.logspace(-3, 3, 11))

        is equivalent to:

        > lam = np.logspace(-3, 3, 11)
        > lams = np.array([lam] * 4)
        > gam.gridsearch(X, y, lam=lams)
        """
        # special checks if model not fitted
        if not self._is_fitted:
            self._validate_params()

        y = check_y(y, self.link, self.distribution)
        X = check_X(X)

        X_to_check = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        y_to_check = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y
        check_X_y(X_to_check, y_to_check)

        # special checks if model not fitted
        if not self._is_fitted:
            self._validate_data_dep_params(X)

        if weights is not None:
            weights = np.array(weights).astype("f").ravel()
            weights = check_array(weights, input_name="sample weights", ensure_2d=False)
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype("float64")

        # validate objective
        objectives = ["auto", "GCV", "UBRE", "AIC", "AICc"]
        if objective not in objectives:
            msg = f"{objective} not in {objectives}"
            raise ValueError(msg)

        # check objective
        if self.distribution._known_scale:
            if objective == "GCV":
                raise ValueError("GCV should be used for models with unknown scale")
            if objective == "auto":
                objective = "UBRE"

        else:
            if objective == "UBRE":
                raise ValueError("UBRE should be used for models with known scale")
            if objective == "auto":
                objective = "GCV"

        # if no params, then set up default gridsearch
        if not bool(param_grids):
            param_grids["lam"] = np.logspace(-3, 3, 11)

        # validate params
        admissible_params = list(self.get_params()) + self._plural
        params = []
        grids = []

        for param, grid in list(param_grids.items()):
            # check param exists
            if param not in (admissible_params):
                raise ValueError("unknown parameter: {}".format(param))

            # check grid is iterable at all
            if not (isiterable(grid) and (len(grid) > 1)):
                raise ValueError(
                    "{} grid must either be iterable of "
                    "iterables, or an iterable of lengnth > 1, "
                    "but found {}".format(param, grid)
                )

            # prepare grid
            if any(isiterable(g) for g in grid):
                # get required parameter shape
                target_len = len(flatten(getattr(self, param)))

                # check if cartesian product needed
                cartesian = not isinstance(grid, np.ndarray) or grid.ndim != 2

                # build grid
                grid = [np.atleast_1d(g) for g in grid]

                # check chape
                msg = "{} grid should have {} columns, " "but found grid with {} columns".format(
                    param, target_len, len(grid)
                )

                if cartesian:
                    if len(grid) != target_len:
                        raise ValueError(msg)

                    # we should consider each element in `grid` its own dimension
                    grid = product(*grid)
                else:
                    if not all([len(subgrid) == target_len for subgrid in grid]):
                        raise ValueError(msg)

            # save param name and grid
            params.append(param)
            grids.append(grid)

        # set up data collection
        best_model = None  # keep the best model
        best_score = np.inf
        scores = []
        models = []

        # check if our model has been fitted already and store it
        if self._is_fitted:
            models.append(self)
            scores.append(self.statistics_[objective])

            # our model is currently the best
            best_model = models[-1]
            best_score = scores[-1]

        # loop through candidate model params
        for grid in product(*grids):
            # build dict of candidate model params
            param_grid = dict(zip(params, grid))

            try:
                # try fitting
                # define new model
                gam = deepcopy(self)
                gam.set_params(self.get_params())
                gam.set_params(**param_grid)

                # warm start with parameters from previous build
                if models:
                    coef = models[-1].coef_
                    gam.set_params(coef_=coef, force=True, verbose=False)
                gam.fit(X, y, weights)

            except ValueError as error:
                msg = str(error) + "\non model with params:\n" + str(param_grid)
                msg += "\nskipping...\n"
                if self.verbose:
                    warnings.warn(msg)
                continue

            # record results
            models.append(gam)
            scores.append(gam.statistics_[objective])

            logger.info(grid)
            logger.info(f"Score ({objective}) : {scores[-1]}")

            # track best
            if scores[-1] < best_score:
                best_model = models[-1]
                best_score = scores[-1]

        # problems
        if len(models) == 0:
            msg = "No models were fitted."
            if self.verbose:
                warnings.warn(msg)
            return self

        # copy over the best
        if keep_best:
            self.set_params(deep=True, force=True, **best_model.get_params(deep=True))
        if return_scores:
            return OrderedDict(zip(models, scores))
        else:
            return self

    def sample(self, X, y, quantity="y", sample_at_X=None, weights=None, n_draws=100, n_bootstraps=5, objective="auto"):
        """Simulate from the posterior of the coefficients and smoothing params.

        Samples are drawn from the posterior of the coefficients and smoothing
        parameters given the response in an approximate way. The GAM must
        already be fitted before calling this method; if the model has not
        been fitted, then an exception is raised. Moreover, it is recommended
        that the model and its hyperparameters be chosen with `gridsearch`
        (with the parameter `keep_best=True`) before calling `sample`, so that
        the result of that gridsearch can be used to generate useful response
        data and so that the model's coefficients (and their covariance matrix)
        can be used as the first bootstrap sample.

        These samples are drawn as follows. Details are in the reference below.

        1. ``n_bootstraps`` many "bootstrap samples" of the response (``y``) are
        simulated by drawing random samples from the model's distribution
        evaluated at the expected values (``mu``) for each sample in ``X``.

        2. A copy of the model is fitted to each of those bootstrap samples of
        the response. The result is an approximation of the distribution over
        the smoothing parameter ``lam`` given the response data ``y``.

        3. Samples of the coefficients are simulated from a multivariate normal
        using the bootstrap samples of the coefficients and their covariance
        matrices.

        Notes
        -----
        A ``gridsearch`` is done ``n_bootstraps`` many times, so keep
        ``n_bootstraps`` small. Make ``n_bootstraps < n_draws`` to take advantage
        of the expensive bootstrap samples of the smoothing parameters.

        Parameters
        -----------
        X : array of shape (n_samples, m_features)
              empirical input data

        y : array of shape (n_samples,)
              empirical response vector

        quantity : {'y', 'coef', 'mu'}, default: 'y'
            What quantity to return pseudorandom samples of.
            If `sample_at_X` is not None and `quantity` is either `'y'` or
            `'mu'`, then samples are drawn at the values of `X` specified in
            `sample_at_X`.

        sample_at_X : array of shape (n_samples_to_simulate, m_features) or
        None, optional
            Input data at which to draw new samples.

            Only applies for `quantity` equal to `'y'` or to `'mu`'.
            If `None`, then `sample_at_X` is replaced by `X`.

        weights : np.array of shape (n_samples,)
            sample weights

        n_draws : positive int, optional (default=100)
            The number of samples to draw from the posterior distribution of
            the coefficients and smoothing parameters

        n_bootstraps : positive int, optional (default=5)
            The number of bootstrap samples to draw from simulations of the
            response (from the already fitted model) to estimate the
            distribution of the smoothing parameters given the response data.
            If `n_bootstraps` is 1, then only the already fitted model's
            smoothing parameter is used, and the distribution over the
            smoothing parameters is not estimated using bootstrap sampling.

        objective : string, optional (default='auto'
            metric to optimize in grid search. must be in
            ['AIC', 'AICc', 'GCV', 'UBRE', 'auto']
            if 'auto', then grid search will optimize GCV for models with
            unknown scale and UBRE for models with known scale.

        Returns
        -------
        draws : 2D array of length n_draws
            Simulations of the given `quantity` using samples from the
            posterior distribution of the coefficients and smoothing parameter
            given the response data. Each row is a pseudorandom sample.

            If `quantity == 'coef'`, then the number of columns of `draws` is
            the number of coefficients (`len(self.coef_)`).

            Otherwise, the number of columns of `draws` is the number of
            rows of `sample_at_X` if `sample_at_X` is not `None` or else
            the number of rows of `X`.

        References
        ----------
        Simon N. Wood, 2006. Generalized Additive Models: an introduction with
        R. Section 4.9.3 (pages 198???199) and Section 5.4.2 (page 256???257).
        """
        if quantity not in {"mu", "coef", "y"}:
            raise ValueError("`quantity` must be one of 'mu', 'coef', 'y';" " got {}".format(quantity))

        coef_draws = self._sample_coef(
            X, y, weights=weights, n_draws=n_draws, n_bootstraps=n_bootstraps, objective=objective
        )

        if quantity == "coef":
            return coef_draws

        if sample_at_X is None:
            sample_at_X = X

        linear_predictor = self._modelmat(sample_at_X).dot(coef_draws.T)
        mu_shape_n_draws_by_n_samples = self.link.mu(linear_predictor, self.distribution).T
        if quantity == "mu":
            return mu_shape_n_draws_by_n_samples
        else:
            return self.distribution.sample(mu_shape_n_draws_by_n_samples)

    def _sample_coef(self, X, y, weights=None, n_draws=100, n_bootstraps=1, objective="auto"):
        """Simulate from the posterior of the coefficients.

        NOTE: A `gridsearch` is done `n_bootstraps` many times, so keep
        `n_bootstraps` small. Make `n_bootstraps < n_draws` to take advantage
        of the expensive bootstrap samples of the smoothing parameters.

        Parameters
        -----------
        X : array of shape (n_samples, m_features)
              input data

        y : array of shape (n_samples,)
              response vector

        weights : np.array of shape (n_samples,)
            sample weights

        n_draws : positive int, optional (default=100
            The number of samples to draw from the posterior distribution of
            the coefficients and smoothing parameters

        n_bootstraps : positive int, optional (default=1
            The number of bootstrap samples to draw from simulations of the
            response (from the already fitted model) to estimate the
            distribution of the smoothing parameters given the response data.
            If `n_bootstraps` is 1, then only the already fitted model's
            smoothing parameters is used.

        objective : string, optional (default='auto'
            metric to optimize in grid search. must be in
            ['AIC', 'AICc', 'GCV', 'UBRE', 'auto']
            if 'auto', then grid search will optimize GCV for models with
            unknown scale and UBRE for models with known scale.

        Returns
        -------
        coef_samples : array of shape (n_draws, n_samples)
            Approximate simulations of the coefficients drawn from the
            posterior distribution of the coefficients and smoothing
            parameters given the response data

        References
        ----------
        Simon N. Wood, 2006. Generalized Additive Models: an introduction with
        R. Section 4.9.3 (pages 198???199) and Section 5.4.2 (page 256???257).
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")
        if n_bootstraps < 1:
            raise ValueError("n_bootstraps must be >= 1;" " got {}".format(n_bootstraps))
        if n_draws < 1:
            raise ValueError("n_draws must be >= 1;" " got {}".format(n_draws))

        coef_bootstraps, cov_bootstraps = self._bootstrap_samples_of_smoothing(
            X, y, weights=weights, n_bootstraps=n_bootstraps, objective=objective
        )
        coef_draws = self._simulate_coef_from_bootstraps(n_draws, coef_bootstraps, cov_bootstraps)

        return coef_draws

    def _bootstrap_samples_of_smoothing(self, X, y, weights=None, n_bootstraps=1, objective="auto"):
        """Sample the smoothing parameters using simulated response data.


        For now, the grid of `lam` values is 11 random points in M-dimensional
        space, where M = the number of lam values, ie len(flatten(gam.lam))

        all values are in [1e-3, 1e3]
        """
        mu = self.predict_mu(X)  # Wood pg. 198 step 1
        coef_bootstraps = [self.coef_]
        cov_bootstraps = [load_diagonal(self.statistics_["cov"])]

        for _ in range(n_bootstraps - 1):  # Wood pg. 198 step 2
            # generate response data from fitted model (Wood pg. 198 step 3)
            y_bootstrap = self.distribution.sample(mu)

            # fit smoothing parameters on the bootstrap data
            # (Wood pg. 198 step 4)
            # TODO: Either enable randomized searches over hyperparameters
            # (like in sklearn's RandomizedSearchCV), or draw enough samples of
            # `lam` so that each of these bootstrap samples get different
            # values of `lam`. Right now, each bootstrap sample uses the exact
            # same grid of values for `lam`, so it is not worth setting
            # `n_bootstraps > 1`.
            gam = deepcopy(self)
            gam.set_params(self.get_params())

            # create a random search of 11 points in lam space
            # with all values in [1e-3, 1e3]
            lam_grid = np.random.randn(11, len(flatten(self.lam))) * 6 - 3
            lam_grid = np.exp(lam_grid)
            gam.gridsearch(X, y_bootstrap, weights=weights, lam=lam_grid, objective=objective)
            lam = gam.lam

            # fit coefficients on the original data given the smoothing params
            # (Wood pg. 199 step 5)
            gam = deepcopy(self)
            gam.set_params(self.get_params())
            gam.lam = lam
            gam.fit(X, y, weights=weights)

            coef_bootstraps.append(gam.coef_)

            cov = load_diagonal(gam.statistics_["cov"])

            cov_bootstraps.append(cov)
        return coef_bootstraps, cov_bootstraps

    def _simulate_coef_from_bootstraps(self, n_draws, coef_bootstraps, cov_bootstraps):
        """Simulate coefficients using bootstrap samples."""
        # Sample indices uniformly from {0, ..., n_bootstraps - 1}
        # (Wood pg. 199 step 6)
        random_bootstrap_indices = np.random.choice(np.arange(len(coef_bootstraps)), size=n_draws, replace=True)

        # Simulate `n_draws` many random coefficient vectors from a
        # multivariate normal distribution with mean and covariance given by
        # the bootstrap samples (indexed by `random_bootstrap_indices`) of
        # `coef_bootstraps` and `cov_bootstraps`. Because it's faster to draw
        # many samples from a certain distribution all at once, we make a dict
        # mapping bootstrap indices to draw indices and use the `size`
        # parameter of `np.random.multivariate_normal` to sample the draws
        # needed from that bootstrap sample all at once.
        bootstrap_index_to_draw_indices = defaultdict(list)
        for draw_index, bootstrap_index in enumerate(random_bootstrap_indices):
            bootstrap_index_to_draw_indices[bootstrap_index].append(draw_index)

        coef_draws = np.empty((n_draws, len(self.coef_)))

        for bootstrap, draw_indices in bootstrap_index_to_draw_indices.items():
            coef_draws[draw_indices] = np.random.multivariate_normal(
                coef_bootstraps[bootstrap], cov_bootstraps[bootstrap], size=len(draw_indices)
            )

        return coef_draws


class LinearGAM(GAM):
    """Linear GAM

    This is a GAM with a Normal error distribution, and an identity link.

    Parameters
    ----------
    terms : expression specifying terms to model, optional.

        By default a univariate spline term will be allocated for each feature.

        For example:

        > GAM(s(0) + l(1) + f(2) + te(3, 4))

        will fit a spline term on feature 0, a linear term on feature 1,
        a factor term on feature 2, and a tensor term on features 3 and 4.

    callbacks : list of str or list of CallBack objects, optional
        Names of callback objects to call during the optimization loop.

    fit_intercept : bool, optional
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
        Note: the intercept receives no smoothing penalty.

    max_iter : int, optional
        Maximum number of iterations allowed for the solver to converge.

    tol : float, optional
        Tolerance for stopping criteria.

    verbose : bool, optional
        whether to show pyGAM warnings.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as ``{callback: [...]}``

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """

    def __init__(
        self,
        terms="auto",
        max_iter=100,
        tol=1e-4,
        scale=None,
        callbacks=["deviance", "diffs"],
        fit_intercept=True,
        verbose=False,
        **kwargs,
    ):
        self.scale = scale
        super().__init__(
            terms=terms,
            distribution=NormalDist(scale=self.scale),
            link="identity",
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
            verbose=verbose,
            **kwargs,
        )

        self._exclude += ["distribution", "link"]

        logger.info("Created linearGAM")

    def _validate_params(self):
        """
        method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        self.distribution = NormalDist(scale=self.scale)
        super()._validate_params()

    def prediction_intervals(self, X, width=0.95, quantiles=None):
        """
        estimate prediction intervals for LinearGAM

        Parameters
        ----------
        X : array-like of shape (n_samples, m_features)
            input data matrix
        width : float on [0,1], optional (default=0.95
        quantiles : array-like of floats in [0, 1], default: None)
            instead of specifying the prediciton width, one can specify the
            quantiles. so width=.95 is equivalent to quantiles=[.025, .975]

        Returns
        -------
        intervals: np.array of shape (n_samples, 2 or len(quantiles))
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        X = check_array(X, force_all_finite=True, input_name="X", ensure_2d=True, estimator=self)

        X = check_X(
            X,
            n_feats=self.statistics_["m_features"],
            edge_knots=self.edge_knots_,
            dtypes=self.dtype,
            features=self.feature,
        )

        return self._get_quantiles(X, width, quantiles, prediction=True)


class LogisticGAM(GAM):
    """Logistic GAM

    This is a GAM with a Binomial error distribution, and a logit link.

    Parameters
    ----------
    terms : expression specifying terms to model, optional.

        By default a univariate spline term will be allocated for each feature.

        For example:

        > GAM(s(0) + l(1) + f(2) + te(3, 4))

        will fit a spline term on feature 0, a linear term on feature 1,
        a factor term on feature 2, and a tensor term on features 3 and 4.

    callbacks : list of str or list of CallBack objects, optional
        Names of callback objects to call during the optimization loop.

    fit_intercept : bool, optional
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
        Note: the intercept receives no smoothing penalty.

    max_iter : int, optional
        Maximum number of iterations allowed for the solver to converge.

    tol : float, optional
        Tolerance for stopping criteria.

    verbose : bool, optional
        whether to show pyGAM warnings.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as ``{callback: [...]}``

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """

    def __init__(
        self,
        terms="auto",
        max_iter=100,
        tol=1e-4,
        callbacks=["deviance", "diffs", "accuracy"],
        fit_intercept=True,
        verbose=False,
        **kwargs,
    ):
        # call super
        super().__init__(
            terms=terms,
            distribution="binomial",
            link="logit",
            max_iter=max_iter,
            tol=tol,
            callbacks=callbacks,
            fit_intercept=fit_intercept,
            verbose=verbose,
            **kwargs,
        )
        # ignore any variables
        self._exclude += ["distribution", "link"]

    def accuracy(self, X=None, y=None, mu=None):
        """
        computes the accuracy of the LogisticGAM

        Parameters
        ----------
        note: X or mu must be defined. defaults to mu

        X : array-like of shape (n_samples, m_features), optional (default=None)
            containing input data
        y : array-like of shape (n,)
            containing target data
        mu : array-like of shape (n_samples,), optional (default=None
            expected value of the targets given the model and inputs

        Returns
        -------
        float in [0, 1]
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        y = check_y(y, self.link, self.distribution)
        if X is not None:
            X = check_X(
                X,
                n_feats=self.statistics_["m_features"],
                edge_knots=self.edge_knots_,
                dtypes=self.dtype,
                features=self.feature,
            )

        if mu is None:
            mu = self.predict_mu(X)

        return ((mu > 0.5).astype(int) == y).mean()

    def score(self, X, y):
        """
        method to compute the accuracy for a trained model for a given X data and y labels

        Parameters
        ----------
        X : array-like
            Input data array of shape (n_samples, m_features)
        y : array-like
            Output data vector of shape (n_samples,)

        Returns
        -------
        accuracy score: np.array() (n_samples,)

        """

        return self.accuracy(X, y, None)

    def predict(self, X):
        """
        preduct binary targets given model and input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing binary targets under the model
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")
        X = check_array(X, force_all_finite=True, input_name="X", ensure_2d=True, estimator=self)
        return self.predict_mu(X) > 0.5

    def predict_proba(self, X):
        """
        preduct targets given model and input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features), optional (default=None
            containing the input dataset

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing expected values under the model
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")
        X = check_array(X, force_all_finite=True, input_name="X", ensure_2d=True, estimator=self)
        return self.predict_mu(X)


class PoissonGAM(GAM):
    """Poisson GAM

    This is a GAM with a Poisson error distribution, and a log link.

    Parameters
    ----------
    terms : expression specifying terms to model, optional.

        By default a univariate spline term will be allocated for each feature.

        For example:

        > GAM(s(0) + l(1) + f(2) + te(3, 4))

        will fit a spline term on feature 0, a linear term on feature 1,
        a factor term on feature 2, and a tensor term on features 3 and 4.

    callbacks : list of str or list of CallBack objects, optional
        Names of callback objects to call during the optimization loop.

    fit_intercept : bool, optional
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
        Note: the intercept receives no smoothing penalty.

    max_iter : int, optional
        Maximum number of iterations allowed for the solver to converge.

    tol : float, optional
        Tolerance for stopping criteria.

    verbose : bool, optional
        whether to show pyGAM warnings.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as ``{callback: [...]}``

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """

    def __init__(
        self,
        terms="auto",
        max_iter=100,
        tol=1e-4,
        callbacks=["deviance", "diffs"],
        fit_intercept=True,
        verbose=False,
        **kwargs,
    ):
        # call super
        super().__init__(
            terms=terms,
            distribution="poisson",
            link="log",
            max_iter=max_iter,
            tol=tol,
            callbacks=callbacks,
            fit_intercept=fit_intercept,
            verbose=verbose,
            **kwargs,
        )
        # ignore any variables
        self._exclude += ["distribution", "link"]

    def _loglikelihood(self, y, mu, weights=None, rescale_y=True):
        """
        compute the log-likelihood of the dataset using the current model

        Parameters
        ---------
        y : array-like of shape (n,)
            containing target values
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs
        weights : array-like of shape (n,)
            containing sample weights
        rescale_y : boolean, defaul: True
            whether to scale the targets back up.
            useful when fitting with an exposure, in which case the count observations
            were scaled into rates. this rescales rates into counts.

        Returns
        -------
        log-likelihood : np.array of shape (n,)
            containing log-likelihood scores
        """
        if rescale_y:
            y = np.round(y * weights).astype("int")

        return self.distribution.log_pdf(y=y, mu=mu, weights=weights).sum()

    def loglikelihood(self, X, y, exposure=None, weights=None):
        """
        compute the log-likelihood of the dataset using the current model

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing the input dataset
        y : array-like of shape (n,)
            containing target values
        exposure : array-like shape (n_samples,) or None, default: None
            containing exposures
            if None, defaults to array of ones
        weights : array-like of shape (n,)
            containing sample weights

        Returns
        -------
        log-likelihood : np.array of shape (n,)
            containing log-likelihood scores
        """
        y = check_y(y, self.link, self.distribution)
        mu = self.predict_mu(X)

        if weights is not None:
            weights = np.array(weights).astype("f").ravel()
            weights = check_array(weights, input_name="sample weights", ensure_2d=False)
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype("float64")

        y, weights = self._exposure_to_weights(y, exposure, weights)
        return self._loglikelihood(y, mu, weights=weights, rescale_y=True)

    def _exposure_to_weights(self, y, exposure=None, weights=None):
        """simple tool to create a common API

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.
        exposure : array-like shape (n_samples,) or None, default: None
            containing exposures
            if None, defaults to array of ones
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        y : y normalized by exposure
        weights : array-like shape (n_samples,)
        """
        y = y.ravel()

        if exposure is not None:
            exposure = np.array(exposure).astype("f").ravel()
            exposure = check_array(exposure, input_name="sample exposure", ensure_2d=False)
        else:
            exposure = np.ones_like(y.ravel()).astype("float64")

        # check data
        exposure = exposure.ravel()
        check_lengths(y, exposure)

        # normalize response
        y = y / exposure

        if weights is not None:
            weights = np.array(weights).astype("f").ravel()
            weights = check_array(weights, input_name="sample weights", ensure_2d=False)
        else:
            weights = np.ones_like(y).astype("float64")
        check_lengths(weights, exposure)

        # set exposure as the weight
        # we do this because we have divided our response
        # so if we make an error of 1 now, we need it to count more heavily.
        weights = weights * exposure

        return y, weights

    def fit(self, X, y, exposure=None, weights=None):
        """Fit the generalized additive model.

        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training vectors, where n_samples is the number of samples
            and m_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.

        exposure : array-like shape (n_samples,) or None, default: None
            containing exposures
            if None, defaults to array of ones

        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        self : object
            Returns fitted GAM object
        """
        y, weights = self._exposure_to_weights(y, exposure, weights)
        return super().fit(X, y, weights)

    def predict(self, X, exposure=None):
        """
        preduct expected value of target given model and input X
        often this is done via expected value of GAM given input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features), default: None
            containing the input dataset

        exposure : array-like shape (n_samples,) or None, default: None
            containing exposures
            if None, defaults to array of ones

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing predicted values under the model
        """
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")

        X = check_array(X, force_all_finite=True, input_name="X", ensure_2d=True, estimator=self)

        X = check_X(
            X,
            n_feats=self.statistics_["m_features"],
            edge_knots=self.edge_knots_,
            dtypes=self.dtype,
            features=self.feature,
        )

        if exposure is not None:
            exposure = np.array(exposure).astype("f")
        else:
            exposure = np.ones(X.shape[0]).astype("f")
        check_lengths(X, exposure)

        return self.predict_mu(X) * exposure

    def gridsearch(
        self, X, y, exposure=None, weights=None, return_scores=False, keep_best=True, objective="auto", **param_grids
    ):
        """
        performs a grid search over a space of parameters for a given objective

        NOTE:
        gridsearch method is lazy and will not remove useless combinations
        from the search space, eg.

        > n_splines=np.arange(5,10)
        > fit_splines=[True, False]

        will result in 10 loops, of which 5 are equivalent because
        even though fit_splines==False

        it is not recommended to search over a grid that alternates
        between known scales and unknown scales, as the scores of the
        candidate models will not be comparable.

        Parameters
        ----------
        X : array
          input data of shape (n_samples, m_features)

        y : array
          label data of shape (n_samples,)

        exposure : array-like shape (n_samples,) or None, default: None
            containing exposures
            if None, defaults to array of ones

        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        return_scores : boolean, default False
          whether to return the hyperpamaters
          and score for each element in the grid

        keep_best : boolean
          whether to keep the best GAM as self.
          default: True

        objective : string, default: 'auto'
          metric to optimize. must be in ['AIC', 'AICc', 'GCV', 'UBRE', 'auto']
          if 'auto', then grid search will optimize GCV for models with unknown
          scale and UBRE for models with known scale.

        **kwargs : dict, default {'lam': np.logspace(-3, 3, 11)}
          pairs of parameters and iterables of floats, or
          parameters and iterables of iterables of floats.

          if iterable of iterables of floats, the outer iterable must have
          length m_features.

          the method will make a grid of all the combinations of the parameters
          and fit a GAM to each combination.


        Returns
        -------
        if return_values == True:
            model_scores : dict
                Contains each fitted model as keys and corresponding
                objective scores as values
        else:
            self, ie possibly the newly fitted model
        """
        y, weights = self._exposure_to_weights(y, exposure, weights)
        return super().gridsearch(
            X, y, weights=weights, return_scores=return_scores, keep_best=keep_best, objective=objective, **param_grids
        )


class GammaGAM(GAM):
    """Gamma GAM

    This is a GAM with a Gamma error distribution, and a log link.

    NB
    Although canonical link function for the Gamma GLM is the inverse link,
    this function can create problems for numerical software because it becomes
    difficult to enforce the requirement that the mean of the Gamma distribution
    be positive. The log link guarantees this.

    If you need to use the inverse link function, simply construct a custom GAM:

    >>> from pygam import GAM
    >>> gam = GAM(distribution='gamma', link='inverse')


    Parameters
    ----------
    terms : expression specifying terms to model, optional.

        By default a univariate spline term will be allocated for each feature.

        For example:

        > GAM(s(0) + l(1) + f(2) + te(3, 4))

        will fit a spline term on feature 0, a linear term on feature 1,
        a factor term on feature 2, and a tensor term on features 3 and 4.

    callbacks : list of str or list of CallBack objects, optional
        Names of callback objects to call during the optimization loop.

    fit_intercept : bool, optional
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
        Note: the intercept receives no smoothing penalty.

    max_iter : int, optional
        Maximum number of iterations allowed for the solver to converge.

    tol : float, optional
        Tolerance for stopping criteria.

    verbose : bool, optional
        whether to show pyGAM warnings.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as ``{callback: [...]}``

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """

    def __init__(
        self,
        terms="auto",
        max_iter=100,
        tol=1e-4,
        scale=None,
        callbacks=["deviance", "diffs"],
        fit_intercept=True,
        verbose=False,
        **kwargs,
    ):
        self.scale = scale
        super().__init__(
            terms=terms,
            distribution=GammaDist(scale=self.scale),
            link="log",
            max_iter=max_iter,
            tol=tol,
            callbacks=callbacks,
            fit_intercept=fit_intercept,
            verbose=verbose,
            **kwargs,
        )

        self._exclude += ["distribution", "link"]

    def _validate_params(self):
        """
        method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        self.distribution = GammaDist(scale=self.scale)
        super()._validate_params()


class InvGaussGAM(GAM):
    """Inverse Gaussian GAM

    This is a GAM with a Inverse Gaussian error distribution, and a log link.

    NB
    Although canonical link function for the Inverse Gaussian GLM is the inverse squared link,
    this function can create problems for numerical software because it becomes
    difficult to enforce the requirement that the mean of the Inverse Gaussian distribution
    be positive. The log link guarantees this.

    If you need to use the inverse squared link function, simply construct a custom GAM:

    >>> from pygam import GAM
    >>> gam = GAM(distribution='inv_gauss', link='inv_squared')


    Parameters
    ----------
    terms : expression specifying terms to model, optional.

        By default a univariate spline term will be allocated for each feature.

        For example:

        > GAM(s(0) + l(1) + f(2) + te(3, 4))

        will fit a spline term on feature 0, a linear term on feature 1,
        a factor term on feature 2, and a tensor term on features 3 and 4.

    callbacks : list of str or list of CallBack objects, optional
        Names of callback objects to call during the optimization loop.

    fit_intercept : bool, optional
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
        Note: the intercept receives no smoothing penalty.

    max_iter : int, optional
        Maximum number of iterations allowed for the solver to converge.

    tol : float, optional
        Tolerance for stopping criteria.

    verbose : bool, optional
        whether to show pyGAM warnings.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as ``{callback: [...]}``

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """

    def __init__(
        self,
        terms="auto",
        max_iter=100,
        tol=1e-4,
        scale=None,
        callbacks=["deviance", "diffs"],
        fit_intercept=True,
        verbose=False,
        **kwargs,
    ):
        self.scale = scale
        super().__init__(
            terms=terms,
            distribution=InvGaussDist(scale=self.scale),
            link="log",
            max_iter=max_iter,
            tol=tol,
            callbacks=callbacks,
            fit_intercept=fit_intercept,
            verbose=verbose,
            **kwargs,
        )

        self._exclude += ["distribution", "link"]

    def _validate_params(self):
        """
        method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        self.distribution = InvGaussDist(scale=self.scale)
        super()._validate_params()


class ExpectileGAM(GAM):
    """Expectile GAM

    This is a GAM with a Normal distribution and an Identity Link,
    but minimizing the Least Asymmetrically Weighted Squares

    https://freakonometrics.hypotheses.org/files/2017/05/erasmus-1.pdf
    https://sites.google.com/site/csphilipps/expectiles


    Parameters
    ----------
    terms : expression specifying terms to model, optional.

        By default a univariate spline term will be allocated for each feature.

        For example:

        > GAM(s(0) + l(1) + f(2) + te(3, 4))

        will fit a spline term on feature 0, a linear term on feature 1,
        a factor term on feature 2, and a tensor term on features 3 and 4.

    callbacks : list of str or list of CallBack objects, optional
        Names of callback objects to call during the optimization loop.

    fit_intercept : bool, optional
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
        Note: the intercept receives no smoothing penalty.

    max_iter : int, optional
        Maximum number of iterations allowed for the solver to converge.

    tol : float, optional
        Tolerance for stopping criteria.

    verbose : bool, optional
        whether to show pyGAM warnings.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as ``{callback: [...]}``

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """

    def __init__(
        self,
        terms="auto",
        max_iter=100,
        tol=1e-4,
        scale=None,
        callbacks=["deviance", "diffs"],
        fit_intercept=True,
        expectile=0.5,
        verbose=False,
        **kwargs,
    ):
        self.scale = scale
        self.expectile = expectile
        super().__init__(
            terms=terms,
            distribution=NormalDist(scale=self.scale),
            link="identity",
            max_iter=max_iter,
            tol=tol,
            callbacks=callbacks,
            fit_intercept=fit_intercept,
            verbose=verbose,
            **kwargs,
        )

        self._exclude += ["distribution", "link"]

    def _validate_params(self):
        """
        method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        if not (0 < self.expectile < 1):
            raise ValueError("expectile must be in (0,1), but found {self.expectile}")

        super()._validate_params()

    def _W(self, mu, weights, y):
        """
        compute the PIRLS weights for model predictions.

        TODO lets verify the formula for this.
        if we use the square root of the mu with the stable opt,
        we get the same results as when we use non-sqrt mu with naive opt.

        this makes me think that they are equivalent.

        also, using non-sqrt mu with stable opt gives very small edofs for even lam=0.001
        and the parameter variance is huge. this seems strange to me.

        computed [V * d(link)/d(mu)] ^(-1/2) by hand and the math checks out as hoped.

        ive since moved the square to the naive pirls method to make the code modular.

        Parameters
        ---------
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs
        weights : array-like of shape (n_samples,)
            containing sample weights
        y = array-like of shape (n_samples,) or None, default None
            useful for computing the asymmetric weight.

        Returns
        -------
        weights : scipy.sparse array of shape (n_samples, n_samples)
        """
        # asymmetric weight
        asym = (y > mu) * self.expectile + (y <= mu) * (1 - self.expectile)

        return (
            self.link.gradient(mu, self.distribution) ** 2 * self.distribution.V(mu=mu) * weights**-1
        ) ** -0.5 * asym**0.5

    def _get_quantile_ratio(self, X, y):
        """find the expirical quantile of the model

        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training vectors, where n_samples is the number of samples
            and m_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.

        Returns
        -------
        ratio : float on [0, 1]
        """
        y_predictions = self.predict(X)
        return (y_predictions > y).mean()

    def fit_quantile(self, X, y, quantile, max_iter=20, tol=0.01, weights=None):
        """fit ExpectileGAM to a desired quantile via binary search

        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training vectors, where n_samples is the number of samples
            and m_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.
        quantile : float on (0, 1)
            desired quantile to fit.
        max_iter : int, default: 20
            maximum number of binary search iterations to perform
        tol : float > 0, default: 0.01
            maximum distance between desired quantile and fitted quantile
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        self : fitted GAM object
        """

        def _within_tol(a, b, tol):
            return abs(a - b) <= tol

        # validate arguments
        if not (0 < quantile < 1):
            raise ValueError(f"`quantile` must be in (0, 1), but found: {quantile}")

        if tol <= 0:
            raise ValueError(f"`tol` must be float > 0, but found: {tol}")

        if max_iter <= 0:
            raise ValueError("`max_iter` must be int > 0, but found: {max_iter}")

        # perform a first fit if necessary
        if not self._is_fitted:
            self.fit(X, y, weights=weights)

        # Perform binary search
        # The goal is to choose `expectile` such that the empirical quantile
        # matches the desired quantile. The reason for not using
        # scipy.optimize.bisect is that bisect evalutes the endpoints first,
        # resulting in extra unneccesary fits (we assume that 0 -> 0 and 1 -> 1)
        min_, max_ = 0.0, 1.0
        for iteration in range(max_iter):
            empirical_quantile = self._get_quantile_ratio(X, y)
            logger.debug(f"Fitting with expectile={self.expectile} gave an empirical quantile {empirical_quantile}")
            logger.debug(f"Attempting to get close to quantile={quantile}")

            if _within_tol(empirical_quantile, quantile, tol):
                break

            if empirical_quantile < quantile:
                min_ = self.expectile  # Move up
            else:
                max_ = self.expectile  # Move down

            expectile = (min_ + max_) / 2.0
            logger.debug(f"Fitting with expectile={expectile}")
            self.set_params(expectile=expectile)
            self.fit(X, y, weights=weights)

        # print diagnostics
        if not _within_tol(empirical_quantile, quantile, tol) and self.verbose:
            warnings.warn(f"Maximum iterations of {max_iter} reached, but tolerance {tol} not achieved.")

        return self


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules"])

    if False:
        rng = np.random.default_rng(21)
        X = rng.normal(size=(1000, 3))
        y = 5 + np.array([np.sin(X[:, i] * 1 * (i + 1)) for i in range(X.shape[1])]).sum(axis=0)
        y = y + rng.normal(size=(X.shape[0]), scale=0.1)

        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt

        for i in range(X.shape[1]):
            X_sorted = X[np.argsort(X[:, i]), :]
            y_sorted = y[np.argsort(X[:, i])]

            plt.plot(X_sorted[:, i], y_sorted)
            plt.show()

        # =====================================================================

        from scipy.optimize import minimize

        methods = "Nelder-Mead,L-BFGS-B,TNC,SLSQP,Powell,trust-constr".split(",")

        x0 = np.ones(X.shape[1]) * 100

        for method in methods:
            print("==============================================")
            print(f"================= {method} =====================")
            print("==============================================")

            scores = []

            def func(x):
                global scores
                lam = list(x)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
                gam = LinearGAM(lam=lam).fit(X_train, y_train)
                score = np.mean((y_test - gam.predict(X_test)) ** 2)

                scores.append(score)

                return score

            minimize(
                func,
                x0=x0,
                method=method,
                bounds=[(0, np.inf) for _ in range(len(x0))],
            )

            plt.title(method)
            plt.plot(scores)
            plt.grid(True)
            plt.show()

            import time

            time.sleep(3)
