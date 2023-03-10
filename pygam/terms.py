"""
Link functions
"""
import collections.abc
import functools
import numbers
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import defaultdict
from copy import deepcopy

import numpy as np
import scipy as sp

from pygam.core import Core, nice_repr
from pygam.log import setup_custom_logger
from pygam.penalties import CONSTRAINTS, PENALTIES
from pygam.utils import b_spline_basis, check_param, flatten, gen_edge_knots, isiterable, tensor_product

logger = setup_custom_logger(__name__)


EPS = np.finfo(np.float64).eps  # machine epsilon


class Term(Core, metaclass=ABCMeta):
    def __init__(
        self,
        feature,
        lam=0.6,
        dtype="numerical",
        fit_linear=False,
        fit_splines=True,
        penalties="auto",
        constraints=None,
        verbose=False,
    ):
        """creates an instance of a Term

        Parameters
        ----------
        feature : int
            Index of the feature to use for the feature function.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        constraints : {None, 'convex', 'concave', 'monotonic_inc', 'monotonic_dec'}
            or callable or iterable

            Type of constraint to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.

        dtype : {'numerical', 'categorical'}
            String describing the data-type of the feature.

        fit_linear : bool
            whether to fit a linear model of the feature

        fit_splines : bool
            whether to fit spliens to the feature

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self.feature = feature

        self.lam = lam
        self.dtype = dtype
        self.fit_linear = fit_linear
        self.fit_splines = fit_splines
        self.penalties = penalties
        self.constraints = constraints
        self.verbose = verbose

        if not (hasattr(self, "_name")):
            self._name = "term"

        super().__init__(name=self._name)
        self._validate_arguments()

    def __len__(self):
        return 1

    def __eq__(self, other):
        if isinstance(other, Term):
            return self.info == other.info
        return False

    def __radd__(self, other):
        return TermList(other, self)

    def __add__(self, other):
        return TermList(self, other)

    def __mul__(self, other):
        raise NotImplementedError()

    def __repr__(self):
        if hasattr(self, "_minimal_name"):
            name = self._minimal_name
        else:
            name = self.__class__.__name__

        features = [] if self.feature is None else self.feature
        features = np.atleast_1d(features).tolist()
        return nice_repr(
            name, {}, line_width=self._line_width, line_offset=self._line_offset, decimals=4, args=features
        )

    def _validate_arguments(self):
        """Validate and sanitize arguments."""
        # dtype
        if self.dtype not in ["numerical", "categorical"]:
            msg = f"dtype must be in ['numerical', 'categorical'], found dtype = {self.dtype}"
            raise ValueError(msg)

        # fit_linear XOR fit_splines
        if self.fit_linear == self.fit_splines:
            msg = "Either 'fit_linear' or 'fit_splines' must be True, but not both."
            raise ValueError(msg)

        # Convert to list
        self.penalties = (
            [self.penalties]
            if (isinstance(self.penalties, str) or self.penalties is None or callable(self.penalties))
            else list(self.penalties)
        )

        # Validate each penalty
        for i, penalty in enumerate(self.penalties):
            is_callable = callable(penalty)
            is_valid_str = penalty in PENALTIES
            is_None = penalty is None

            if not any((is_callable, is_valid_str, is_None)):
                msg = f"Penalty number {i} ({penalty}) not in {list(PENALTIES.keys())}"
                raise ValueError(msg)

        # check lams and distribute to penalites
        if not isinstance(self.lam, collections.abc.Iterable):
            self.lam = [self.lam]

        for lam in self.lam:
            if not isinstance(lam, numbers.Real):
                raise TypeError("Parameter 'lam' must be a number")
            if lam < 0:
                raise ValueError("Paramter 'lam' must be >= 0")

        if len(self.lam) == 1:
            self.lam = self.lam * len(self.penalties)

        if len(self.lam) != len(self.penalties):
            msg = f"Length of penalties ({len(self.penalties)}) does not match length of lam ({len(self.lam)})"
            raise ValueError(msg)

        # constraints
        self.constraints = (
            [self.constraints]
            if (isinstance(self.constraints, str) or self.constraints is None or callable(self.constraints))
            else list(self.constraints)
        )

        # Validate each constraint
        for i, constraint in enumerate(self.constraints):
            is_callable = callable(constraint)
            is_valid_str = constraint in CONSTRAINTS
            is_None = constraint is None

            if not any((is_callable, is_valid_str, is_None)):
                msg = f"Constraint number {i} ({constraint}) not in {list(CONSTRAINTS.keys())}"
                raise ValueError(msg)

        return self

    @property
    def istensor(self):
        return isinstance(self, TensorTerm)

    @property
    def isintercept(self):
        return isinstance(self, Intercept)

    @property
    def info(self):
        """get information about this term

        Parameters
        ----------

        Returns
        -------
        dict containing information to duplicate this term
        """
        info = self.get_params()
        info.update({"term_type": self._name})
        return info

    @classmethod
    def build_from_info(cls, info):
        """build a Term instance from a dict

        Parameters
        ----------
        cls : class

        info : dict
            contains all information needed to build the term

        Return
        ------
        Term instance
        """
        info = deepcopy(info)
        if "term_type" in info:
            cls_ = TERMS[info.pop("term_type")]

            if issubclass(cls_, MetaTermMixin):
                return cls_.build_from_info(info)
        else:
            cls_ = cls
        return cls_(**info)

    @property
    def hasconstraint(self):
        """bool, whether the term has any constraints"""
        return np.not_equal(np.atleast_1d(self.constraints), None).any()

    @property
    @abstractproperty
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model"""
        pass

    @abstractmethod
    def compile(self, X, verbose=False):
        """method to validate and prepare data-dependent parameters

        Parameters
        ---------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        return self

    @abstractmethod
    def build_columns(self, X, verbose=False):
        """construct the model matrix columns for the term

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        pass

    def _determine_auto(self):
        """Map 'auto' penalty to appropriate penalty type."""
        if self.dtype == "numerical":
            if self._name == "spline_term":
                if self.basis in ["cp"]:
                    penalty = "periodic"
                else:
                    penalty = "derivative"
            else:
                penalty = "l2"
        if self.dtype == "categorical":
            penalty = "l2"

        return penalty

    def build_penalties(self, verbose=False):
        """
        builds the GAM block-diagonal penalty matrix in quadratic form
        out of penalty matrices specified for each feature.

        each feature penalty matrix is multiplied by a lambda for that feature.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]


        Parameters
        ---------
        None

        Returns
        -------
        P : sparse CSC matrix containing the model penalties in quadratic form
        """

        logger.debug(f"Building penalty matrix for term: {self.get_params()}")
        logger.debug(f"Number of coefficients in term: {self.n_coefs}")

        if self.isintercept:
            return np.array([[0.0]])

        penalty_matrices = []
        for penalty, lam in zip(self.penalties, self.lam):
            if penalty == "auto":
                penalty = self._determine_auto()
            elif penalty is None:
                penalty = "none"

            if penalty in PENALTIES:
                penalty = PENALTIES[penalty]

            if not callable(penalty):
                raise TypeError(f"'penalty must be callable. Found: {penalty}'")

            penalty_matrix = penalty(self.n_coefs)  # penalties dont need coef

            # Multiply by penalty, we want the square root of the quaddratic form
            # lam coef^T P^T P coef
            # so we create sqrt(lam) P here
            penalty_matrix = penalty_matrix * np.sqrt(lam)
            # penalty_matrix = penalty_matrix + np.eye(self.n_coefs) * 1e-1

            penalty_matrices.append(penalty_matrix)

        # Sum to a single (n_coefs, n_coefs) matrix
        penalty_matrix = functools.reduce(np.add, penalty_matrices)
        assert isinstance(penalty_matrix, np.ndarray)
        assert penalty_matrix.shape == (self.n_coefs, self.n_coefs)
        return penalty_matrix

    def build_constraints(self, coef, constraint_lam):
        """
        builds the GAM block-diagonal constraint matrix in quadratic form
        out of constraint matrices specified for each feature.

        behaves like a penalty, but with a very large lambda value, ie 1e6.

        Parameters
        ---------
        coefs : array-like containing the coefficients of a term

        constraint_lam : float,
            penalty to impose on the constraint.

            typically this is a very large number.

        Returns
        -------
        C : sparse CSC matrix containing the model constraints in quadratic form
        """
        if self.isintercept:
            return np.array([[0.0]])

        constraint_matrices = []
        for constraint in self.constraints:
            if constraint is None:
                constraint = "none"
            if constraint in CONSTRAINTS:
                constraint = CONSTRAINTS[constraint]

            if not callable(constraint):
                raise TypeError(f"'constraint must be callable. Found: {constraint}'")

            C = constraint(coef) * np.sqrt(constraint_lam)
            constraint_matrices.append(C)

        # Sum to a single (n_coefs, n_coefs) matrix
        constraint_matrix = np.sum(constraint_matrices, axis=0)
        assert isinstance(constraint_matrix, np.ndarray)
        assert constraint_matrix.shape == (self.n_coefs, self.n_coefs)
        return constraint_matrix


class Intercept(Term):
    def __init__(self, verbose=False):
        """creates an instance of an Intercept term

        Parameters
        ----------

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self._name = "intercept_term"
        self._minimal_name = "intercept"

        super().__init__(
            feature=None,
            fit_linear=False,
            fit_splines=False,
            lam=None,
            penalties=None,
            constraints=None,
            verbose=verbose,
        )

        self._exclude += ["fit_splines", "fit_linear", "lam", "penalties", "constraints", "feature", "dtype"]

    def __repr__(self):
        return self._minimal_name

    def _validate_arguments(self):
        """method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        return self

    @property
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model"""
        return 1

    def compile(self, X, verbose=False):
        """method to validate and prepare data-dependent parameters

        Parameters
        ---------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        return self

    def build_columns(self, X, verbose=False):
        """construct the model matrix columns for the term

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        return sp.sparse.csc_matrix(np.ones((len(X), 1)))


class LinearTerm(Term):
    def __init__(self, feature, lam=0.6, penalties="auto", verbose=False):
        """creates an instance of a LinearTerm

        Parameters
        ----------
        feature : int
            Index of the feature to use for the feature function.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self._name = "linear_term"
        self._minimal_name = "l"
        super().__init__(
            feature=feature,
            lam=lam,
            penalties=penalties,
            constraints=None,
            dtype="numerical",
            fit_linear=True,
            fit_splines=False,
            verbose=verbose,
        )
        self._exclude += ["fit_splines", "fit_linear", "dtype", "constraints"]

    @property
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model"""
        return 1

    def compile(self, X, verbose=False):
        """method to validate and prepare data-dependent parameters

        Parameters
        ---------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        if self.feature >= X.shape[1]:
            raise ValueError(
                "term requires feature {}, " "but X has only {} dimensions".format(self.feature, X.shape[1])
            )

        self.edge_knots_ = gen_edge_knots(X[:, self.feature], self.dtype)
        return self

    def build_columns(self, X, verbose=False):
        """construct the model matrix columns for the term

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        return sp.sparse.csc_matrix(X[:, self.feature][:, np.newaxis])


class SplineTerm(Term):
    _bases = ["ps", "cp"]

    def __init__(
        self,
        feature,
        n_splines=20,
        spline_order=3,
        lam=0.6,
        penalties="auto",
        constraints=None,
        dtype="numerical",
        basis="ps",
        by=None,
        edge_knots=None,
        verbose=False,
    ):
        """creates an instance of a SplineTerm

        Parameters
        ----------
        feature : int
            Index of the feature to use for the feature function.

        n_splines : int
            Number of splines to use for the feature function.
            Must be non-negative.

        spline_order : int
            Order of spline to use for the feature function.
            Must be non-negative.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        constraints : {None, 'convex', 'concave', 'monotonic_inc', 'monotonic_dec'}
            or callable or iterable

            Type of constraint to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.

        dtype : {'numerical', 'categorical'}
            String describing the data-type of the feature.

        basis : {'ps', 'cp'}
            Type of basis function to use in the term.

            'ps' : p-spline basis

            'cp' : cyclic p-spline basis, useful for building periodic functions.
                   by default, the maximum and minimum of the feature values
                   are used to determine the function's period.

                   to specify a custom period use argument `edge_knots`

        edge_knots : optional, array-like of floats of length 2

            these values specify minimum and maximum domain of the spline function.

            in the case that `spline_basis="cp"`, `edge_knots` determines
            the period of the cyclic function.

            when `edge_knots=None` these values are inferred from the data.

            default: None

        by : int, optional
            Feature to use as a by-variable in the term.

            For example, if `feature` = 2 `by` = 0, then the term will produce:
            x0 * f(x2)

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self.basis = basis
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.by = by
        self._name = "spline_term"
        self._minimal_name = "s"

        if edge_knots is not None:
            self.edge_knots_ = edge_knots

        super().__init__(
            feature=feature,
            lam=lam,
            penalties=penalties,
            constraints=constraints,
            fit_linear=False,
            fit_splines=True,
            dtype=dtype,
            verbose=verbose,
        )

        self._exclude += ["fit_linear", "fit_splines"]

    def _validate_arguments(self):
        """method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        super()._validate_arguments()

        if self.basis not in self._bases:
            raise ValueError(f"basis must be one of {self._bases}, but found: {self.basis}")

        # n_splines
        if not isinstance(self.n_splines, numbers.Integral):
            raise TypeError("Argument 'n_splines' must be an integer")

        if self.n_splines < 0:
            raise ValueError("Argument 'n_splines' must be >= 0")

        # spline_order
        self.spline_order = check_param(self.spline_order, param_name="spline_order", dtype="int", constraint=">= 0")

        # by
        if self.by is not None:
            self.by = check_param(self.by, param_name="by", dtype="int", constraint=">= 0")

        return self

    @property
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model"""
        return self.n_splines

    def compile(self, X, verbose=False):
        """method to validate and prepare data-dependent parameters

        Parameters
        ---------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        if self.feature >= X.shape[1]:
            raise ValueError(
                "term requires feature {}, " "but X has only {} dimensions".format(self.feature, X.shape[1])
            )

        if self.by is not None and self.by >= X.shape[1]:
            raise ValueError(
                "by variable requires feature {}, " "but X has only {} dimensions".format(self.by, X.shape[1])
            )

        if not hasattr(self, "edge_knots_"):
            self.edge_knots_ = gen_edge_knots(X[:, self.feature], self.dtype)
        return self

    def build_columns(self, X, verbose=False):
        """construct the model matrix columns for the term

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        # X[:, self.feature][:, np.newaxis]

        splines = b_spline_basis(
            X[:, self.feature],
            edge_knots=self.edge_knots_,
            spline_order=self.spline_order,
            n_splines=self.n_splines,
            sparse=True,
            periodic=self.basis in ["cp"],
        )

        if self.by is not None:
            splines = splines.multiply(X[:, self.by][:, np.newaxis])

        return splines


class FactorTerm(SplineTerm):
    _encodings = ["one-hot", "dummy"]

    def __init__(self, feature, lam=0.6, penalties="auto", coding="one-hot", verbose=False):
        """creates an instance of a FactorTerm

        Parameters
        ----------
        feature : int
            Index of the feature to use for the feature function.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        coding : {'one-hot'} type of contrast encoding to use.
            currently, only 'one-hot' encoding has been developed.
            this means that we fit one coefficient per category.

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self.coding = coding
        super().__init__(
            feature=feature,
            lam=lam,
            dtype="categorical",
            spline_order=0,
            penalties=penalties,
            by=None,
            constraints=None,
            verbose=verbose,
        )
        self._name = "factor_term"
        self._minimal_name = "f"
        self._exclude += ["dtype", "spline_order", "by", "n_splines", "basis", "constraints"]

    def _validate_arguments(self):
        """method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        super()._validate_arguments()
        if self.coding not in self._encodings:
            msg = f"coding must be one of {self._encodings}, but found: {self.coding}"
            raise ValueError(msg)

        return self

    def compile(self, X, verbose=False):
        """method to validate and prepare data-dependent parameters

        Parameters
        ---------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        super().compile(X)

        self.n_splines = len(np.unique(X[:, self.feature]))
        self.edge_knots_ = gen_edge_knots(X[:, self.feature], self.dtype)
        return self

    def build_columns(self, X, verbose=False):
        """construct the model matrix columns for the term

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        columns = super().build_columns(X, verbose=verbose)
        if self.coding == "dummy":
            columns = columns[:, 1:]

        return columns

    @property
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model"""
        return self.n_splines - 1 * (self.coding in ["dummy"])


class MetaTermMixin:
    _plural = [
        "feature",
        "dtype",
        "fit_linear",
        "fit_splines",
        "lam",
        "n_splines",
        "spline_order",
        "constraints",
        "penalties",
        "basis",
        "edge_knots_",
    ]
    _term_location = "_terms"

    def _super_get(self, name):
        return super().__getattribute__(name)

    def _super_has(self, name):
        try:
            self._super_get(name)
            return True
        except AttributeError:
            return False

    def _has_terms(self):
        """bool, whether the instance has any sub-terms"""
        loc = self._super_get("_term_location")
        return (
            self._super_has(loc)
            and isiterable(self._super_get(loc))
            and len(self._super_get(loc)) > 0
            and all([isinstance(term, Term) for term in self._super_get(loc)])
        )

    def _get_terms(self):
        """get the terms in the instance

        Parameters
        ----------
        None

        Returns
        -------
        list containing terms
        """
        if self._has_terms():
            return getattr(self, self._term_location)

    def __setattr__(self, name, value):
        if self._has_terms() and name in self._super_get("_plural"):
            # get the total number of arguments
            size = np.atleast_1d(flatten(getattr(self, name))).size

            # check shapes
            if isiterable(value):
                value = flatten(value)
                if len(value) != size:
                    raise ValueError("Expected {} to have length {}, but found {} = {}".format(name, size, name, value))
            else:
                value = [value] * size

            # now set each term's sequence of arguments
            for term in self._get_terms()[::-1]:
                # skip intercept
                if term.isintercept:
                    continue

                # how many values does this term get?
                n = np.atleast_1d(getattr(term, name)).size

                # get the next n values and set them on this term
                vals = [value.pop() for _ in range(n)][::-1]
                setattr(term, name, vals[0] if n == 1 else vals)

                term._validate_arguments()

            return
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if self._has_terms() and name in self._super_get("_plural"):
            # collect value from each term
            values = []
            for term in self._get_terms():
                # skip the intercept
                if term.isintercept:
                    continue

                values.append(getattr(term, name, None))
            return values

        return self._super_get(name)


class TensorTerm(SplineTerm, MetaTermMixin):
    _N_SPLINES = 10  # default num splines

    def __init__(self, *args, **kwargs):
        """creates an instance of a TensorTerm

        This is useful for creating interactions between features, or other terms.

        Parameters
        ----------
        *args : marginal Terms to combine into a tensor product

        feature : list of integers
            Indices of the features to use for the marginal terms.

        n_splines : list of integers
            Number of splines to use for each marginal term.
            Must be of same length as `feature`.

        spline_order : list of integers
            Order of spline to use for the feature function.
            Must be of same length as `feature`.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        constraints : {None, 'convex', 'concave', 'monotonic_inc', 'monotonic_dec'}
            or callable or iterable

            Type of constraint to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.

        dtype : list of {'numerical', 'categorical'}
            String describing the data-type of the feature.

            Must be of same length as `feature`.

        basis : list of {'ps'}
            Type of basis function to use in the term.

            'ps' : p-spline basis

            NotImplemented:
            'cp' : cyclic p-spline basis

            Must be of same length as `feature`.

        by : int, optional
            Feature to use as a by-variable in the term.

            For example, if `feature` = [1, 2] `by` = 0, then the term will produce:
            x0 * te(x1, x2)

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self.verbose = kwargs.pop("verbose", False)
        by = kwargs.pop("by", None)
        terms = self._parse_terms(args, **kwargs)

        feature = [term.feature for term in terms]
        super().__init__(feature, by=by, verbose=self.verbose)

        self._name = "tensor_term"
        self._minimal_name = "te"

        self._exclude = [
            "feature",
            "dtype",
            "fit_linear",
            "fit_splines",
            "lam",
            "n_splines",
            "spline_order",
            "constraints",
            "penalties",
            "basis",
        ]
        for param in self._exclude:
            delattr(self, param)

        self._terms = terms

    def _parse_terms(self, args, **kwargs):
        m = len(args)
        if m < 2:
            raise ValueError("TensorTerm requires at least 2 marginal terms")

        for k, v in kwargs.items():
            if isiterable(v):
                if len(v) != m:
                    msg = f"Expected {k} to have length {m}, but found {k} = {v}"
                    raise ValueError(msg)
            else:
                kwargs[k] = [v] * m

        terms = []
        for i, arg in enumerate(np.atleast_1d(args)):
            if isinstance(arg, type(self)):
                msg = "TensorTerm does not accept other TensorTerms.\n"
                msg += "Please build a flat TensorTerm instead of a nested one."
                raise ValueError(msg)

            if isinstance(arg, Term):
                if self.verbose and kwargs:
                    warnings.warn("kwargs are skipped when Term instances are passed to TensorTerm constructor")
                terms.append(arg)
                continue

            kwargs_ = {"n_splines": self._N_SPLINES}
            kwargs_.update({k: v[i] for k, v in kwargs.items()})

            terms.append(SplineTerm(arg, **kwargs_))

        return terms

    def __len__(self):
        return len(self._terms)

    def __getitem__(self, i):
        return self._terms[i]

    def _validate_arguments(self):
        """method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        if self._has_terms():
            [term._validate_arguments() for term in self._terms]
        else:
            super()._validate_arguments()

        return self

    @property
    def info(self):
        """get information about this term

        Parameters
        ----------

        Returns
        -------
        dict containing information to duplicate this term
        """
        info = super().info
        info.update({"terms": [term.info for term in self._terms]})
        return info

    @classmethod
    def build_from_info(cls, info):
        """build a TensorTerm instance from a dict

        Parameters
        ----------
        cls : class

        info : dict
            contains all information needed to build the term

        Return
        ------
        TensorTerm instance
        """
        terms = []
        for term_info in info["terms"]:
            terms.append(SplineTerm.build_from_info(term_info))
        return cls(*terms)

    @property
    def hasconstraint(self):
        """bool, whether the term has any constraints"""
        return any(term.hasconstraint for term in self._terms)

    @property
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model"""
        return np.prod([term.n_coefs for term in self._terms])

    def compile(self, X, verbose=False):
        """method to validate and prepare data-dependent parameters

        Parameters
        ---------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        for term in self._terms:
            term.compile(X, verbose=False)

        if self.by is not None and self.by >= X.shape[1]:
            msg = f"by variable requires feature {self.by}, but X has only {X.shape[1]} dimensions"
            raise ValueError(msg)
        return self

    def build_columns(self, X, verbose=False):
        """construct the model matrix columns for the term

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        columns_list = [term.build_columns(X, verbose=verbose) for term in self._terms]
        splines = functools.reduce(tensor_product, columns_list)

        if self.by is not None:
            splines *= X[:, self.by][:, np.newaxis]

        return sp.sparse.csc_matrix(splines)

    def build_penalties(self):
        """
        builds the GAM block-diagonal penalty matrix in quadratic form
        out of penalty matrices specified for each feature.

        each feature penalty matrix is multiplied by a lambda for that feature.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]

        Parameters
        ----------
        None

        Returns
        -------
        P : sparse CSC matrix containing the model penalties in quadratic form

        Examples
        --------
        The coefficients are imagined to be structured as
        [[b_11, b_12, b_13, b14],
         [b_21, b_22, b_23, b24],
         [b_31, b_32, b_33, b34]]
        and .ravel()'ed into a vector of
        [b_11, b_12, b_13, b_14, b_21, b_22, ...]
        The example below shows a penalty matrix:

        >>> spline1 = s(0, n_splines=3, lam=1)
        >>> spline2 = s(1, n_splines=4, lam=1)
        >>> te(spline1, spline2).build_penalties().astype(int)
        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 1,  0,  0,  0, -2,  0,  0,  0,  1,  0,  0,  0],
               [ 0,  1,  0,  0,  1, -4,  1,  0,  0,  1,  0,  0],
               [ 0,  0,  1,  0,  0,  1, -4,  1,  0,  0,  1,  0],
               [ 0,  0,  0,  1,  0,  0,  0, -2,  0,  0,  0,  1],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        """

        marginal_penalty_matrices = [self._build_marginal_penalties(i) for i, _ in enumerate(self._terms)]

        return functools.reduce(np.add, marginal_penalty_matrices)

    def _build_marginal_penalties(self, i):
        """

        Examples
        --------
        >>> spline1 = s(0, n_splines=3, lam=1)
        >>> spline2 = s(1, n_splines=4, lam=1)
        >>> te(spline1, spline2)._build_marginal_penalties(0).astype(int)
        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 1,  0,  0,  0, -2,  0,  0,  0,  1,  0,  0,  0],
               [ 0,  1,  0,  0,  0, -2,  0,  0,  0,  1,  0,  0],
               [ 0,  0,  1,  0,  0,  0, -2,  0,  0,  0,  1,  0],
               [ 0,  0,  0,  1,  0,  0,  0, -2,  0,  0,  0,  1],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        >>> te(spline1, spline2)._build_marginal_penalties(1).astype(int)
        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  1, -2,  1,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  1, -2,  1,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        """

        # i = 0 -> sp.sparse.kron(term.build_penalties(), sparse.eye, sparse.eye)
        # i = 1 -> sp.sparse.kron(sparse.eye, term.build_penalties(), sparse.eye)
        # i = 1 -> sp.sparse.kron(sparse.eye, sparse.eye, term.build_penalties())

        penalty_matrices = [
            (term.build_penalties() if i == j else np.eye(term.n_coefs)) for j, term in enumerate(self._terms)
        ]
        return functools.reduce(sp.linalg.kron, penalty_matrices)

    def build_constraints(self, coef, constraint_lam):
        """
        builds the GAM block-diagonal constraint matrix in quadratic form
        out of constraint matrices specified for each feature.

        Parameters
        ----------
        coefs : array-like containing the coefficients of a term

        constraint_lam : float,
            penalty to impose on the constraint.

            typically this is a very large number.

        Returns
        -------
        C : sparse CSC matrix containing the model constraints in quadratic form

        Examples
        --------
        >>> spline1 = s(0, n_splines=3, lam=1)
        >>> spline2 = s(1, n_splines=4, lam=1, constraints="monotonic_inc")
        >>> coef = np.array([0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11])
        >>> tensor = te(spline1, spline2)
        >>> tensor.build_constraints(-coef, 1).astype(int)
        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [-1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1]])

        """
        marginal_constraints = [
            self._build_marginal_constraints(i, coef, constraint_lam) for i, _ in enumerate(self._terms)
        ]

        return functools.reduce(np.add, marginal_constraints)

    def _build_marginal_constraints(self, i, coef, constraint_lam):
        """builds a constraint matrix for a marginal term in the tensor term

        takes a tensor's coef vector, and slices it into pieces corresponding
        to term i, then builds a constraint matrix for each piece of the coef vector,
        and assembles them into a composite constraint matrix

        Parameters
        ----------
        i : int,
            index of the marginal term for which to build a constraint matrix

        coefs : array-like containing the coefficients of the tensor term

        constraint_lam : float,
            penalty to impose on the constraint.

            typically this is a very large number.

        Returns
        -------
        C : sparse CSC matrix containing the model constraints in quadratic form

        Examples
        --------
        >>> spline1 = s(0, n_splines=3, lam=1)
        >>> spline2 = s(1, n_splines=4, lam=1, constraints="monotonic_inc")
        >>> coef = np.array([0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11])
        >>> tensor = te(spline1, spline2)
        >>> spline2.build_constraints(np.array([5, 4, 3, 4]), 1).astype(int)
        array([[ 0,  0,  0,  0],
               [-1,  1,  0,  0],
               [ 0, -1,  1,  0],
               [ 0,  0,  0,  0]])
        >>> tensor._build_marginal_constraints(1, -coef, 1).astype(int)
        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [-1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1]])
        """

        composite_C = np.zeros((len(coef), len(coef)))

        for slice_ in self._iterate_marginal_coef_slices(i):
            # get the slice of coefficient vector
            coef_slice = coef[slice_]

            # build the constraint matrix for that slice
            slice_C = self._terms[i].build_constraints(coef_slice, constraint_lam)

            # now enter it into the composite
            composite_C[tuple(np.meshgrid(slice_, slice_))] = slice_C.T

        return composite_C

    def _iterate_marginal_coef_slices(self, i):
        """iterator of indices into tensor's coef vector for marginal term i's coefs

        takes a tensor_term and returns an iterator of indices
        that chop up the tensor's coef vector into slices belonging to term i

        Parameters
        ----------
        i : int,
            index of marginal term

        Yields
        ------
        np.ndarray of ints

        >>> spline = s(0, n_splines=3, lam=1)
        >>> spline2 = s(0, n_splines=4, lam=1)
        >>> for slice in te(spline, spline2)._iterate_marginal_coef_slices(0):
        ...     print(slice)
        [0 4 8]
        [1 5 9]
        [ 2  6 10]
        [ 3  7 11]
        >>> for slice in te(spline, spline2)._iterate_marginal_coef_slices(1):
        ...     print(slice)
        [0 1 2 3]
        [4 5 6 7]
        [ 8  9 10 11]
        """
        # Example: dims = [2, 3, 4]
        dims = [term_.n_coefs for term_ in self]

        # make all linear indices
        # Example: array([0, 1, 2, ... , 22, 23])
        idxs = np.arange(np.prod(dims))

        # reshape indices to a Nd matrix
        # Example has shape (2, 3, 4) and entries
        # array([[[ 0,  1,  2,  3],
        #         [ 4,  5,  6,  7],
        #         [ 8,  9, 10, 11]],

        #        [[12, 13, 14, 15],
        #         [16, 17, 18, 19],
        #         [20, 21, 22, 23]]])
        idxs = idxs.reshape(dims)

        # reshape to a 2d matrix, where we can loop over rows
        # Example (with i=0)
        # array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
        #        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
        idxs = np.moveaxis(idxs, i, 0).reshape(idxs.shape[i], idxs.size // idxs.shape[i])

        # loop over rows
        # Example yields:
        # -> [ 0 12]
        # -> [ 1 13]
        # ...
        # -> [11 23]
        for slice_ in idxs.T:
            yield slice_


class TermList(Core, MetaTermMixin):
    _terms = []

    def __init__(self, *terms, **kwargs):
        """creates an instance of a TermList

        If duplicate terms are supplied, only the first instance will be kept.

        Parameters
        ----------
        *terms : list of terms to

        verbose : bool
            whether to show warnings

        Attributes
        ----------
        n_coefs : int
            Total number of coefficients in the model

        hasconstraint : bool
            whether the model has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term list
        """
        super().__init__()
        self.verbose = kwargs.pop("verbose", False)

        if bool(kwargs):
            raise ValueError("Unexpected keyword argument {}".format(kwargs.keys()))

        def deduplicate(term, term_list, uniques_dict):
            """adds a term to the term_list only if it is new

            Parameters
            ----------
            term : Term
                new term in consideration

            term_list : list
                contains all unique terms

            uniques_dict : defaultdict
                keys are term info,
                values are bool: True if the term has been seen already

            Returns
            -------
            term_list : list
                contains `term` if it was unique
            """
            key = str(sorted(term.info.items()))
            if not uniques_dict[key]:
                uniques_dict[key] = True
                term_list.append(term)
            else:
                if self.verbose:
                    warnings.warn("skipping duplicate term: {}".format(repr(term)))
            return term_list

        # process terms
        uniques = defaultdict(bool)
        term_list = []
        for term in terms:
            if isinstance(term, Term):
                term_list = deduplicate(term, term_list, uniques)
            elif isinstance(term, TermList):
                for term_ in term._terms:
                    term_list = deduplicate(term_, term_list, uniques)
            else:
                msg = f"Invalid object added to TermList: {term}"
                raise TypeError(msg)

        self._terms = self._terms + term_list
        self._exclude = [
            "feature",
            "dtype",
            "fit_linear",
            "fit_splines",
            "lam",
            "n_splines",
            "spline_order",
            "constraints",
            "penalties",
            "basis",
        ]
        self.verbose = any([term.verbose for term in self._terms]) or self.verbose

    def __eq__(self, other):
        if isinstance(other, TermList):
            return self.info == other.info
        return False

    def __repr__(self):
        return " + ".join(repr(term) for term in self)

    def __len__(self):
        return len(self._terms)

    def __getitem__(self, i):
        return self._terms[i]

    def __radd__(self, other):
        return TermList(other, self)

    def __add__(self, other):
        return TermList(self, other)

    def __mul__(self, other):
        raise NotImplementedError()

    def _validate_arguments(self):
        """method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        if self._has_terms():
            [term._validate_arguments() for term in self._terms]
        return self

    @property
    def info(self):
        """get information about the terms in the term list

        Parameters
        ----------

        Returns
        -------
        dict containing information to duplicate the term list
        """
        info = {"term_type": "term_list", "verbose": self.verbose}
        info.update({"terms": [term.info for term in self._terms]})
        return info

    @classmethod
    def build_from_info(cls, info):
        """build a TermList instance from a dict

        Parameters
        ----------
        cls : class

        info : dict
            contains all information needed to build the term

        Return
        ------
        TermList instance
        """
        info = deepcopy(info)
        terms = []
        for term_info in info["terms"]:
            terms.append(Term.build_from_info(term_info))
        return cls(*terms)

    def compile(self, X, verbose=False):
        """method to validate and prepare data-dependent parameters

        Parameters
        ---------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        for term in self._terms:
            term.compile(X, verbose=verbose)

        # TODO: remove duplicate intercepts
        return self

    def pop(self, index=-1):
        return self._terms.pop(index)

    @property
    def hasconstraint(self):
        """bool, whether the term has any constraints"""
        return any(term.hasconstraint for term in self._terms)

    @property
    def n_coefs(self):
        """Total number of coefficients contributed by the terms in the model"""
        return sum(term.n_coefs for term in self._terms)

    def get_coef_indices(self, i=-1):
        """get the indices for the coefficients of a term in the term list

        Parameters
        ---------
        i : int
            by default `int=-1`, meaning that coefficient indices are returned
            for all terms in the term list

        Returns
        -------
        list of integers
        """
        if i == -1:
            return list(range(self.n_coefs))

        if i >= len(self._terms):
            raise ValueError("requested {}th term, but found only {} terms".format(i, len(self._terms)))

        start = 0
        for term in self._terms[:i]:
            start += term.n_coefs
        stop = start + self._terms[i].n_coefs
        return list(range(start, stop))

    def build_columns(self, X, term=-1, verbose=False):
        """construct the model matrix columns for the term

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        if term == -1:
            term = range(len(self._terms))
        term = list(np.atleast_1d(term))

        columns = []
        for term_id in term:
            columns.append(self._terms[term_id].build_columns(X, verbose=verbose))
        return sp.sparse.hstack(columns, format="csc")

    def build_penalties(self):
        """
        builds the GAM block-diagonal penalty matrix in quadratic form
        out of penalty matrices specified for each feature.

        each feature penalty matrix is multiplied by a lambda for that feature.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]


        Parameters
        ----------
        None

        Returns
        -------
        P : sparse CSC matrix containing the model penalties in quadratic form
        """
        return sp.sparse.block_diag([term.build_penalties() for term in self._terms])

    def build_constraints(self, coefs, constraint_lam):
        """
        builds the GAM block-diagonal constraint matrix in quadratic form
        out of constraint matrices specified for each feature.

        behaves like a penalty, but with a very large lambda value, ie 1e6.

        Parameters
        ---------
        coefs : array-like containing the coefficients of a term

        constraint_lam : float,
            penalty to impose on the constraint.

            typically this is a very large number.

        Returns
        -------
        C : sparse CSC matrix containing the model constraints in quadratic form
        """
        C = []
        for i, term in enumerate(self._terms):
            idxs = self.get_coef_indices(i=i)
            C.append(term.build_constraints(coefs[idxs], constraint_lam))

        block_matrix = sp.sparse.block_diag(C)
        assert block_matrix.shape[0] == block_matrix.shape[1]
        return block_matrix


# Minimal representations
def l(feature, lam=0.6, penalties="auto", verbose=False):
    """

    See Also
    --------
    LinearTerm : for developer details
    """
    return LinearTerm(feature=feature, lam=lam, penalties=penalties, verbose=verbose)


def s(
    feature,
    n_splines=20,
    spline_order=3,
    lam=0.6,
    penalties="auto",
    constraints=None,
    dtype="numerical",
    basis="ps",
    by=None,
    edge_knots=None,
    verbose=False,
):
    """

    See Also
    --------
    SplineTerm : for developer details
    """
    return SplineTerm(
        feature=feature,
        n_splines=n_splines,
        spline_order=spline_order,
        lam=lam,
        penalties=penalties,
        constraints=constraints,
        dtype=dtype,
        basis=basis,
        by=by,
        edge_knots=edge_knots,
        verbose=verbose,
    )


def f(feature, lam=0.6, penalties="auto", coding="one-hot", verbose=False):
    """

    See Also
    --------
    FactorTerm : for developer details
    """
    return FactorTerm(feature=feature, lam=lam, penalties=penalties, coding=coding, verbose=verbose)


def te(*args, **kwargs):
    """

    See Also
    --------
    TensorTerm : for developer details
    """
    return TensorTerm(*args, **kwargs)


intercept = Intercept()

# copy docs
for minimal_, class_ in zip([l, s, f, te], [LinearTerm, SplineTerm, FactorTerm, TensorTerm]):
    minimal_.__doc__ = class_.__init__.__doc__ + minimal_.__doc__


TERMS = {
    "term": Term,
    "intercept_term": Intercept,
    "linear_term": LinearTerm,
    "spline_term": SplineTerm,
    "factor_term": FactorTerm,
    "tensor_term": TensorTerm,
    "term_list": TermList,
}


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
            "--capture=sys",
            "--doctest-modules",
        ]
    )

    spline = s(0, n_splines=3, lam=1)
    spline2 = s(0, n_splines=4, lam=1)

    for slice in te(spline, spline2)._iterate_marginal_coef_slices(0):
        print(slice)
