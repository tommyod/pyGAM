"""
Pygam utilities
"""

import collections.abc
import numbers
import sys
import warnings

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import SplineTransformer
from sklearn.utils import check_array

from pygam.log import setup_custom_logger

logger = setup_custom_logger(__name__)


class NotPositiveDefiniteError(ValueError):
    """Exception class to raise if a matrix is not positive definite"""


class OptimizationError(ValueError):
    """Exception class to raise if PIRLS optimization fails"""


def make_2d(array, verbose=True):
    """
    tiny tool to expand 1D arrays the way i want

    Parameters
    ----------
    array : array-like

    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    np.array of with ndim = 2
    """
    array = np.asarray(array)
    if array.ndim < 2:
        msg = "Expected 2D input data array, but found {}D. " "Expanding to 2D.".format(array.ndim)
        if verbose:
            warnings.warn(msg)
        array = np.atleast_1d(array)[:, None]
    return array


def check_y(y, link, dist, min_samples=1, verbose=True):
    """
    tool to ensure that the targets:
    - are in the domain of the link function
    - are numerical
    - have at least min_samples
    - is finite

    Parameters
    ----------
    y : array-like
    link : Link object
    dist : Distribution object
    min_samples : int, default: 1
    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    y : array containing validated y-data
    """
    y = np.ravel(y)
    # assert y.ndim == 1

    y = check_array(y, ensure_2d=False, ensure_min_samples=min_samples, input_name="y data")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if np.any(np.isnan(link.link(y, dist))):
            raise ValueError(
                "y data is not in domain of {} link function. "
                "Expected domain: {}, but found {}".format(
                    link, link.get_domain(dist), [float("%.2f" % np.min(y)), float("%.2f" % np.max(y))]
                )
            )
    return y


def check_X(X, n_feats=None, min_samples=1, edge_knots=None, dtypes=None, features=None):
    """
    tool to ensure that X:
    - is 2 dimensional
    - contains float-compatible data-types
    - has at least min_samples
    - has n_feats
    - has categorical features in the right range
    - is finite

    Parameters
    ----------
    X : array-like
    n_feats : int. default: None
              represents number of features that X should have.
              not enforced if n_feats is None.
    min_samples : int, default: 1
    edge_knots : list of arrays, default: None
    dtypes : list of strings, default: None
    features : list of ints,
        which features are considered by the model

    Returns
    -------
    X : array with ndims == 2 containing validated X-data
    """

    # check all features are there
    if bool(features):
        features = flatten(features)
        max_feat = max(flatten(features))

        if n_feats is None:
            n_feats = max_feat

        n_feats = max(n_feats, max_feat)

    # basic diagnostics
    to_check = X.values if isinstance(X, (pd.Series, pd.DataFrame)) else X
    X = check_array(to_check, ensure_2d=False, ensure_min_samples=min_samples, input_name="X data")

    # check our categorical data has no new categories
    if (edge_knots is not None) and (dtypes is not None) and (features is not None):
        # get a flattened list of tuples
        edge_knots = flatten(edge_knots)[::-1]
        dtypes = flatten(dtypes)
        assert len(edge_knots) % 2 == 0  # sanity check

        # form pairs
        n = len(edge_knots) // 2
        edge_knots = [(edge_knots.pop(), edge_knots.pop()) for _ in range(n)]

        # check each categorical term
        for i, ek in enumerate(edge_knots):
            dt = dtypes[i]
            feature = features[i]
            x = X[:, feature]

            if dt == "categorical":
                min_ = ek[0]
                max_ = ek[-1]
                if (np.unique(x) < min_).any() or (np.unique(x) > max_).any():
                    min_ += 0.5
                    max_ -= 0.5
                    raise ValueError(
                        "X data is out of domain for categorical "
                        "feature {}. Expected data on [{}, {}], "
                        "but found data on [{}, {}]".format(i, min_, max_, x.min(), x.max())
                    )

    return X


def check_lengths(*arrays):
    """
    tool to ensure input and output data have the same number of samples

    Parameters
    ----------
    *arrays : iterable of arrays to be checked

    Returns
    -------
    None
    """
    lengths = [len(array) for array in arrays]
    if len(np.unique(lengths)) > 1:
        raise ValueError("Inconsistent data lengths: {}".format(lengths))


def check_param(param, param_name, dtype, constraint=None, iterable=True, max_depth=2):
    """
    checks the dtype of a parameter,
    and whether it satisfies a numerical contraint

    Parameters
    ---------
    param : object
    param_name : str, name of the parameter
    dtype : str, desired dtype of the parameter
    contraint : str, default: None
                numerical constraint of the parameter.
                if None, no constraint is enforced
    iterable : bool, default: True
               whether to allow iterable param
    max_depth : int, default: 2
                maximum nesting of the iterable.
                only used if iterable == True
    Returns
    -------
    list of validated and converted parameter(s)
    """
    msg = []
    msg.append(param_name + " must be " + dtype)
    if iterable:
        msg.append(" or nested iterable of depth " + str(max_depth) + " containing " + dtype + "s")

    msg.append(", but found " + param_name + " = {}".format(repr(param)))

    if constraint is not None:
        msg = (" " + constraint).join(msg)
    else:
        msg = "".join(msg)

    # check param is numerical
    try:
        param_dt = np.array(flatten(param))  # + np.zeros_like(flatten(param), dtype='int')
        # param_dt = np.array(param).astype(dtype)
    except (ValueError, TypeError):
        raise TypeError(msg)

    # check iterable
    if (not iterable) and isiterable(param):
        raise TypeError(msg)

    # check param is correct dtype
    if not (param_dt == np.array(flatten(param)).astype(float)).all():
        raise TypeError(msg)

    # check constraint
    if constraint is not None:
        if not (eval("np." + repr(param_dt) + constraint)).all():
            raise ValueError(msg)

    return param


def load_diagonal(cov, load=None):
    """Return the given square matrix with a small amount added to the diagonal
    to make it positive semi-definite.

    Examples
    --------
    >>> A = np.arange(9).reshape(3, 3)
    >>> A
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> load_diagonal(A, 10)
    array([[10,  1,  2],
           [ 3, 14,  5],
           [ 6,  7, 18]])
    """

    n, m = cov.shape
    assert n == m, f"matrix must be square, but found shape {cov.shape}"
    cov = cov.copy()

    if load is None:
        machine_epsilon = np.finfo(float).eps  # 2.220446049250313e-16
        load = np.sqrt(machine_epsilon)

    # Idea from : https://github.com/scikit-learn/scikit-learn/blob/f9637ee0ae118d73834d240c09d683f692918911/sklearn/linear_model/_ridge.py#L215
    # Copying a (999, 999) matrix takes 423 µs, adding np.eye(999) takes 1.56 ms
    cov.flat[:: n + 1] += load
    return cov


def round_to_n_decimal_places(array, n=3):
    """
    tool to keep round a float to n decimal places.

    n=3 by default

    Parameters
    ----------
    array : np.array
    n : int. number of decimal places to keep

    Returns
    -------
    array : rounded np.array
    """
    # check if in scientific notation
    if issubclass(array.__class__, float) and "%.e" % array == str(array):
        return array  # do nothing

    shape = np.shape(array)
    out = (np.atleast_1d(array) * 10**n).round().astype("int") / (10.0**n)
    return out.reshape(shape)


# Credit to Hugh Bothwell from http://stackoverflow.com/questions/5084743/how-to-print-pretty-string-output-in-python
class TablePrinter:
    "Print a list of dicts as a table"

    def __init__(self, fmt, sep=" ", ul=None):
        """
        @param fmt: list of tuple(heading, key, width)
                        heading: str, column label
                        key: dictionary key to value to print
                        width: int, column width in chars
        @param sep: string, separation between columns
        @param ul: string, character to underline column label, or None for no underlining
        """
        super().__init__()
        self.fmt = str(sep).join("{lb}{0}:{1}{rb}".format(key, width, lb="{", rb="}") for heading, key, width in fmt)
        self.head = {key: heading for heading, key, width in fmt}
        self.ul = {key: str(ul) * width for heading, key, width in fmt} if ul else None
        self.width = {key: width for heading, key, width in fmt}

    def row(self, data):
        if sys.version_info < (3,):
            return self.fmt.format(**{k: str(data.get(k, ""))[:w] for k, w in self.width.iteritems()})
        else:
            return self.fmt.format(**{k: str(data.get(k, ""))[:w] for k, w in self.width.items()})

    def __call__(self, dataList):
        _r = self.row
        res = [_r(data) for data in dataList]
        res.insert(0, _r(self.head))
        if self.ul:
            res.insert(1, _r(self.ul))
        return "\n".join(res)


def space_row(left, right, filler=" ", total_width=-1):
    """space the data in a row with optional filling

    Arguments
    ---------
    left : str, to be aligned left
    right : str, to be aligned right
    filler : str, default ' '.
        must be of length 1
    total_width : int, width of line.
        if negative number is specified,
        then that number of spaces is used between the left and right text

    Returns
    -------
    str
    """
    left = str(left)
    right = str(right)
    filler = str(filler)[:1]

    if total_width < 0:
        spacing = -total_width
    else:
        spacing = total_width - len(left) - len(right)

    return left + filler * spacing + right


def sig_code(p_value):
    """create a significance code in the style of R's lm

    Arguments
    ---------
    p_value : float on [0, 1]

    Returns
    -------
    str
    """
    assert 0 <= p_value <= 1, "p_value must be on [0, 1]"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return " "


def gen_edge_knots(data, dtype):
    """
    generate uniform knots from data including the edges of the data

    for discrete data, assumes k categories in [0, k-1] interval

    Parameters
    ----------
    data : array-like with one dimension
    dtype : str in {'categorical', 'numerical'}

    Returns
    -------
    np.array containing ordered knots
    """
    if dtype not in ["categorical", "numerical"]:
        raise ValueError("unsupported dtype: {}".format(dtype))
    if dtype == "categorical":
        return np.r_[np.min(data) - 0.5, np.max(data) + 0.5]
    else:
        knots = np.r_[np.min(data), np.max(data)]
        if knots[0] == knots[1]:
            warnings.warn(
                "Data contains constant feature. " "Consider removing and setting fit_intercept=True", stacklevel=2
            )
        return knots


def b_spline_basis(x, edge_knots, n_splines=20, spline_order=3, sparse=True, periodic=True):
    """
    tool to generate b-spline basis using vectorized De Boor recursion
    the basis functions extrapolate linearly past the end-knots.

    Parameters
    ----------
    x : array-like, with ndims == 1.
    edge_knots : array-like contaning locations of the 2 edge knots.
    n_splines : int. number of splines to generate. must be >= spline_order+1
                default: 20
    spline_order : int. order of spline basis to create
                   default: 3
    sparse : boolean. whether to return a sparse basis matrix or not.
             default: True
    periodic: bool, default: True
        whether to repeat basis functions (True) or linearly extrapolate (False).

    Returns
    -------
    basis : sparse csc matrix or array containing b-spline basis functions
            with shape (len(x), n_splines)
    """
    if np.ravel(x).ndim != 1:
        raise ValueError("Data must be 1-D, but found {}".format(np.ravel(x).ndim))

    if (n_splines < 1) or not isinstance(n_splines, numbers.Integral):
        raise ValueError("n_splines must be int >= 1")

    if (spline_order < 0) or not isinstance(spline_order, numbers.Integral):
        raise ValueError("spline_order must be int >= 1")

    # Use sklearn here
    # TODO: Support knots?
    include_bias = True
    transformer = SplineTransformer(
        n_knots=n_splines + 2 - include_bias + (0 if periodic else -spline_order),
        degree=spline_order,
        knots="uniform",
        extrapolation="periodic" if periodic else "linear",
        include_bias=include_bias,
        order="C",
    )

    # Fit on edges to extrapolate properly
    transformer.fit((np.array(edge_knots)).reshape(-1, 1))

    bases = transformer.transform(x.reshape(-1, 1))
    assert bases.shape == (len(x), n_splines)

    if sparse:
        return sp.sparse.csc_matrix(bases)

    return bases


def isiterable(obj, reject_string=True):
    """convenience tool to detect if something is iterable.
    in python3, strings count as iterables to we have the option to exclude them

    Parameters:
    -----------
    obj : object to analyse
    reject_string : bool, whether to ignore strings

    Returns:
    --------
    bool, if the object is itereable.
    """

    # iterable = hasattr(obj, "__len__")
    iterable = isinstance(obj, collections.abc.Sized)

    if reject_string:
        iterable = iterable and not isinstance(obj, str)

    return iterable


def flatten(iterable):
    """convenience tool to flatten any nested iterable

    Examples
    --------
    >>> flatten([[[],[4]],[[[5,[6,7, []]]]]])
    [4, 5, 6, 7]
    >>> flatten('hello')
    'hello'

    Parameters
    ----------
    iterable

    Returns
    -------
    flattened object
    """
    if isiterable(iterable):
        flat = []
        for item in list(iterable):
            item = flatten(item)
            if not isiterable(item):
                item = [item]
            flat += item
        return flat
    else:
        return iterable


def tensor_product(a, b, reshape=True):
    """
    compute the tensor protuct of two matrices a and b

    if a is (n, m_a), b is (n, m_b),
    then the result is
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise

    Parameters
    ---------
    a : array-like of shape (n, m_a)
    b : array-like of shape (n, m_b)

    reshape : bool, default True
        whether to reshape the result to be 2-dimensional ie
        (n, m_a * m_b)
        or return a 3-dimensional tensor ie
        (n, m_a, m_b)

    Returns
    -------
    dense np.ndarray of shape
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise

    Examples
    --------
    >>> A = np.eye(3, dtype=int)
    >>> B = np.arange(9).reshape(3, 3)
    >>> tensor_product(A, B)
    array([[0, 1, 2, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 3, 4, 5, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 6, 7, 8]])
    >>> A = np.diag([1, 2, 3])
    >>> A[0, :] = [1, 2, 3]
    >>> tensor_product(A, B, reshape=True)
    array([[ 0,  1,  2,  0,  2,  4,  0,  3,  6],
           [ 0,  0,  0,  6,  8, 10,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 18, 21, 24]])
    >>> tensor_product(A, B, reshape=False)
    array([[[ 0,  1,  2],
            [ 0,  2,  4],
            [ 0,  3,  6]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 6,  8, 10],
            [ 0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 0,  0,  0],
            [18, 21, 24]]])


    """
    assert a.ndim == 2, f"matrix a must be 2-dimensional, but found {a.ndim} dimensions"
    assert b.ndim == 2, f"matrix b must be 2-dimensional, but found {b.nim} dimensions"

    na, ma = a.shape
    nb, mb = b.shape

    if na != nb:
        raise ValueError("both arguments must have the same number of samples")

    if sp.sparse.issparse(a):
        a = a.A

    if sp.sparse.issparse(b):
        b = b.A

    tensor = a[..., :, None] * b[..., None, :]

    if reshape:
        return tensor.reshape(na, ma * mb)

    return tensor


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules"])

    A = np.diag([1, 2, 3])
    A[0, :] = [1, 2, 3]
    B = np.arange(9).reshape(3, 3)
    T = tensor_product(A, B, False)
    print(T)
