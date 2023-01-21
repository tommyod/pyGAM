"""
Penalty matrix generators
"""

import warnings

import numpy as np
import scipy as sp
import functools


class FDMatrix:
    """Finite differene matrices."""

    @staticmethod
    def forward_diff(n, periodic=True):
        """Create a forward difference matrix D such that f' = D f.

        Parameters
        ----------
        n : int
            Size of matrix.
        periodic : bool, optional
            Whether to wrap around the boundary. The default is True.

        Returns
        -------
        np.ndarray
            Forward difference matrix.

        Examples
        --------
        >>> FDMatrix.forward_diff(4, periodic=True)
        array([[-1.,  1.,  0.,  0.],
               [ 0., -1.,  1.,  0.],
               [ 0.,  0., -1.,  1.],
               [ 1.,  0.,  0., -1.]])
        >>> FDMatrix.forward_diff(4, periodic=False)
        array([[-1.,  1.,  0.,  0.],
               [ 0., -1.,  1.,  0.],
               [ 0.,  0., -1.,  1.],
               [ 0.,  0.,  0., -1.]])
        """
        if periodic:
            return sp.linalg.circulant([-1] + [0] * (n - 2) + [1]).astype(float)
        else:
            return (np.eye(n, k=1) - np.eye(n)).astype(float)

    @staticmethod
    def backward_diff(n, periodic=True):
        """Create a backward difference matrix D such that f' = D f.

        Parameters
        ----------
        n : int
            Size of matrix.
        periodic : bool, optional
            Whether to wrap around the boundary. The default is True.

        Returns
        -------
        np.ndarray
            Backward difference matrix.

        Examples
        --------
        >>> FDMatrix.backward_diff(4, periodic=True)
        array([[ 1.,  0.,  0., -1.],
               [-1.,  1.,  0.,  0.],
               [ 0., -1.,  1.,  0.],
               [ 0.,  0., -1.,  1.]])
        >>> FDMatrix.backward_diff(4, periodic=False)
        array([[ 1.,  0.,  0.,  0.],
               [-1.,  1.,  0.,  0.],
               [ 0., -1.,  1.,  0.],
               [ 0.,  0., -1.,  1.]])
        """
        if periodic:
            return sp.linalg.circulant([1] + [0] * (n - 2) + [-1]).T.astype(float)
        else:
            return (np.eye(n) - np.eye(n, k=-1)).astype(float)

    @classmethod
    def centered_diff(cls, n, periodic=True):
        """Create a centered difference matrix D such that f' = D f.

        Parameters
        ----------
        n : int
            Size of matrix.
        periodic : bool, optional
            Whether to wrap around the boundary. The default is True.

        Returns
        -------
        np.ndarray
            Backward difference matrix.

        Examples
        --------
        >>> FDMatrix.centered_diff(6, True)
        array([[ 0. ,  0.5,  0. ,  0. ,  0. , -0.5],
               [-0.5,  0. ,  0.5,  0. ,  0. ,  0. ],
               [ 0. , -0.5,  0. ,  0.5,  0. ,  0. ],
               [ 0. ,  0. , -0.5,  0. ,  0.5,  0. ],
               [ 0. ,  0. ,  0. , -0.5,  0. ,  0.5],
               [ 0.5,  0. ,  0. ,  0. , -0.5,  0. ]])
        >>> FDMatrix.backward_diff(4, periodic=False)
        array([[ 1.,  0.,  0.,  0.],
               [-1.,  1.,  0.,  0.],
               [ 0., -1.,  1.,  0.],
               [ 0.,  0., -1.,  1.]])
        """
        if periodic:
            return (cls.forward_diff(n, periodic) + cls.backward_diff(n, periodic)) / 2.0
        else:
            D = (cls.forward_diff(n, periodic) + cls.backward_diff(n, periodic)) / 2.0
            D[0, :2] = [-1.0, 1.0]  # Use forward difference for first element
            D[-1, -2:] = [-1.0, 1.0]  # Use backward difference for last element
            return D
        
        
    @classmethod
    def derivative(cls, n, order=1, periodic=True):
        D = cls.centered_diff(n, periodic)
        return np.linalg.matrix_power(D, n=order)
        
        


# =============================================================================
# PENALTIES
# =============================================================================


def derivative(n, coef, derivative=2, periodic=False):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes the squared differences between basis coefficients.

    Parameters
    ----------
    n : int
        number of splines

    coef : unused
        for compatibility with constraints

    derivative: int, default: 2
        which derivative do we penalize.
        derivative is 1, we penalize 1st order derivatives,
        derivative is 2, we penalize 2nd order derivatives, etc

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    
    # return FDMatrix.derivative(n, order=derivative, periodic=periodic)

    
    if n == 1:
        # no derivative for constant functions
        return sp.sparse.csc_matrix(0.0)
    D = sparse_diff(sp.sparse.identity(n + 2 * derivative * periodic).tocsc(), n=derivative).tolil()

    if periodic:
        # wrap penalty
        cols = D[:, :derivative]
        D[:, -2 * derivative : -derivative] += cols * (-1) ** derivative

        # do symmetric operation on lower half of matrix
        n_rows = int((n + 2 * derivative) / 2)
        D[-n_rows:] = D[:n_rows][::-1, ::-1]

        # keep only the center of the augmented matrix
        D = D[derivative:-derivative, derivative:-derivative]
        
    # print(f"Shape of D: {D.shape}")
    return D.dot(D.T).tocsc()


def periodic(n, coef, derivative=2, _penalty=derivative):
    # return FDMatrix.derivative(n, order=derivative, periodic=True)
    
    return _penalty(n, coef, derivative=derivative, periodic=True)


def l2(n, coef):
    """
    Builds a penalty matrix for P-Splines with categorical features.
    Penalizes the squared value of each basis coefficient.

    Parameters
    ----------
    n : int
        number of splines

    coef : unused
        for compatibility with constraints

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return sp.sparse.eye(n).tocsc()


# =============================================================================
# CONSTRAINTS
# =============================================================================


def monotonic_inc(n, coef):    
    return monotonicity_(n, coef, increasing=True)

    coef= coef.ravel()
    # For an array   [1, 4,  3,  2, 5]
    # the penalty is [0, 0, -1, -1, 0]
    # and this encourages elements corresponding to 3 and 2 to grow
    # in the minimization problem
    D = FDMatrix.backward_diff(n, periodic=False)
    mask = np.diff(coef) > 0
    print(coef, coef.shape)
    mask = np.hstack(([True], mask))
    D[mask, :] = 0
    return D


def monotonic_dec(n, coef):
    return monotonicity_(n, coef, increasing=False)

    coef= coef.ravel()
    # For an array   [ 1, 4, 3,  2, 5]
    # the penalty is [-3, 0, 0, -3, 0]
    # and this encourages elements corresponding to 3 and 2 to grow
    # in the minimization problem
    
    # equals to -monotonic_inc(5, a[::-1]) @ a
    D = FDMatrix.forward_diff(n, periodic=False)
    mask = np.diff(coef) < 0
    mask = np.hstack((mask, [True]))
    D[mask, :] = 0
    return -D


def monotonicity_(n, coef, increasing=True):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of monotonicity in the feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like
        coefficients of the feature function
    increasing : bool, default: True
        whether to enforce monotic increasing, or decreasing functions
    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    
    if n != len(coef.ravel()):
        raise ValueError(
            "dimension mismatch: expected n equals len(coef), "
            "but found n = {}, coef.shape = {}.".format(n, coef.shape)
        )

    if n == 1:
        # no monotonic penalty for constant functions
        return sp.sparse.csc_matrix(0.0)

    if increasing:
        # only penalize the case where coef_i-1 > coef_i
        mask = sp.sparse.diags((np.diff(coef.ravel()) < 0).astype(float))
    else:
        # only penalize the case where coef_i-1 < coef_i
        mask = sp.sparse.diags((np.diff(coef.ravel()) > 0).astype(float))

    derivative = 1
    D = sparse_diff(sp.sparse.identity(n).tocsc(), n=derivative) * mask
    # print(f"Shape of D: {D.shape}")
    return D.dot(D.T).tocsc()


def convexity_(n, coef, convex=True):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of convexity in the feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like
        coefficients of the feature function
    convex : bool, default: True
        whether to enforce convex, or concave functions
    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    if n != len(coef.ravel()):
        raise ValueError(
            "dimension mismatch: expected n equals len(coef), "
            "but found n = {}, coef.shape = {}.".format(n, coef.shape)
        )

    if n == 1:
        # no convex penalty for constant functions
        return sp.sparse.csc_matrix(0.0)

    if convex:
        mask = sp.sparse.diags((np.diff(coef.ravel(), n=2) < 0).astype(float))
    else:
        mask = sp.sparse.diags((np.diff(coef.ravel(), n=2) > 0).astype(float))

    derivative = 2
    D = sparse_diff(sp.sparse.identity(n).tocsc(), n=derivative) * mask
    # print(f"Shape of D: {D.shape}")
    return D.dot(D.T).tocsc()


def convex(n, coef):
    return convexity_(n, coef, convex=True)


def concave(n, coef):
    return convexity_(n, coef, convex=False)


def none(n, coef):
    """
    Build a matrix of zeros for features that should go unpenalized

    Parameters
    ----------
    n : int
        number of splines
    coef : unused
        for compatibility with constraints

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return sp.sparse.csc_matrix((n, n))


# =============================================================================
# OTHER
# =============================================================================


def wrap_penalty(p, fit_linear, linear_penalty=0.0):
    """
    tool to account for unity penalty on the linear term of any feature.

    example:
        p = wrap_penalty(derivative, fit_linear=True)(n, coef)

    Parameters
    ----------
    p : callable.
        penalty-matrix-generating function.
    fit_linear : boolean.
        whether the current feature has a linear term or not.
    linear_penalty : float, default: 0.
        penalty on the linear term

    Returns
    -------
    wrapped_p : callable
      modified penalty-matrix-generating function
    """

    def wrapped_p(n, *args):
        if fit_linear:
            if n == 1:
                return sp.sparse.block_diag([linear_penalty], format="csc")
            return sp.sparse.block_diag([linear_penalty, p(n - 1, *args)], format="csc")

        return p(n, *args)

    return wrapped_p


def sparse_diff(array, n=1, axis=-1):
    """
    A ported sparse version of np.diff.
    Uses recursion to compute higher order differences

    Parameters
    ----------
    array : sparse array
    n : int, default: 1
        differencing order
    axis : int, default: -1
        axis along which differences are computed

    Returns
    -------
    diff_array : sparse array
                 same shape as input array,
                 but 'axis' dimension is smaller by 'n'.
    """
    if (n < 0) or (int(n) != n):
        raise ValueError("Expected order is non-negative integer, " "but found: {}".format(n))
    if not sp.sparse.issparse(array):
        warnings.warn("Array is not sparse. Consider using numpy.diff")

    if n == 0:
        return array

    nd = array.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    A = sparse_diff(array, n - 1, axis=axis)
    return A[slice1] - A[slice2]


PENALTIES = {"auto": "auto", "derivative": derivative, "l2": l2, "none": none, "periodic": periodic}

CONSTRAINTS = {
    "convex": convex,
    "concave": concave,
    "monotonic_inc": monotonic_inc,
    "monotonic_dec": monotonic_dec,
    "none": none,
}

if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules"])

    import time

    for penalty_name, penalty_func in PENALTIES.items():

        if not callable(penalty_func):
            continue

        st = time.perf_counter()
        penalty_func(1000, np.random.randn(1000))
        elapsed = round(time.perf_counter() - st, 8)
        print(f"{penalty_name} in {elapsed} seconds")

    for penalty_name, penalty_func in CONSTRAINTS.items():

        if not callable(penalty_func):
            continue

        st = time.perf_counter()
        penalty_func(1000, np.random.randn(1000))
        elapsed = round(time.perf_counter() - st, 8)
        print(f"{penalty_name} in {elapsed} seconds")

    x = np.linspace(0, 2 * np.pi, num=2**10, endpoint=False)
    y = np.sin(x)
    dx = x[1] - x[0]
    n = len(x)
    periodic = True
    D = FDMatrix.centered_diff(n, periodic=periodic) / dx

    assert np.allclose(np.cos(x), D @ y)
