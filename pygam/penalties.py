"""
Penalty matrix generators
"""

import warnings
import functools

import numpy as np
import scipy as sp


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
               [ 0.,  0.,  0.,  0.]])
        """
        if periodic:
            return sp.linalg.circulant([-1] + [0] * (n - 2) + [1]).astype(float)
        else:
            D = (np.eye(n, k=1) - np.eye(n)).astype(float)
            D[-1, -1] = 0
            return D

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
        array([[ 0.,  0.,  0.,  0.],
               [-1.,  1.,  0.,  0.],
               [ 0., -1.,  1.,  0.],
               [ 0.,  0., -1.,  1.]])
        """
        if periodic:
            return sp.linalg.circulant([1] + [0] * (n - 2) + [-1]).T.astype(float)
        else:
            D = (np.eye(n) - np.eye(n, k=-1)).astype(float)
            D[0, 0] = 0
            return D

    @classmethod
    def centered_diff(cls, n, periodic=True):
        """Create a centered difference matrix D such that f' = D f.

        At the boundaries, forward and backward diffs are used.

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
        >>> FDMatrix.centered_diff(6, False)
        array([[-1. ,  1. ,  0. ,  0. ,  0. ,  0. ],
               [-0.5,  0. ,  0.5,  0. ,  0. ,  0. ],
               [ 0. , -0.5,  0. ,  0.5,  0. ,  0. ],
               [ 0. ,  0. , -0.5,  0. ,  0.5,  0. ],
               [ 0. ,  0. ,  0. , -0.5,  0. ,  0.5],
               [ 0. ,  0. ,  0. ,  0. , -1. ,  1. ]])
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


def derivative(n, derivative=2, periodic=False):
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
    # print(D.A)
    return D.A
    return D.dot(D.T).tocsc()


def periodic(n, derivative=2, _penalty=derivative):
    # return FDMatrix.derivative(n, order=derivative, periodic=True)

    return _penalty(n, derivative=derivative, periodic=True)


def l2(n):
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
    return np.eye(n)


def no_penalty(n):
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
    return np.zeros((n, n))


# =============================================================================
# CONSTRAINTS
# =============================================================================


def monotonic_inc(coef):
    """Monotonic increasing constraint, penalizing decreases.


    Parameters
    ----------
    coef : np.ndarray
        Array of coefficients.

    Returns
    -------
    D : np.ndarray
        Matrix such that penalty = (D @ coef).dot(D @ coef).

    Examples
    --------
    >>> a = np.array([1, 2, 3, 4])
    >>> monotonic_inc(a) @ a
    array([0., 0., 0., 0.])
    >>> a = np.array([1, 2, 0, 4])
    >>> monotonic_inc(a) @ a
    array([ 0.,  0., -2.,  0.])
    >>> a = np.array([0, 2, 3, 4])
    >>> monotonic_inc(a) @ a
    array([0., 0., 0., 0.])
    >>> a = np.array([1, 2, 3, 0])
    >>> monotonic_inc(a) @ a
    array([ 0.,  0.,  0., -3.])

    """
    # return monotonicity_(coef, increasing=True)

    coef = coef.ravel()
    # For an array   [1, 4,  3,  2, 5]
    # the penalty is [0, 0, -1, -1, 0]
    # and this encourages elements corresponding to 3 and 2 to grow
    # in the minimization problem
    D = FDMatrix.backward_diff(len(coef), periodic=False)
    mask = np.diff(coef) > 0
    mask = np.hstack(([True], mask))
    D[mask, :] = 0
    return D


def monotonic_dec(coef):
    """


    Parameters
    ----------
    coef : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    Examples
    --------
    >>> a = np.array([4, 3, 2, 1])
    >>> monotonic_dec(a) @ a
    array([0., 0., 0., 0.])
    >>> a = np.array([4, 3, 0, 1])
    >>> monotonic_dec(a) @ a
    array([ 0.,  0., -1.,  0.])
    >>> a = np.array([4, 3, 2, 0])
    >>> monotonic_dec(a) @ a
    array([0., 0., 0., 0.])
    >>> a = np.array([0, 3, 2, 1])
    >>> monotonic_dec(a) @ a
    array([-3.,  0.,  0.,  0.])

    """
    # return monotonicity_(coef, increasing=False)

    coef = coef.ravel()
    # For an array   [ 1, 4, 3,  2, 5]
    # the penalty is [-3, 0, 0, -3, 0]
    # and this encourages elements corresponding to 3 and 2 to grow
    # in the minimization problem

    # equals to -monotonic_inc(5, a[::-1]) @ a
    D = FDMatrix.forward_diff(len(coef), periodic=False)
    mask = np.diff(coef) < 0
    mask = np.hstack((mask, [True]))
    D[mask, :] = 0
    return -D


def monotonicity_(coef, increasing=True):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of monotonicity in the feature function.

    Parameters
    ----------
    coef : array-like
        coefficients of the feature function
    increasing : bool, default: True
        whether to enforce monotic increasing, or decreasing functions
    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    n = len(coef)

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
    return D.A.T
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
    return D.A.T


def convex(coef):
    """


    Parameters
    ----------
    coef : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    Examples
    --------
    >>> a = np.array([4, 10, 6])
    >>> convex(a) @ a
    array([0., 5., 0.])


    """
    if len(coef) <= 2:
        return np.zeros(shape=(len(coef), len(coef)))

    # return monotonicity_(coef, increasing=True)

    coef = coef.ravel()
    # For an array   [1, 4,  3,  2, 5]
    # the penalty is [0, 0, -1, -1, 0]
    # and this encourages elements corresponding to 3 and 2 to grow
    # in the minimization problem
    Db = FDMatrix.backward_diff(len(coef), periodic=False)
    Df = FDMatrix.forward_diff(len(coef), periodic=False)
    D = (Df @ Db) / 2
    mask = np.diff(coef, n=2) > 0
    mask = np.hstack(([True], mask, [True]))
    D[mask, :] = 0
    return -D

    # return convexity_(n, coef, convex=True)


def concave(coef):
    if len(coef) <= 2:
        return np.zeros(shape=(len(coef), len(coef)))

    coef = coef.ravel()
    # For an array   [ 1, 4, 3,  2, 5]
    # the penalty is [-3, 0, 0, -3, 0]
    # and this encourages elements corresponding to 3 and 2 to grow
    # in the minimization problem

    # equals to -monotonic_inc(5, a[::-1]) @ a
    Db = FDMatrix.backward_diff(len(coef), periodic=False)
    Df = FDMatrix.forward_diff(len(coef), periodic=False)
    D = (Df @ Db) / 2
    mask = np.diff(coef, n=2) < 0
    mask = np.hstack(([True], mask, [True]))
    D[mask, :] = 0
    return -D

    # return convexity_(n, coef, convex=False)


def no_constraint(coef):
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
    return np.eye(len(coef)) * 0
    return sp.sparse.diags([0] * len(coef))


# =============================================================================
# OTHER
# =============================================================================


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


PENALTIES = {
    "auto": "auto",
    "derivative": functools.partial(FDMatrix.derivative, order=2, periodic=False),
    "l2": l2,
    "none": no_penalty,
    "periodic": functools.partial(FDMatrix.derivative, order=2, periodic=True),
}


# =============================================================================
# PENALTIES
# =============================================================================


CONSTRAINTS = {
    "convex": convex,
    "concave": concave,
    "monotonic_inc": monotonic_inc,
    "monotonic_dec": monotonic_dec,
    "none": no_constraint,
}

if __name__ == "__main__":
    import pytest

    # pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules"])

    if True:

        np.random.seed(1)
        a = np.random.randn(20)
        a = np.array([1, 5, 3, 2])

        constraint = convex

        import matplotlib.pyplot as plt

        # plt.title(error_vec.dot(error_vec))
        plt.plot(a)
        plt.plot(constraint(a) @ a)
        plt.show()

        error_vec = constraint(a) @ a
        while error_vec.dot(error_vec) > 1e-18:

            print(a)
            print(error_vec)
            print()

            a = a - 1.1 * error_vec

            import time

            time.sleep(0.01)

            import matplotlib.pyplot as plt

            plt.title(error_vec.dot(error_vec))
            plt.plot(a)
            plt.plot(constraint(a) @ a)
            plt.show()

            error_vec = constraint(a) @ a

        assert np.allclose(error_vec, 0)
