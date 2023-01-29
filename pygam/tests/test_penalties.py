# -*- coding: utf-8 -*-

import itertools

import numpy as np
import pytest

from pygam import LinearGAM, s
from pygam.penalties import concave, convex, derivative, l2, monotonic_dec, monotonic_inc, no_constraint, no_penalty


class TestFiniteDifferenceMatrices:
    pass


class TestContraints:
    @pytest.mark.parametrize("n", [5, 9, 15])
    def test_monotonic_increase(self, n):
        coefs = np.arange(n)
        P = monotonic_inc(coefs)

        # Increasing coefficients get no penalty
        assert np.allclose(P @ coefs, 0)

        # Set one variable to be zero in the middle
        coefs[len(coefs) // 2] = 0
        P = monotonic_inc(coefs)
        penalty = (P @ coefs).dot(P @ coefs)
        print(P @ coefs)
        print(penalty)
        assert penalty > 0

    @pytest.mark.parametrize(
        "constraint,length", itertools.product([convex, concave, monotonic_inc, monotonic_dec], list(range(3, 31)))
    )
    def test_constraint_as_optimization_problem(self, constraint, length):

        rng = np.random.default_rng(1)
        a = rng.normal(size=length)

        # Error vector
        num_iterations = 0
        error_vec = constraint(a) @ a
        while error_vec.dot(error_vec) > 1e-18:

            # Move to minimize error
            step = 1.1
            a = a - step * error_vec
            error_vec = constraint(a) @ a
            num_iterations += 1

        assert np.allclose(error_vec, 0)

        # TODO: Sometimes many iteations (> length) is needed.
        # Could there be a smarter way to do this?
        assert num_iterations <= 4**length


def test_single_spline_penalty():
    """
    check that feature functions with only 1 basis are penalized correctly

    derivative penalty should be 0.
    l2 should penalty be 1.
    monotonic_ and convexity_ should be 0.
    """
    coef = np.array([1.0])

    # Penalties
    assert np.alltrue(derivative(1) == 0.0)
    assert np.alltrue(l2(1) == 1.0)
    assert np.alltrue(no_penalty(1) == 0.0)

    # Constraints
    assert np.alltrue(no_constraint(coef) == 0.0)
    assert np.alltrue(monotonic_inc(coef) == 0.0)
    assert np.alltrue(monotonic_dec(coef) == 0.0)
    assert np.alltrue(convex(coef) == 0.0)
    assert np.alltrue(concave(coef) == 0.0)


def test_monotonic_inchepatitis_X_y(hepatitis_X_y):
    """
    check that monotonic_inc constraint produces monotonic increasing function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints="monotonic_inc", lam=1), max_iter=99)
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=1)

    # This is the OR function
    assert ((diffs >= 0) + np.isclose(diffs, 0.0, atol=1e-3)).all()


def test_monotonic_dec(hepatitis_X_y):
    """
    check that monotonic_dec constraint produces monotonic decreasing function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints="monotonic_dec"))
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=1)
    assert ((diffs <= 0) + np.isclose(diffs, 0.0, atol=1e-6)).all()


def test_convex(hepatitis_X_y):
    """
    check that convex constraint produces convex function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints="convex", lam=1), max_iter=99)
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=2)
    assert ((diffs >= 0) + np.isclose(diffs, 0.0, atol=1e-4)).all()


def test_concave(hepatitis_X_y):
    """
    check that concave constraint produces concave function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints="concave", lam=1), max_iter=99)
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=2)
    assert ((diffs <= 0) + np.isclose(diffs, 0.0, atol=1e-4)).all()


# TODO penalties gives expected matrix structure
# TODO circular constraints
if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "-k test_convex"])

    if True:
        from pygam.datasets import hepatitis

        X, y = hepatitis(True)

        for max_iter in range(2, 99):

            # This dataset only has one feature

            X, y = hepatitis(True)

            # Add a random interaction data
            rng = np.random.default_rng(1)
            X = rng.normal(size=(100, 1))
            X = X[np.argsort(X[:, 0]), :]
            y = 1 / (1 + np.exp(-X[:, 0]))
            y = np.sin(X[:, 0])

            # constrain useless dimension
            gam = LinearGAM(s(0, n_splines=10, lam=1, constraints="convex", spline_order=3), max_iter=max_iter)
            gam.fit(X, y)

            import matplotlib.pyplot as plt

            plt.scatter(X[:, 0], y)
            plt.plot(X[:, 0], gam.predict(X), color="red")

            plt.show()

            import time

            time.sleep(2)
