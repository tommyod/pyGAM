# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam import LinearGAM, s


class TestInvariancesToAdditionAndMultiPlication:
    def test_that_cyclic_spline_can_match_sine_wave(self):

        X = np.linspace(0, 1, 1000, endpoint=False).reshape(-1, 1)
        y = np.sin(X.ravel() * 2 * np.pi)

        gam = LinearGAM(s(0, basis="cp", n_splines=4, spline_order=2)).fit(X, y)

        assert gam.predict(X).shape == y.shape
        assert np.allclose(y, gam.predict(X), atol=0.1)

        # Increase the number of splines a little bit
        gam = LinearGAM(s(0, basis="cp", n_splines=16)).fit(X, y)
        assert np.allclose(y, gam.predict(X), atol=0.05)

    def test_that_spline_can_match_logarithm(self):
        X = np.linspace(0, 10, 1000).reshape(-1, 1)
        y = np.log(1 + X.ravel())
        gam = LinearGAM(s(0, basis="ps", n_splines=8, lam=0.1)).fit(X, y)

        y_pred = gam.predict(X)
        assert np.allclose(y, y_pred, atol=0.1)

    @pytest.mark.parametrize("factor", np.logspace(-5, 5, num=11))
    def test_that_cyclic_y_is_invariant_to_addition(self, factor):
        X = np.linspace(0, 1, 1000, endpoint=False).reshape(-1, 1)
        y = np.sin(X.ravel() * 2 * np.pi) + factor

        gam = LinearGAM(s(0, basis="cp")).fit(X, y)
        assert np.allclose(y, gam.predict(X), atol=0.01)

    @pytest.mark.parametrize("factor", np.logspace(-5, 5, num=11))
    def test_that_cyclic_spline_X_is_invariant_to_addition(self, factor):
        X = np.linspace(0, 1, 1000, endpoint=False).reshape(-1, 1) + factor
        y = np.sin(X.ravel() * 2 * np.pi)

        gam = LinearGAM(s(0, basis="cp")).fit(X, y)
        assert np.allclose(y, gam.predict(X), atol=0.01)

    @pytest.mark.parametrize("factor", [1, 2, 5])
    def test_that_cyclic_spline_X_is_invariant_to_multiplication(self, factor):

        X = np.linspace(0, 1, 1000, endpoint=False).reshape(-1, 1) * factor
        y = np.sin(X.ravel() * 2 * np.pi) * factor

        gam = LinearGAM(s(0, basis="cp", n_splines=8 * factor)).fit(X, y)
        assert np.allclose(y, gam.predict(X), atol=0.1)

    @pytest.mark.parametrize("factor", np.logspace(0, 6, num=7))
    def test_intercept_equals_mean(self, factor):
        X = np.linspace(0, 1, 1000, endpoint=False).reshape(-1, 1)
        y = np.sin(X.ravel() * 2 * np.pi) + factor

        gam = LinearGAM(s(0, basis="cp")).fit(X, y)

        for term, coefs in gam.yield_terms_and_betas():
            if term.isintercept:
                # Intercept term should equal the mean value in the data
                assert np.allclose(coefs[0], factor)
            else:
                # Betas in the spline term should be close to 1
                assert np.isclose(np.sum(coefs), 0)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "-k test_intercept_equals_mean"])

    X = np.linspace(0, 10, 1000).reshape(-1, 1)
    y = np.log(1 + X.ravel())
    gam = LinearGAM(s(0, basis="ps", n_splines=8, lam=0.1)).fit(X, y)

    y_pred = gam.predict(X)

    import matplotlib.pyplot as plt

    plt.scatter(X, y)
    plt.plot(X, gam.predict(X), color="black")

    assert np.allclose(y, y_pred, atol=0.1)
