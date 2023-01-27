# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam import LinearGAM, s, l


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

    def test_that_linear_term_penalties_work_as_expected(self):

        # Create a problem
        generator = np.random.default_rng(23)
        X = generator.normal(size=(10_000, 2))
        y = 100 + X @ np.array([1, 2]) + generator.normal(size=(10_000), scale=0.01)

        # Fit a GAM with linear terms and regularization
        gam = LinearGAM(l(0, lam=10000) + l(1, lam=9)).fit(X, y)
        penalties = np.diag(gam._P().A)

        assert np.isclose(penalties[0], np.sqrt(10000)), "sqrt(lam) penalty on linear term"
        assert np.isclose(penalties[1], np.sqrt(9)), "sqrt(lam) penalty on linear term"
        assert np.isclose(penalties[2], 0), "No penalty on constant term"

        # First estimate is pulled down by the strong regularization
        # print((repr(gam.coef_.round(6))))
        assert np.allclose(gam.coef_, np.array([0.502616, 1.999487, 99.990853]))

        # Fit a GAM with linear terms with zero regularization
        gam = LinearGAM(l(0, lam=0) + l(1, lam=0)).fit(X, y)
        penalties = np.diag(gam._P().A)

        assert np.allclose(penalties, 0), "No penalties on any term"

        # print((repr(gam.coef_.round(6))))
        assert np.allclose(gam.coef_, np.array([0.999936, 1.999744, 99.999993]))


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules"])

    generator = np.random.default_rng(23)
    X = generator.normal(size=(1000, 2))
    y = 100 + X @ np.array([1, 2])
    gam = LinearGAM(l(0, lam=100) + l(1, lam=9)).fit(X, y)
    penalties = np.diag(gam._P().A)

    assert np.isclose(penalties[0], np.sqrt(100)), "sqrt(lam) penalty on linear term"
    assert np.isclose(penalties[1], np.sqrt(9)), "sqrt(lam) penalty on linear term"
    assert np.isclose(penalties[2], 0), "No penalty on constant term"

    y_pred = gam.predict(X)
