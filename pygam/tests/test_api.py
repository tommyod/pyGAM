# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam import LinearGAM, s


class TestSimpleAPIExamples:
    def test_that_spline_can_match_sine_wave(self):

        X = np.linspace(0, 1, 5000).reshape(-1, 1)
        y = np.sin(X.ravel() * 2 * np.pi)

        gam = LinearGAM(s(0, basis="cp", n_splines=4, spline_order=2, lam=1)).fit(X, y)

        assert gam.predict(X).shape == y.shape
        assert np.allclose(y, gam.predict(X), atol=0.1)

        # Increase the number of splines a little bit
        gam = LinearGAM(s(0, basis="cp", n_splines=16, spline_order=2, lam=1)).fit(X, y)
        assert np.allclose(y, gam.predict(X), atol=0.05)

    def test_that_spline_can_match_logarithm(self):

        # define square wave
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = np.log(1 + X.ravel())
        gam = LinearGAM(s(0, basis="ps", n_splines=8, spline_order=2, lam=1)).fit(X, y)

        y_pred = gam.predict(X)
        assert np.allclose(y, y_pred, atol=0.25)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys", "--doctest-modules", "-k TestSimpleAPIExamples"])

    X = np.linspace(0, 1, 5000).reshape(-1, 1)
    y = np.sin(X.ravel() * 2 * np.pi) + 10

    gam = LinearGAM(s(0, basis="cp", n_splines=4, spline_order=2, lam=1)).fit(X, y)

    y_pred = gam.predict(X)

    import matplotlib.pyplot as plt

    plt.plot(X.ravel(), y)
    plt.plot(X.ravel(), gam.predict(X))
    plt.show()
