# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam.links import LINKS
from pygam.distributions import BinomialDist


class TestLink:
    @pytest.mark.parametrize("linkname,link", LINKS.items())
    def test_that_links_are_inverses_link_mu(self, linkname, link):

        argument = np.random.rand(100)
        dist = BinomialDist(levels=1)
        assert np.allclose(link().mu(link().link(argument, dist), dist), argument)

    @pytest.mark.parametrize("linkname,link", LINKS.items())
    def test_that_links_are_inverses_mu_link(self, linkname, link):

        argument = np.random.rand(100)
        dist = BinomialDist(levels=1)
        assert np.allclose(link().link(link().mu(argument, dist), dist), argument)

    @pytest.mark.parametrize("linkname,link", LINKS.items())
    def test_that_links_derivatives_are_close_to_finite_differences(self, linkname, link):

        argument = np.random.rand(100)
        dist = BinomialDist(levels=1)
        epsilon = np.ones_like(argument) * 1e-8  # 8, 9, 10 seems to work

        # Derivative from equation vs. finite difference approximation to the derivative
        f_x_deriv = link().gradient(argument, dist)
        f_x_finite_diff = (link().link(argument + epsilon, dist) - link().link(argument, dist)) / epsilon

        # Atleast 90% must be close. This test is not exact. Depends on random
        # numbers and numerics...
        assert np.mean(np.isclose(f_x_deriv, f_x_finite_diff) > 0.9)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys"])