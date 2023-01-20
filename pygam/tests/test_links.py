# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam.distributions import BinomialDist
from pygam.links import LINKS


class TestLink:
    @pytest.mark.parametrize("link", LINKS.values())
    def test_that_links_are_inverses_link_mu(self, link):

        argument = np.random.rand(100)
        dist = BinomialDist(levels=1)
        assert np.allclose(link().mu(link().link(argument, dist), dist), argument)

    @pytest.mark.parametrize("link", LINKS.values())
    def test_that_links_are_inverses_mu_link(self, link):

        argument = np.random.rand(100)
        dist = BinomialDist(levels=1)
        assert np.allclose(link().link(link().mu(argument, dist), dist), argument)

    @pytest.mark.parametrize("link", LINKS.values())
    def test_that_links_derivatives_are_close_to_finite_differences(self, link):

        argument = np.random.rand(100)
        dist = BinomialDist(levels=1)
        epsilon = np.ones_like(argument) * 1e-8  # 8, 9, 10 seems to work

        # Derivative from equation vs. finite difference approximation to the derivative
        f_x_deriv = link().gradient(argument, dist)
        f_x_finite_diff = (link().link(argument + epsilon, dist) - link().link(argument, dist)) / epsilon

        # Atleast 90% must be close. This test is not exact. Depends on random
        # numbers and numerics...
        assert np.mean(np.isclose(f_x_deriv, f_x_finite_diff) > 0.9)

    def test_loglink_domain(self):
        loglink = LINKS["log"]()
        assert loglink.get_domain(dist=None) == [0, np.inf]


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys"])
