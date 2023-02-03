"""
Link functions
"""
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import special

from pygam.core import Core
import warnings


class Link(Core, metaclass=ABCMeta):
    # https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function
    @abstractmethod
    def link(self, mu, dist):
        # The link function
        pass

    @abstractmethod
    def mu(self, lp, dist):
        # The inverse link function
        pass

    @abstractmethod
    def gradient(self, mu, dist):
        # Gradient of the link function
        pass

    def get_domain(self, dist):
        """
        Identify the domain of a given monotonic link function

        Parameters
        ----------
        dist : Distribution object

        Returns
        -------
        domain : list of length 2, representing the interval of the domain.
        """
        domain = np.array([-np.inf, -1, 0, 1, np.inf])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            low, *_, high = domain[~np.isnan(self.link(domain, dist))]
        return [low, high]


class IdentityLink(Link):
    name = "identity"
    domain = (-np.inf, np.inf)

    def link(self, mu, dist):
        return mu

    def mu(self, lp, dist):
        return lp

    def gradient(self, mu, dist):
        return np.ones_like(mu)


class LogitLink(Link):
    name = "logit"
    domain = (-np.inf, np.inf)

    def link(self, mu, dist):
        return np.log(mu) - np.log(dist.levels - mu)

    def mu(self, lp, dist):
        # expit(x) = 1 / (1 + exp(-x))
        return dist.levels * special.expit(lp)

    def gradient(self, mu, dist):
        return dist.levels / (mu * (dist.levels - mu))


class CLogLogLink(Link):
    name = "cloglog"

    def link(self, mu, dist):
        return np.log(np.log(dist.levels) - np.log(dist.levels - mu))

    def mu(self, lp, dist):
        return dist.levels * np.exp(-np.exp(lp)) * (np.exp(np.exp(lp)) - 1)

    def gradient(self, mu, dist):
        return 1 / ((dist.levels - mu) * (np.log(dist.levels) - np.log(dist.levels - mu)))


class LogLink(Link):
    name = "log"

    def link(self, mu, dist):
        return np.log(mu)

    def mu(self, lp, dist):
        return np.exp(lp)

    def gradient(self, mu, dist):
        return 1.0 / mu


class InverseLink(Link):
    name = "inverse"

    def link(self, mu, dist):
        return 1.0 / mu

    def mu(self, lp, dist):
        return 1.0 / lp

    def gradient(self, mu, dist):
        return -1.0 / mu**2


class InvSquaredLink(Link):
    name = "inv_squared"

    def link(self, mu, dist):
        return 1.0 / mu**2

    def mu(self, lp, dist):
        return 1.0 / np.sqrt(lp)

    def gradient(self, mu, dist):
        return -2.0 / mu**3


# Dict comprehension instead of hard-coding the names again here
LINKS = {l.name: l for l in [IdentityLink, LogLink, LogitLink, InverseLink, InvSquaredLink, CLogLogLink]}


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "-v", "--capture=sys", "-k TestLink"])
