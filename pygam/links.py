"""
Link functions
"""
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import special

from pygam.core import Core


class Link(Core, metaclass=ABCMeta):
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
        tIdentify the domain of a given monotonic link function

        Parameters
        ----------
        dist : Distribution object

        Returns
        -------
        domain : list of length 2, representing the interval of the domain.
        """
        domain = np.array([-np.inf, -1, 0, 1, np.inf])
        low, *_, high = domain[~np.isnan(self.link(domain, dist))]
        return [low, high]


class IdentityLink(Link):

    name = "identity"

    def link(self, mu, dist):
        return mu

    def mu(self, lp, dist):
        return lp

    def gradient(self, mu, dist):
        return np.ones_like(mu)


class LogitLink(Link):

    name = "logit"

    def link(self, mu, dist):
        return special.logit(mu / dist.levels) - np.log(dist.levels)

    def mu(self, lp, dist):
        return dist.levels * special.expit(lp)

    def gradient(self, mu, dist):
        return dist.levels / (mu * (dist.levels - mu))


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
LINKS = {
    l.name: l
    for l in [
        IdentityLink,
        LogLink,
        LogitLink,
        InverseLink,
        InvSquaredLink,
    ]
}


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "-v", "--capture=sys", "-k TestLink"])
