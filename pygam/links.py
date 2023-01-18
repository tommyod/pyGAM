"""
Link functions
"""
import numpy as np

from pygam.core import Core


class Link(Core):
    pass


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
        return np.log(mu) - np.log(dist.levels - mu)

    def mu(self, lp, dist):
        elp = np.exp(lp)
        return dist.levels * elp / (elp + 1)

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
        return mu**-1.0

    def mu(self, lp, dist):
        return lp**-1.0

    def gradient(self, mu, dist):
        return -1 * mu**-2.0


class InvSquaredLink(Link):

    name = "inv_squared"

    def link(self, mu, dist):
        return mu**-2.0

    def mu(self, lp, dist):
        return lp**-0.5

    def gradient(self, mu, dist):
        return -2 * mu**-3.0


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
