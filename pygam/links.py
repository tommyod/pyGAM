"""
Link functions
"""
import numpy as np

from pygam.core import Core


class Link(Core):
    def __init__(self, name=None):
        super().__init__(name=name)


class IdentityLink(Link):

    name = "identity"

    def __init__(self):
        super().__init__(name="identity")

    def link(self, mu, dist):
        return mu

    def mu(self, lp, dist):
        return lp

    def gradient(self, mu, dist):
        return np.ones_like(mu)


class LogitLink(Link):
    def __init__(self):
        super().__init__(name="logit")

    def link(self, mu, dist):
        return np.log(mu) - np.log(dist.levels - mu)

    def mu(self, lp, dist):
        elp = np.exp(lp)
        return dist.levels * elp / (elp + 1)

    def gradient(self, mu, dist):
        return dist.levels / (mu * (dist.levels - mu))


class LogLink(Link):
    def __init__(self):
        super().__init__(name="log")

    def link(self, mu, dist):
        return np.log(mu)

    def mu(self, lp, dist):
        return np.exp(lp)

    def gradient(self, mu, dist):
        return 1.0 / mu


class InverseLink(Link):
    def __init__(self):
        super().__init__(name="inverse")

    def link(self, mu, dist):
        return mu**-1.0

    def mu(self, lp, dist):
        return lp**-1.0

    def gradient(self, mu, dist):
        return -1 * mu**-2.0


class InvSquaredLink(Link):
    def __init__(self):
        super().__init__(name="inv_squared")

    def link(self, mu, dist):
        return mu**-2.0

    def mu(self, lp, dist):
        return lp**-0.5

    def gradient(self, mu, dist):
        return -2 * mu**-3.0


LINKS = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "inverse": InverseLink,
    "inv_squared": InvSquaredLink,
}
