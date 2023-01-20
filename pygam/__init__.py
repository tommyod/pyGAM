"""
GAM toolkit
"""

from pygam.gam import GAM, ExpectileGAM, GammaGAM, InvGaussGAM, LinearGAM, LogisticGAM, PoissonGAM
from pygam.terms import f, intercept, l, s, te

__all__ = [
    "GAM",
    "LinearGAM",
    "LogisticGAM",
    "GammaGAM",
    "PoissonGAM",
    "InvGaussGAM",
    "ExpectileGAM",
    "l",
    "s",
    "f",
    "te",
    "intercept",
]

__version__ = "0.8.0"
