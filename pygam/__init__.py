"""
GAM toolkit
"""

from pygam.gam import GAM
from pygam.gam import LinearGAM
from pygam.gam import LogisticGAM
from pygam.gam import GammaGAM
from pygam.gam import PoissonGAM
from pygam.gam import InvGaussGAM
from pygam.gam import ExpectileGAM

from pygam.terms import l
from pygam.terms import s
from pygam.terms import f
from pygam.terms import te
from pygam.terms import intercept

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
