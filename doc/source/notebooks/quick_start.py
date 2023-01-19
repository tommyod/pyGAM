# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quick Start
#
# This quick start will show how to do the following:
#
# - `Install` everything needed to use pyGAM.
# - `fit a regression model` with custom terms
# - search for the `best smoothing parameters`
# - plot `partial dependence` functions
#
#
# ## Install pyGAM
# #### Pip
#
#     pip install pygam
#
#
# #### Conda
# pyGAM is on conda-forge, however this is typically less up-to-date:
#
#     conda install -c conda-forge pygam
#     
#
# #### Bleeding edge
# You can install the bleeding edge from github using `flit`.
# First clone the repo, ``cd`` into the main directory and do:
#
#     pip install flit
#     flit install

# %% [markdown]
# #### Get `pandas` and `matplotlib`
#
#     pip install pandas matplotlib
#
#

# %% [markdown]
# ## Fit a Model
#
# Let's get to it. First we need some data:

# %%
from pygam.datasets import wage

X, y = wage()

# %% [markdown]
# Now let's import a GAM that's made for regression problems.
#
# Let's fit a spline term to the first 2 features, and a factor term to the 3rd feature.

# %%
from pygam import LinearGAM, s, f

gam = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)

# %% [markdown]
# Let's take a look at the model fit:

# %%
gam.summary()

# %% [markdown]
# Even though we have 3 terms with a total of `(20 + 20 + 5) = 45` free variables, the default smoothing penalty (`lam=0.6`) reduces the effective degrees of freedom to just ~25.

# %% [markdown]
# By default, the spline terms, `s(...)`, use 20 basis functions. This is a good starting point. The rule of thumb is to use a fairly large amount of flexibility, and then let the smoothing penalty regularize the model.
#
# However, we can always use our expert knowledge to add flexibility where it is needed, or remove basis functions, and make fitting easier:

# %%
gam = LinearGAM(s(0, n_splines=5) + s(1) + f(2)).fit(X, y)

# %% [markdown]
# ## Automatically tune the model

# %% [markdown]
# By default, spline terms, `s()` have a penalty on their 2nd derivative, which encourages the functions to be smoother, while factor terms, `f()` and linear terms `l()`, have a l2, ie ridge penalty, which encourages them to take on smaller values.
#
# `lam`, short for $\lambda$, controls the strength of the regularization penalty on each term. Terms can have multiple penalties, and therefore multiple `lam`.

# %%
print(gam.lam)

# %% [markdown]
# Our model has 3 `lam` parameters, currently just one per term.
#
# Let's perform a grid-search over multiple `lam` values to see if we can improve our model.  
# We will seek the model with the lowest generalized cross-validation (GCV) score.
#
# Our search space is 3-dimensional, so we have to be conservative with the number of points we consider per dimension.
#
# Let's try 5 values for each smoothing parameter, resulting in a total of `5*5*5 = 125` points in our grid.

# %%
import numpy as np

lam = np.logspace(-3, 5, 5)
lams = [lam] * 3

gam.gridsearch(X, y, lam=lams)
gam.summary()

# %% [markdown]
# This is quite a bit better. Even though the in-sample $R^2$ value is lower, we can expect our model to generalize better because the GCV error is lower.
#
# We could be more rigorous by using a train/test split, and checking our model's error on the test set. We were also quite lazy and only tried 125 values in our hyperopt. We might find a better model if we spent more time searching across more points.

# %% [markdown]
# For high-dimensional search-spaces, it is sometimes a good idea to try a **randomized search**.  
# We can acheive this by using numpy's `random` module:

# %%
lams = np.random.rand(100, 3) # random points on [0, 1], with shape (100, 3)
lams = lams * 6 - 3 # shift values to -3, 3
lams = 10 ** lams # transforms values to 1e-3, 1e3

# %%
random_gam =  LinearGAM(s(0) + s(1) + f(2)).gridsearch(X, y, lam=lams)
random_gam.summary()

# %% [markdown]
# In this case, our deterministic search found a better model:

# %%
gam.statistics_['GCV'] < random_gam.statistics_['GCV']

# %% [markdown]
# The `statistics_` attribute is populated after the model has been fitted.
# There are lots of interesting model statistics to check out, although many are automatically reported in the model summary:

# %%
list(gam.statistics_.keys())

# %% [markdown]
# ## Partial Dependence Functions

# %% [markdown]
# One of the most attractive properties of GAMs is that we can decompose and inspect the contribution of each feature to the overall prediction. 
#
# This is done via **partial dependence** functions.
#
# Let's plot the partial dependence for each term in our model, along with a 95% confidence interval for the estimated function.

# %%
import matplotlib.pyplot as plt

# %%
for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue
        
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    
    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()

# %% [markdown]
# Note: we skip the intercept term because it has nothing interesting to plot.
