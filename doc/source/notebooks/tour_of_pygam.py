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
# # A Tour of pyGAM
#
# ## Introduction
#
# Generalized Additive Models (GAMs) are smooth semi-parametric models of the form:
#
# $$
#     g(\mathbb{E}[y|X]) = \beta_0 + f_1(X_1) + f_2(X_2, X3) + \ldots + f_M(X_N)
# $$
#
# where `X.T = [X_1, X_2, ..., X_N]` are independent variables, `y` is the dependent variable, and `g()` is the link function that relates our predictor variables to the expected value of the dependent variable.
#
# The feature functions `f_i()` are built using **penalized B splines**, which allow us to **automatically model non-linear relationships** without having to manually try out many different transformations on each variable.
#
#
# ![Basis splines](pygam_basis.png)
#
# GAMs extend generalized linear models by allowing non-linear functions of features while maintaining additivity. Since the model is additive, it is easy to examine the effect of each `X_i` on `Y` individually while holding all other predictors constant.
#
# The result is a very flexible model, where it is easy to incorporate prior knowledge and control overfitting.
#
#

# %% [markdown]
# ## Generalized Additive Models, in general

# %% [markdown]
# $$
# y \sim ExponentialFamily(\mu|X)
# $$
#
# where 
# $$
# g(\mu|X) = \beta_0 + f_1(X_1) + f_2(X_2, X3) + \ldots + f_M(X_N)
# $$
#
# So we can see that a GAM has 3 components:
#
# - ``distribution`` from the exponential family
# - ``link function`` $g(\cdot)$
# - ``functional form`` with an additive structure $\beta_0 + f_1(X_1) + f_2(X_2, X3) + \ldots + f_M(X_N)$
#
# ### Distribution: 
# Specified via: ``GAM(distribution='...')``
#
# Currently you can choose from the following:
#
# - `'normal'`
# - `'binomial'`
# - `'poisson'`
# - `'gamma'`
# - `'inv_gauss'`
#
# ### Link function: 
# We specify this using: ``GAM(link='...')``
#
# Link functions take the distribution mean to the linear prediction. So far, the following are available:
#
# - `'identity'`
# - `'logit'`
# - `'inverse'`
# - `'log'`
# - `'inverse-squared'`
#
#
# ### Functional Form: 
# Speficied in ``GAM(terms=...)`` or more simply ``GAM(...)``
#
# In pyGAM, we specify the functional form using terms:
#
# - `l()` linear terms: for terms like $X_i$
# - `s()` spline terms
# - `f()` factor terms
# - `te()` tensor products
# - `intercept`  

# %% [markdown]
# With these, we can quickly and compactly build models like:

# %%
import numpy as np
from pygam import GAM, s, te

GAM(s(0, n_splines=200) + te(3,1) + s(2), distribution='poisson', link='log')

# %% [markdown]
# which specifies that we want a:
#
# - spline function on feature 0, with 200 basis functions
# - tensor spline interaction on features 1 and 3
# - spline function on feature 2

# %% [markdown]
# Note:
#
# ``GAM(..., intercept=True)`` so models include an intercept by default.

# %% [markdown]
# ### in Practice...
# in **pyGAM** you can build custom models by specifying these 3 elements, **or** you can choose from **common models**:
#
# - `LinearGAM` identity link and normal distribution
# - `LogisticGAM` logit link and binomial distribution
# - `PoissonGAM` log link and Poisson distribution
# - `GammaGAM` log link and gamma distribution
# - `InvGauss` log link and inv_gauss distribution
#
# The benefit of the common models is that they have some extra features, apart from reducing boilerplate code.

# %% [markdown]
# ## Terms and Interactions
#
# pyGAM can also fit interactions using tensor products via `te()`

# %%
from pygam import PoissonGAM, s, te
from pygam.datasets import chicago

X, y = chicago(return_X_y=True)

gam = PoissonGAM(s(0, n_splines=200) + te(3, 1) + s(2)).fit(X, y)

# %% [markdown]
# and plot a 3D surface:

# %%
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

plt.ion()
plt.rcParams['figure.figsize'] = (12, 8)

# %%
XX = gam.generate_X_grid(term=1, meshgrid=True)
Z = gam.partial_dependence(term=1, X=XX, meshgrid=True)

ax = plt.axes(projection='3d')
ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')

# %% [markdown]
# For simple interactions it is sometimes useful to add a by-variable to a term

# %%
from pygam import LinearGAM, s
from pygam.datasets import toy_interaction

X, y = toy_interaction(return_X_y=True)

gam = LinearGAM(s(0, by=1)).fit(X, y)
gam.summary()

# %% [markdown]
# ## Regression
#
# For **regression** problems, we can use a **linear GAM** which models:
#
# $$
#     \mathbb{E}[y|X]=\beta_0+f_1(X_1)+f_2(X_2, X3)+\dots+f_M(X_N)
# $$

# %%
from pygam import LinearGAM, s, f
from pygam.datasets import wage

X, y = wage(return_X_y=True)

## model
gam = LinearGAM(s(0) + s(1) + f(2))
gam.gridsearch(X, y)


## plotting
plt.figure();
fig, axs = plt.subplots(1,3);

titles = ['year', 'age', 'education']
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    if i == 0:
        ax.set_ylim(-30,30)
    ax.set_title(titles[i]);

# %% [markdown]
# Even though our model allows coefficients, our smoothing penalty reduces us to just 19 effective degrees of freedom:

# %%
gam.summary()

# %% [markdown]
# With **LinearGAMs**, we can also check the **prediction intervals**:

# %%
from pygam import LinearGAM
from pygam.datasets import mcycle

X, y = mcycle(return_X_y=True)

gam = LinearGAM(n_splines=25).gridsearch(X, y)
XX = gam.generate_X_grid(term=0, n=500)

plt.plot(XX, gam.predict(XX), 'r--')
plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--')

plt.scatter(X, y, facecolor='gray', edgecolors='none')
plt.title('95% prediction interval');


# %% [markdown]
# And simulate from the posterior:

# %%
# continuing last example with the mcycle dataset
for response in gam.sample(X, y, quantity='y', n_draws=50, sample_at_X=XX):
    plt.scatter(XX, response, alpha=.03, color='k')
plt.plot(XX, gam.predict(XX), 'r--')
plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--')
plt.title('draw samples from the posterior of the coefficients')


# %% [markdown]
# ## Classification
#
# For **binary classification** problems, we can use a **logistic GAM** which models:
#
# $$
#     log\left(\frac{P(y=1|X)}{P(y=0|X)}\right)=\beta_0+f_1(X_1)+f_2(X_2, X3)+\dots+f_M(X_N)
# $$

# %%
from pygam import LogisticGAM, s, f
from pygam.datasets import default

X, y = default(return_X_y=True)

gam = LogisticGAM(f(0) + s(1) + s(2)).gridsearch(X, y)

fig, axs = plt.subplots(1, 3)
titles = ['student', 'balance', 'income']

for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, width=.95)

    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi, c='r', ls='--')
    ax.set_title(titles[i]);


# %% [markdown]
# We can then check the accuracy:

# %%
gam.accuracy(X, y)

# %% [markdown]
# Since the **scale** of the **Binomial distribution** is known, our gridsearch minimizes the **Un-Biased Risk Estimator** (UBRE) objective:

# %%
gam.summary()

# %% [markdown]
# ## Poisson and Histogram Smoothing
# We can intuitively perform **histogram smoothing** by modeling the counts in each bin
# as being distributed Poisson via **PoissonGAM**.

# %%
from pygam import PoissonGAM
from pygam.datasets import faithful

X, y = faithful(return_X_y=True)

gam = PoissonGAM().gridsearch(X, y)

plt.hist(faithful(return_X_y=False)['eruptions'], bins=200, color='k');
plt.plot(X, gam.predict(X), color='r')
plt.title('Best Lambda: {0:.2f}'.format(gam.lam[0][0]));

# %% [markdown]
# ## Expectiles
# GAMs with a Normal distribution suffer from the limitation of an assumed constant variance.
# Sometimes this is not an appropriate assumption, because we'd like the variance of our error distribution to vary.  
#
# In this case we can resort to modeling the **expectiles** of a distribution.   
#
# Expectiles are intuitively similar to quantiles, but model tail expectations instead of tail mass. Although they are less interpretable, expectiles are **much** faster to fit, and can also be used to non-parametrically model a distribution.

# %%
from pygam import ExpectileGAM
from pygam.datasets import mcycle

X, y = mcycle(return_X_y=True)

# lets fit the mean model first by CV
gam50 = ExpectileGAM(expectile=0.5).gridsearch(X, y)

# and copy the smoothing to the other models
lam = gam50.lam

# now fit a few more models
gam95 = ExpectileGAM(expectile=0.95, lam=lam).fit(X, y)
gam75 = ExpectileGAM(expectile=0.75, lam=lam).fit(X, y)
gam25 = ExpectileGAM(expectile=0.25, lam=lam).fit(X, y)
gam05 = ExpectileGAM(expectile=0.05, lam=lam).fit(X, y)

# %%
XX = gam50.generate_X_grid(term=0, n=500)

plt.scatter(X, y, c='k', alpha=0.2)
plt.plot(XX, gam95.predict(XX), label='0.95')
plt.plot(XX, gam75.predict(XX), label='0.75')
plt.plot(XX, gam50.predict(XX), label='0.50')
plt.plot(XX, gam25.predict(XX), label='0.25')
plt.plot(XX, gam05.predict(XX), label='0.05')
plt.legend()

# %% [markdown]
# We fit the **mean model** by cross-validation in order to find the best smoothing parameter `lam` and then copy it over to the other models.
#
# This practice makes the expectiles less likely to cross. 

# %% [markdown]
# ## Custom Models
#
# It's also easy to build custom models by using the base **GAM** class and specifying the **distribution** and the **link function**:

# %%
from pygam import GAM
from pygam.datasets import trees

X, y = trees(return_X_y=True)

gam = GAM(distribution='gamma', link='log')
gam.gridsearch(X, y)

plt.scatter(y, gam.predict(X))
plt.xlabel('true volume')
plt.ylabel('predicted volume')


# %% [markdown]
# We can check the quality of the fit by looking at the Pseudo R-Squared:

# %%
gam.summary()

# %% [markdown]
# ## Penalties / Constraints
#
# With GAMs we can encode **prior knowledge** and **control overfitting** by using penalties and constraints.
#
# **Available penalties**
# - second derivative smoothing (default on numerical features)
# - L2 smoothing (default on categorical features)
#
# **Availabe constraints**
# - monotonic increasing/decreasing smoothing
# - convex/concave smoothing
# - periodic smoothing [soon...]
#
#
# We can inject our intuition into our model by using **monotonic** and **concave** constraints:
#

# %%
from pygam import LinearGAM, s
from pygam.datasets import hepatitis

X, y = hepatitis(return_X_y=True)

gam1 = LinearGAM(s(0, constraints='monotonic_inc')).fit(X, y)
gam2 = LinearGAM(s(0, constraints='concave')).fit(X, y)

fig, ax = plt.subplots(1, 2)
ax[0].plot(X, y, label='data')
ax[0].plot(X, gam1.predict(X), label='monotonic fit')
ax[0].legend()

ax[1].plot(X, y, label='data')
ax[1].plot(X, gam2.predict(X), label='concave fit')
ax[1].legend()

# %% [markdown]
# ## API
#
# pyGAM is intuitive, modular, and adheres to a familiar API:

# %%
from pygam import LogisticGAM, s, f
from pygam.datasets import toy_classification

X, y = toy_classification(return_X_y=True, n=5000)

gam = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4) + f(5))
gam.fit(X, y)

# %% [markdown]
# Since GAMs are additive, it is also super easy to visualize each individual **feature function**, `f_i(X_i)`. These feature functions describe the effect of each `X_i` on `y` individually while marginalizing out all other predictors:
#

# %%
plt.figure()
for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue
    plt.plot(gam.partial_dependence(term=i))


# %% [markdown]
# ## Current Features
#
# ### Models
# pyGAM comes with many models out-of-the-box:
#
# - GAM (base class for constructing custom models)
# - LinearGAM
# - LogisticGAM
# - GammaGAM
# - PoissonGAM
# - InvGaussGAM
# - ExpectileGAM

# %% [markdown]
# ### Terms
# - `l()` linear terms
# - `s()` spline terms
# - `f()` factor terms
# - `te()` tensor products
# - `intercept`

# %% [markdown]
# ### Distributions
#
# - Normal
# - Binomial
# - Gamma
# - Poisson
# - Inverse Gaussian
#
# ### Link Functions
# Link functions take the distribution mean to the linear prediction. These are the canonical link functions for the above distributions:
#
# - Identity
# - Logit
# - Inverse
# - Log
# - Inverse-squared
#
# ### Callbacks
# Callbacks are performed during each optimization iteration. It's also easy to write your own.
#
# - deviance - model deviance
# - diffs - differences of coefficient norm
# - accuracy - model accuracy for LogisticGAM
# - coef - coefficient logging
#
# You can check a callback by inspecting:
#

# %%
_ = plt.plot(gam.logs_['deviance'])

# %% [markdown]
# ### Linear Extrapolation

# %%
from pygam import LinearGAM
from pygam.datasets import mcycle

X, y = mcycle()

gam = LinearGAM()
gam.gridsearch(X, y)

XX = gam.generate_X_grid(term=0)

m = X.min()
M = X.max()
XX = np.linspace(m - 10, M + 10, 500)
Xl = np.linspace(m - 10, m, 50)
Xr = np.linspace(M, M + 10, 50)

plt.figure()

plt.plot(XX, gam.predict(XX), 'k')
plt.plot(Xl, gam.confidence_intervals(Xl), color='b', ls='--')
plt.plot(Xr, gam.confidence_intervals(Xr), color='b', ls='--')
_ = plt.plot(X, gam.confidence_intervals(X), color='r', ls='--')

# %% [markdown]
# ## References
# 1. Simon N. Wood, 2006  
# Generalized Additive Models: an introduction with R
#
# 0. Hastie, Tibshirani, Friedman  
# The Elements of Statistical Learning  
# http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf  
#
# 0. James, Witten, Hastie and Tibshirani  
# An Introduction to Statistical Learning  
# http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf  
#
# 0. Paul Eilers & Brian Marx, 1996
# Flexible Smoothing with B-splines and Penalties
# http://www.stat.washington.edu/courses/stat527/s13/readings/EilersMarx_StatSci_1996.pdf
#
# 0. Kim Larsen, 2015  
# GAM: The Predictive Modeling Silver Bullet  
# http://multithreaded.stitchfix.com/assets/files/gam.pdf  
#
# 0. Deva Ramanan, 2008  
# UCI Machine Learning: Notes on IRLS  
# http://www.ics.uci.edu/~dramanan/teaching/ics273a_winter08/homework/irls_notes.pdf  
#
# 0. Paul Eilers & Brian Marx, 2015  
# International Biometric Society: A Crash Course on P-splines  
# http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
#
# 0. Keiding, Niels, 1991  
# Age-specific incidence and prevalence: a statistical perspective
#
