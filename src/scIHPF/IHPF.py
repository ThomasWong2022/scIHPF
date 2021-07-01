#!/usr/bin/python

import functools
import ctypes
import numpy as np
from scipy.sparse import coo_matrix
import numba
from numba.extending import get_cython_function_address as getaddr


from copy import deepcopy
from warnings import warn

import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import digamma, gammaln, psi

try:
    from scipy.misc import logsumexp
except ImportError:
    from scipy.special import logsumexp

from sklearn.base import BaseEstimator

# get numba-compatible digamma/psi and gammaln
# psi/digamma
psi_fnaddr = getaddr("scipy.special.cython_special", "__pyx_fuse_1psi")
psi_ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
psi = psi_ftype(psi_fnaddr)
# gammaln
gammaln_fnaddr = getaddr("scipy.special.cython_special", "gammaln")
gammaln_ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
cgammaln = gammaln_ftype(gammaln_fnaddr)


# Compute poisson likelihood for a batch of datasets
@numba.njit(parallel=True, nogil=True, fastmath=True)
def compute_pois_llh2(
    X_data,
    X_row,
    X_col,
    theta_vi_shape,
    theta_vi_rate,
    beta_vi_shape,
    beta_vi_rate,
    delta_vi_shape,
    delta_vi_rate,
):
    ncells, ngenes = (theta_vi_shape.shape[0], beta_vi_shape.shape[0])
    nfactors, nnz = (theta_vi_shape.shape[1], X_data.shape[0])
    dtype = theta_vi_shape.dtype

    # precompute expectations
    theta_e_x = np.zeros_like(theta_vi_shape, dtype=dtype)
    for i in numba.prange(ncells):
        for k in range(nfactors):
            theta_e_x[i, k] = theta_vi_shape[i, k] / theta_vi_rate[i, k]

    beta_e_x = np.zeros_like(beta_vi_shape, dtype=dtype)
    for i in numba.prange(ngenes):
        for k in range(nfactors):
            beta_e_x[i, k] = beta_vi_shape[i, k] / beta_vi_rate[i, k]

    delta_e_x = np.zeros_like(delta_vi_shape, dtype=dtype)
    for i in numba.prange(ngenes):
        for k in range(nfactors):
            delta_e_x[i, k] = delta_vi_shape[i, k] / delta_vi_rate[i, k]

    # compute llh
    llh = np.zeros(X_data.shape, dtype=dtype)
    for i in numba.prange(nnz):
        e_rate = np.zeros(1, dtype=dtype)[0]
        for k in range(nfactors):
            e_rate += theta_e_x[X_row[i], k] * (
                beta_e_x[X_col[i], k] + delta_e_x[X_col[i], k]
            )
        llh[i] = X_data[i] * np.log(e_rate) - e_rate - cgammaln(X_data[i] + 1.0)
    return llh


@numba.njit(parallel=True, nogil=True)
def compute_Xphi_data(
    X_data,
    X_row,
    X_col,
    theta_vi_shape,
    theta_vi_rate,
    beta_vi_shape,
    beta_vi_rate,
    delta_vi_shape,
    delta_vi_rate,
):
    """Fast version of Xphi computation using numba & gsl_digamma

    Parameters
    ----------
    X_data : ndarray of np.int32
        (number_nonzero, ) array of nonzero values
    X_row : ndarray of np.int32
        (number_nonzero, ) array of row ids for each nonzero value
    X_col : ndarray (np.int32)
        (number_nonzero, ) array of column ids for each nonzero value
    theta_vi_shape : ndarray
        (ncells, nfactors) array of values for theta's variational shape
    theta_vi_rate : ndarray
        (ncells, nfactors) array of values for theta's variational rate
    beta_vi_shape : ndarray
        (ngenes, nfactors) array of values for beta's variational shape
    beta_vi_rate : ndarray
        (ngenes, nfactors) array of values for beta's variational rate
    """
    # convenience
    ncells, ngenes = (theta_vi_shape.shape[0], beta_vi_shape.shape[0])
    nfactors, nnz = (theta_vi_shape.shape[1], X_data.shape[0])
    dtype = theta_vi_shape.dtype

    # precompute theta.e_logx
    theta_e_logx = np.zeros_like(theta_vi_shape, dtype=dtype)
    for i in numba.prange(ncells):
        for k in range(nfactors):
            theta_e_logx[i, k] = psi(theta_vi_shape[i, k]) - np.log(theta_vi_rate[i, k])

    # precompute beta.e_logx
    beta_e_logx = np.zeros_like(beta_vi_shape, dtype=dtype)
    for i in numba.prange(ngenes):
        for k in range(nfactors):
            beta_e_logx[i, k] = psi(beta_vi_shape[i, k]) - np.log(beta_vi_rate[i, k])

    # precompute delta.e_logx
    delta_e_logx = np.zeros_like(delta_vi_shape, dtype=dtype)
    for i in numba.prange(ngenes):
        for k in range(nfactors):
            delta_e_logx[i, k] = psi(delta_vi_shape[i, k]) - np.log(delta_vi_rate[i, k])

    # compute Xphi
    # Scale shared and specific factors to create sparse factors
    Xphi = np.zeros((X_row.shape[0], theta_e_logx.shape[1] * 2), dtype=dtype)
    for i in numba.prange(nnz):
        logrho = np.zeros((Xphi.shape[1]), dtype=dtype)
        for k in range(nfactors):
            logrho[k] = (theta_e_logx[X_row[i], k] + beta_e_logx[X_col[i], k]) * 1
        for k in range(nfactors):
            logrho[k + nfactors] = (
                theta_e_logx[X_row[i], k] + delta_e_logx[X_col[i], k]
            ) * 1

        # log normalizer trick
        rho_shift = np.zeros((Xphi.shape[1]), dtype=dtype)
        normalizer = np.zeros(1, dtype=dtype)[0]
        largest_in = np.max(logrho)
        for k in range(nfactors * 2):
            rho_shift[k] = np.exp(logrho[k] - largest_in)
            normalizer += rho_shift[k]

        for k in range(nfactors * 2):
            Xphi[i, k] = X_data[i] * rho_shift[k] / normalizer

    return Xphi


@numba.njit(fastmath=True)  # results unstable with prange. don't do it.
def compute_loading_shape_update(Xphi_data, X_keep, nkeep, nfactors, shape_prior):
    """Compute gamma shape updates for theta or beta using numba

    Parameters
    ----------
    Xphi_data : ndarray
        (number_nonzero, nfactors) array of X * phi
    X_keep : ndarray
        (number_nonzer,) vector of indices along the axis of interest.
        If X is an (ncell,ngene) coo_matrix, this should be X.row when
        computing updates for theta and X.col when computing updates for
        beta
    nkeep : int
        Number of items on the axis of interest.  ncells when computing
        updates for theta, and ngenes for updates for beta
    shape_prior : float
        Hyperprior for parameter. a for theta, c for beta.

    """
    nnz = Xphi_data.shape[0]
    dtype = Xphi_data.dtype

    result = shape_prior * np.ones((nkeep, nfactors), dtype=dtype)

    for i in range(nnz):
        ikeep = X_keep[i]
        for k in range(nfactors):
            result[ikeep, k] += Xphi_data[i, k + nfactors]

    return result


@numba.njit(fastmath=True)
def compute_loading_rate_update(
    prior_vi_shape,
    prior_vi_rate,
    other_loading_vi_shape,
    other_loading_vi_rate,
):
    # shorter names
    pvs, pvr = (prior_vi_shape, prior_vi_rate)
    olvs, olvr = (other_loading_vi_shape, other_loading_vi_rate)
    dtype = prior_vi_shape.dtype

    other_loading_e_x_sum = np.zeros((olvs.shape[1]), dtype=dtype)
    for i in range(olvs.shape[0]):
        for k in range(olvs.shape[1]):
            other_loading_e_x_sum[k] += olvs[i, k] / olvr[i, k]

    result = np.zeros((pvs.shape[0], olvs.shape[1]), dtype=dtype)

    for i in range(pvs.shape[0]):
        prior_e_x = pvs[i] / pvr[i]
        for k in range(olvs.shape[1]):
            result[i, k] = prior_e_x + other_loading_e_x_sum[k]
    return result


@numba.njit(fastmath=True)
def compute_capacity_rate_update(loading_vi_shape, loading_vi_rate, prior_rate):
    dtype = loading_vi_shape.dtype
    result = prior_rate * np.ones((loading_vi_shape.shape[0],), dtype=dtype)
    for k in range(loading_vi_shape.shape[1]):
        for i in range(loading_vi_shape.shape[0]):
            result[i] += loading_vi_shape[i, k] / loading_vi_rate[i, k]
    return result


@numba.njit(fastmath=True)  # results unstable with prange. don't do it.
def compute_cell_shape_update(Xphi_data, X_keep, nkeep, nfactors, shape_prior):
    """Compute gamma shape updates for theta or beta using numba

    Parameters
    ----------
    Xphi_data : ndarray
        (number_nonzero, nfactors) array of X * phi
    X_keep : ndarray
        (number_nonzer,) vector of indices along the axis of interest.
        If X is an (ncell,ngene) coo_matrix, this should be X.row when
        computing updates for theta and X.col when computing updates for
        beta
    nkeep : int
        Number of items on the axis of interest.  ncells when computing
        updates for theta, and ngenes for updates for beta
    shape_prior : float
        Hyperprior for parameter. a for theta, c for beta.

    """
    nnz = Xphi_data.shape[0]
    dtype = Xphi_data.dtype

    result = shape_prior * np.ones((nkeep, nfactors), dtype=dtype)
    for i in range(nnz):
        ikeep = X_keep[i]
        for k in range(nfactors):
            result[ikeep, k] = (
                result[ikeep, k] + Xphi_data[i, k] + Xphi_data[i, k + nfactors]
            )
    return result


@numba.njit(fastmath=True)  # results unstable with prange. don't do it.
def compute_shared_shape_update(
    result, Xphi_data, X_keep, nkeep, nfactors, shape_prior
):
    """Compute gamma shape updates for theta or beta using numba

    Parameters
    ----------
    Xphi_data : List of ndarray
        List of (number_nonzero, nfactors) array of X * phi
    X_keep : ndarray
        (number_nonzer,) vector of indices along the axis of interest.
        If X is an (ncell,ngene) coo_matrix, this should be X.row when
        computing updates for theta and X.col when computing updates for
        beta
    nkeep : int
        Number of items on the axis of interest.  ncells when computing
        updates for theta, and ngenes for updates for beta
    shape_prior : float
        Hyperprior for parameter. a for theta, c for beta.

    """
    nnz = Xphi_data.shape[0]
    dtype = Xphi_data.dtype
    for i in range(nnz):
        ikeep = X_keep[i]
        for k in range(nfactors):
            result[ikeep, k] = result[ikeep, k] + Xphi_data[i, k]
    return result


@numba.njit(fastmath=True)
def compute_cell_rate_update(
    prior_vi_shape,
    prior_vi_rate,
    other_loading_vi_shape,
    other_loading_vi_rate,
    additional_loading_vi_shape,
    additional_loading_vi_rate,
):
    # shorter names
    pvs, pvr = (prior_vi_shape, prior_vi_rate)
    olvs, olvr = (other_loading_vi_shape, other_loading_vi_rate)
    alvs, alvr = (additional_loading_vi_shape, additional_loading_vi_rate)
    dtype = prior_vi_shape.dtype

    other_loading_e_x_sum = np.zeros((olvs.shape[1]), dtype=dtype)
    for i in range(olvs.shape[0]):
        for k in range(olvs.shape[1]):
            other_loading_e_x_sum[k] += olvs[i, k] / olvr[i, k]

    additional_loading_e_x_sum = np.zeros((alvs.shape[1]), dtype=dtype)
    for i in range(alvs.shape[0]):
        for k in range(alvs.shape[1]):
            additional_loading_e_x_sum[k] += alvs[i, k] / alvr[i, k]

    result = np.zeros((pvs.shape[0], olvs.shape[1]), dtype=dtype)
    for i in range(pvs.shape[0]):
        prior_e_x = pvs[i] / pvr[i]
        for k in range(olvs.shape[1]):
            result[i, k] = (
                prior_e_x + other_loading_e_x_sum[k] + additional_loading_e_x_sum[k]
            )
    return result


@numba.njit(fastmath=True)
def compute_shared_rate_update(
    result,
    prior_vi_shape,
    prior_vi_rate,
    other_loading_vi_shape,
    other_loading_vi_rate,
    firstupdate,
):
    # shorter names
    pvs, pvr = (prior_vi_shape, prior_vi_rate)
    olvs, olvr = (other_loading_vi_shape, other_loading_vi_rate)
    dtype = prior_vi_shape.dtype

    other_loading_e_x_sum = np.zeros((olvs.shape[1]), dtype=dtype)

    for i in range(olvs.shape[0]):
        for k in range(olvs.shape[1]):
            other_loading_e_x_sum[k] += olvs[i, k] / olvr[i, k]

    for i in range(pvs.shape[0]):
        prior_e_x = pvs[i] / pvr[i]
        if firstupdate:
            for k in range(olvs.shape[1]):
                result[i, k] = prior_e_x + other_loading_e_x_sum[k]
        else:
            for k in range(olvs.shape[1]):
                result[i, k] = result[i, k] + other_loading_e_x_sum[k]
    return result


"""
Loss functions and higher order functions that return loss functions for a
given dataset

"""


def loss_function_for_data(loss_function, X):
    """Get a loss function for a fixed dataset

    Parameters
    ----------
    loss_function : function
        The loss function to use.  The data parameter for the function must
        be `X`
    X : coo_matrix
        coo_matrix of data to apply loss function to

    Returns
    -------
    fixed_data_loss_function : function
        A loss function which takes all the same parameters as the input
        `loss_function`, except for the data parameter `X` which is fixed
    """
    return functools.partial(loss_function, X=X)


#### Loss functions


def pois_llh_pointwise(X, *, theta, beta, **kwargs):
    """Poisson log-likelihood for each nonzero entry

    Parameters
    ----------
    X: coo_matrix
        Data to compute Poisson log likelihood of. Assumed to be nonzero.
    theta : HPF_Gamma
    beta : HPF_Gamma
    **kwargs : dict, optional
        extra arguments not used in this loss function

    Returns
    -------
    llh: ndarray


    Note
    ----
    Like all loss functions in this module, all parameters except from data
    must be passed to the function as a keyword argument, and the function
    will accept unused keyword args.
    """
    try:
        llh = compute_pois_llh(
            X.data,
            X.row,
            X.col,
            theta.vi_shape,
            theta.vi_rate,
            beta.vi_shape,
            beta.vi_rate,
        )
    except NameError:
        e_rate = (theta.e_x[X.row] * beta.e_x[X.col]).sum(axis=1)
        llh = X.data * np.log(e_rate) - e_rate - gammaln(X.data + 1)
    return llh


def mean_negative_pois_llh(X, *, theta, beta, **kwargs):
    """Mean Poisson log-likelihood for each nonzero entry

    Parameters
    ----------
    X: coo_matrix
        Data to compute Poisson log likelihood of. Assumed to be nonzero.
    theta : HPF_Gamma
    beta : HPF_Gamma
    **kwargs : dict, optional
        extra arguments not used in this loss function

    Returns
    -------
    llh: ndarray


    Note
    ----
    Like all loss functions in this module, all parameters except from data
    must be passed to the function as a keyword argument, and the function
    will accept unused keyword args.
    """
    return np.mean(-pois_llh_pointwise(X=X, theta=theta, beta=beta))


def pois_llh_pointwise2(X, *, theta, beta, delta, datasetno, **kwargs):
    """Poisson log-likelihood for each nonzero entry

    Parameters
    ----------
    X: coo_matrix
        Data to compute Poisson log likelihood of. Assumed to be nonzero.
    theta : HPF_Gamma
    beta : HPF_Gamma
    **kwargs : dict, optional
        extra arguments not used in this loss function

    Returns
    -------
    llh: ndarray


    Note
    ----
    Like all loss functions in this module, all parameters except from data
    must be passed to the function as a keyword argument, and the function
    will accept unused keyword args.
    """
    try:
        llh = compute_pois_llh2(
            X[datasetno].data,
            X[datasetno].row,
            X[datasetno].col,
            theta.vi_shape,
            theta.vi_rate,
            beta.vi_shape,
            beta.vi_rate,
            delta.vi_shape,
            delta.vi_rate,
        )
    except NameError:
        e_rate = (theta.e_x[X.row] * (beta.e_x[X.col] + delta.e_x[X.col])).sum(axis=1)
        llh = X.data * np.log(e_rate) - e_rate - gammaln(X.data + 1)
    return llh


def mean_negative_pois_llh2(X, *, theta, beta, delta, datasetno, **kwargs):
    """Mean Poisson log-likelihood for each nonzero entry

    Parameters
    ----------
    X: coo_matrix
        Data to compute Poisson log likelihood of. Assumed to be nonzero.
    theta : HPF_Gamma
    beta : HPF_Gamma
    **kwargs : dict, optional
        extra arguments not used in this loss function

    Returns
    -------
    llh: ndarray


    Note
    ----
    Like all loss functions in this module, all parameters except from data
    must be passed to the function as a keyword argument, and the function
    will accept unused keyword args.
    """
    return np.mean(
        -pois_llh_pointwise2(
            X=X, theta=theta, beta=beta, delta=delta, datasetno=datasetno
        )
    )


class HPF_Gamma(object):
    """Gamma variational distributions

    Parameters
    ----------
    vi_shape: np.ndarray
        Gamma shape parameter for the variational Gamma distributions.
        Ndarray.shape[0] must match `vi_rate`
    vi_rate: np.ndarray
        Gamma rate parameter for the variational Gamma distributions.
        Ndarray.shape[0] must match `vi_shape`

    Attributes
    ----------
    vi_shape : ndarray
    vi_rate : ndarray
    dims : ndarray
        The shape of vi_shape and vi_rate
    dtype : dtype
        dtype of both vi_shape and vi_rate
    """

    @staticmethod
    def random_gamma_factory(dims, shape_prior, rate_prior, dtype=np.float64):
        """Factory method to randomly initialize variational distributions

        Parameters
        ----------
        dims: list-like
            Numpy-style shape of the matrix of Gammas.
        shape_prior: float
            Prior for variational Gammas' shapes.  Must be greater than 0.
        rate_prior: float
            Prior for variational Gammas' rates.  Must be greater than 0.

        Returns
        -------
            A randomly initialized HPF_Gamma instance
        """
        vi_shape = np.random.uniform(0.5 * shape_prior, 1.5 * shape_prior, dims).astype(
            dtype
        )
        vi_rate = np.random.uniform(0.5 * rate_prior, 1.5 * rate_prior, dims).astype(
            dtype
        )
        return HPF_Gamma(vi_shape, vi_rate)

    def __init__(self, vi_shape, vi_rate):
        """Initializes HPF_Gamma with variational shape and rates"""
        assert vi_shape.shape == vi_rate.shape
        assert vi_shape.dtype == vi_rate.dtype
        assert np.all(vi_shape > 0)
        assert np.all(vi_rate > 0)
        self.vi_shape = vi_shape
        self.vi_rate = vi_rate
        self.dtype = vi_shape.dtype

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            shape_equal = np.array_equal(self.vi_shape, other.vi_shape)
            rate_equal = np.array_equal(self.vi_rate, other.vi_rate)
            dtype_equal = self.dtype == other.dtype
            return shape_equal and rate_equal and dtype_equal
        return False

    @property
    def dims(self):
        assert self.vi_shape.shape == self.vi_rate.shape
        return self.vi_shape.shape

    @property
    def e_x(self):
        """Expected value of the random variable(s) given variational
        distribution(s)
        """
        return self.vi_shape / self.vi_rate

    @property
    def e_logx(self):
        """Expectation of the log of random variable given variational
        distribution(s)"""
        return digamma(self.vi_shape) - np.log(self.vi_rate)

    @property
    def entropy(self):
        """Entropy of variational Gammas"""
        return (
            self.vi_shape
            - np.log(self.vi_rate)
            + gammaln(self.vi_shape)
            + (1 - self.vi_shape) * digamma(self.vi_shape)
        )

    def sample(self, nsamples=1):
        """Sample from variational distributions

        Parameters
        ----------
        nsamples: int (optional, default 1)
            Number of samples to take.

        Returns
        -------
        X_rep : np.ndarray
            An ndarray of samples from the variational distributions, where
            the last dimension is the number of samples `nsamples`
        """
        samples = []
        for i in range(nsamples):
            samples.append(np.random.gamma(self.vi_shape, 1 / self.vi_rate).T)
        return np.stack(samples).T

    def combine(self, other, other_ixs):
        """Combine with another HPF_Gamma

        Useful for combining variational distributions from training data with
        variational distributions from cells that were projected onto frozen
        beta and eta

        Parameters
        ----------
        other : `HPF_Gamma`
            Other HPF_Gamma to merge with
        other_ixs : list or ndarray
            Ordered indices of other in the merged HPF_Gamma. Must have len
            equal to other.shape[0]. Must have a maximum value less than
            self.dims[0] + other.shape[0]. May not have any repeat values.

        Returns
        -------
        combined_model : `HPF_Gamma`
        """
        assert other.dims[0] == len(other_ixs)
        assert len(np.unique(other_ixs)) == len(other_ixs)
        assert self.dims[0] + other.dims[0] > np.max(other_ixs)

        new_dims = [self.dims[0] + other.dims[0], *self.dims[1:]]
        self_ixs = np.setdiff1d(np.arange(new_dims[0]), other_ixs)

        new_vi_shape = np.empty(new_dims, dtype=self.dtype)
        new_vi_shape[self_ixs] = self.vi_shape
        new_vi_shape[other_ixs] = other.vi_shape

        new_vi_rate = np.empty(new_dims, dtype=self.dtype)
        new_vi_rate[self_ixs] = self.vi_rate
        new_vi_rate[other_ixs] = other.vi_rate

        return HPF_Gamma(new_vi_shape, new_vi_rate)


class scIHPF(BaseEstimator):
    """scHPF components which are the building blocks for integrative HPF
    Parameters
    ----------
    nfactors: int
        Number of factors (K)
    a: float, (optional, default 0.3)
        Hyperparameter a
    ap: float (optional, default 1.0)
        Hyperparameter a'
    bp: float (optional, default None)
        Hyperparameter b'. Set empirically from observed data if not
        given.
    c: float, (optional, default 0.3)
        Hyperparameter c
    cp: float (optional, default 1.0)
        Hyperparameter c'
    dp: float (optional, default None)
        Hyperparameter d'. Set empirically from observed data if not
        given.
    min_iter: int (optional, default 30):
        Minimum number of interations for training.
    max_iter: int (optional, default 1000):
        Maximum number of interations for training.
    check_freq: int (optional, default 10)
        Number of training iterations between calculating loss.
    epsilon: float (optional, default 0.001)
        Percent change of loss for convergence.
    better_than_n_ago: int (optional, default 5)
        Stop condition if loss is getting worse.  Stops training if loss
        is worse than `better_than_n_ago`*`check_freq` training steps
        ago and getting worse.
    xi: HPF_Gamma (optional, default None)
        Variational distributions for xi
    theta: HPF_Gamma (optional, default None)
        Variational distributions for theta
    eta: HPF_Gamma (optional, default None)
        Variational distributions for eta
    beta: HPF_Gamma (optional, default None)
        Variational distributions for beta
    delta: HPF_Gamma (optional, default None)
        Variational distributions for beta
    verbose: bool (optional, default True)
            Print messages at each check_freq
    """

    def __init__(
        self,
        nfactors,
        a=0.3,
        ap=1,
        bp=None,
        c=0.3,
        cp=1,
        dp=None,
        min_iter=30,
        max_iter=500,
        check_freq=10,
        epsilon=0.001,
        better_than_n_ago=5,
        dtype=np.float64,
        xi=None,
        theta=None,
        eta=None,
        beta=None,
        zeta=None,
        delta=None,
        loss=[],
        verbose=True,
        dataset_ratio=0.1,
    ):
        """Initialize HPF instance"""
        self.nfactors = nfactors
        self.a = a
        self.ap = ap
        self.bp = bp
        self.c = c
        self.cp = cp
        self.dp = dp
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.check_freq = check_freq
        self.epsilon = epsilon
        self.better_than_n_ago = better_than_n_ago
        self.dtype = dtype
        self.verbose = verbose

        self.xi = xi
        self.eta = eta
        self.zeta = zeta

        self.theta = theta
        self.beta = beta
        self.delta = delta

        self.loss = []

        # np.random.seed(0)

        self.dataset_ratio = dataset_ratio

    @property
    def ngenes(self):
        return self.eta.dims[0] if self.eta is not None else None

    @property
    def ncells(self):
        return self.xi.dims[0] if self.xi is not None else None

    def cell_scores(self):
        cells = []
        for k in range(len(self.xi)):
            temp = self.xi[k].e_x[:, None] * self.theta[k].e_x
            cells.append(temp)
        return cells

    def shared_gene_scores(self):
        return self.eta.e_x[:, None] * self.beta.e_x

    def gene_scores(self):
        genes = []
        for k in range(len(self.zeta)):
            temp = self.zeta[k].e_x[:, None] * self.delta[k].e_x
            genes.append(temp)
        return genes

    def pois_llh_pointwise(self, X, theta=None, beta=None):
        """Poisson log-likelihood (for each nonzero data)

        Attempt to use numba/cffi/gsl, use numpy otherwise

        Parameters
        ----------
        X: coo_matrix
            Data to compute Poisson log likelihood of. Assumed to be nonzero.
        theta : HPF_Gamma, optional
            If given, use for theta instead of self.theta
        beta : HPF_Gamma, optional
            If given, use for beta instead of self.beta

        Returns
        -------
        llh: ndarray
        """
        theta = self.theta if theta is None else theta
        beta = self.beta if beta is None else beta
        return ls.pois_llh_pointwise(X=X, theta=theta, beta=beta)

    def mean_negative_pois_llh(X, theta=None, beta=None, **kwargs):
        """Convenience method for mean negative llh of nonzero entries"""
        theta = self.theta if theta is None else theta
        beta = self.beta if beta is None else beta
        return ls.mean_negative_pois_llh(X=X, theta=theta, beta=beta)

    def fit(self, X, **kwargs):
        """Fit an scHPF model

        Parameters
        ----------
        X: coo_matrix
            Data to fit
        loss_function : function, optional (Default: None)
            loss function to use for fit. set to negative poisson likelihood
            of X if not given
        """
        (bp, dp, xi, eta, zeta, theta, beta, delta, loss) = self._fit(X, **kwargs)
        self.bp = bp
        self.dp = dp
        self.xi = xi
        self.eta = eta
        self.zeta = zeta
        self.theta = theta
        self.beta = beta
        self.delta = delta
        self.loss = loss
        return self

    def _score(self, capacity, loading):
        """Get the hierarchically normalized loadings which we call the cell
        or gene score in the scHPF paper

        Parameters
        ----------
        capacity : HPF_Gamma
            xi or eta
        loading : HPF_Gamma
            theta or beta


        Returns
        -------
        score : ndarray
        """
        assert loading.dims[0] == capacity.dims[0]
        return loading.e_x * capacity.e_x[:, None]

    def _fit(
        self,
        X,
        freeze_shared_genes=False,
        freeze_genes=False,
        reinit=True,
        loss_function=None,
        min_iter=None,
        max_iter=None,
        epsilon=None,
        check_freq=None,
        checkstep_function=None,
        dataset_ratio=0.1,
        verbose=None,
    ):
        """Combined internal fit/transform function

        Parameters
        ----------
        X: List of coo_matrix
            List of Data to fit
        freeze_genes: bool, (optional, default False)
            Should we update gene variational distributions eta and beta
        reinit: bool, (optional, default True)
            Randomly initialize variational distributions even if they
            already exist. Superseded by freeze_genes. Does not affect
            self.bp and self.dp which will only be set empirically if they
            are None
        loss_function : function, (optional, default None)
            Function to use for loss, which is assumed to be nonzero and
            decrease with improvement. Must accept hyperparameters a, ap,
            bp, c, cp, and dp and the variational distributions for xi, eta,
            theta, and beta even if only some of these values are used.
            Should have an internal reference to any data used (_fit will
            not pass it any data). If `loss_function` is not given or equal
            to None, the mean negative log likelihood of nonzero values in
            training data `X` is used.
        min_iter: int (optional, default None)
            Replaces self.min_iter if given.  Useful when projecting
            new data onto an existing scHPF model.
        max_iter: int (optional, default None)
            Replaces self.max_iter if given.  Useful when projecting
            new data onto an existing scHPF model.
        epsilon: float (optional, default None)
            Replaces self.epsilon if given. Percent change of loss for
            convergence.
        check_freq : int, optional (Default: None)
            Replaces self.check_freq if given.  Useful when projecting
            new data onto an existing scHPF model.
        checkstep_function : function  (optional, default None)
            A function that takes arguments bp, dp, xi, eta, theta, beta,
            and t and, if given, is called at check_interval. Intended use
            is to check additional stats during training, potentially with
            hardcoded data, but is unrestricted.  Use at own risk.
        verbose: bool (optional, default None)
            If not None, overrides self.verbose

        Returns
        -------
        bp: list of float
            Empirically set value for bp
        dp: list of float
            Empirically set value for dp. Unchanged if freeze_genes.
        xi: list of HPF_Gamma
            Learned variational distributions for xi
        eta: HPF_Gamma
            Learned variational distributions for eta. Unchanged if
            freeze_genes.
        theta: list of HPF_Gamma
            Learned variational distributions for theta
        beta: HPF_Gamma
            Learned variational distributions for beta. Unchanged if
            freeze_genes.
        loss : list
            loss at each checkstep
        """
        # local (convenience) vars for model

        nfactors = self.nfactors
        ndatasets = len(X)
        a, ap, c, cp = self.a, self.ap, self.c, self.cp

        # get empirically set hyperparameters and variational distributions
        bp, dp, xi, eta, zeta, theta, beta, delta = self._setup(
            X, freeze_shared_genes, freeze_genes, reinit
        )

        # Make first updates for hierarchical shape prior
        # (vi_shape is constant, but want to update full distribution)

        for i in range(ndatasets):
            xi[i].vi_shape[:] = ap + nfactors * a
            if not freeze_genes:
                delta[i].vi_shape[:] = cp + nfactors * c

        if not freeze_shared_genes:
            eta.vi_shape[:] = cp + nfactors * c

        # setup loss function as mean negative llh of nonzero training data
        # if the loss function is not given
        if loss_function is None:
            loss_function = loss_function_for_data(mean_negative_pois_llh2, X)

        ## init
        loss, pct_change = [], []
        # check variable overrides
        min_iter = self.min_iter if min_iter is None else min_iter
        max_iter = self.max_iter if max_iter is None else max_iter
        epsilon = self.epsilon if epsilon is None else epsilon
        check_freq = self.check_freq if check_freq is None else check_freq
        verbose = self.verbose if verbose is None else verbose

        for t in range(max_iter):

            # Compute X Phi for each batch
            if t == 0 and reinit:  # randomize phi for first iteration
                Xphi_data = []
                for i in range(ndatasets):
                    random_phi = np.random.dirichlet(
                        np.hstack(
                            (np.ones(nfactors), np.ones(nfactors) * dataset_ratio)
                        ),
                        X[i].data.shape[0],
                    )
                    Xphi_data.append(X[i].data[:, None] * random_phi)
            else:
                # For each batch compute X phi
                Xphi_data = []
                for i in range(ndatasets):
                    Xphi_data_temp = compute_Xphi_data(
                        X[i].data,
                        X[i].row,
                        X[i].col,
                        theta[i].vi_shape,
                        theta[i].vi_rate,
                        beta.vi_shape,
                        beta.vi_rate,
                        delta[i].vi_shape,
                        delta[i].vi_rate,
                    )
                    Xphi_data.append(Xphi_data_temp)

            ngenes = beta.vi_shape.shape[0]
            dtype = Xphi_data[0].dtype

            # shared gene updates (if not frozen)
            if not freeze_shared_genes:
                beta.vi_shape = c * np.ones((ngenes, nfactors), dtype=dtype)
                beta.vi_rate = np.zeros((ngenes, nfactors), dtype=dtype)
                for k in range(ndatasets):
                    if k == 0:
                        firstiter = True
                    else:
                        firstiter = False
                    beta.vi_shape = compute_shared_shape_update(
                        beta.vi_shape, Xphi_data[k], X[k].col, ngenes, nfactors, c
                    )
                    beta.vi_rate = compute_shared_rate_update(
                        beta.vi_rate,
                        eta.vi_shape,
                        eta.vi_rate,
                        theta[k].vi_shape,
                        theta[k].vi_rate,
                        firstiter,
                    )
                eta.vi_rate = np.mean(dp) + beta.e_x.sum(1)

            # gene updates
            if not freeze_genes:
                for i in range(ndatasets):
                    delta[i].vi_shape = compute_loading_shape_update(
                        Xphi_data[i], X[i].col, ngenes, nfactors, c
                    )
                    delta[i].vi_rate = compute_loading_rate_update(
                        zeta[i].vi_shape,
                        zeta[i].vi_rate,
                        theta[i].vi_shape,
                        theta[i].vi_rate,
                    )
                    zeta[i].vi_rate = dp[i] + delta[i].e_x.sum(1)

            # cell updates
            for i in range(ndatasets):
                ncells = X[i].shape[0]
                theta[i].vi_shape = compute_cell_shape_update(
                    Xphi_data[i], X[i].row, ncells, nfactors, a
                )
                theta[i].vi_rate = compute_cell_rate_update(
                    xi[i].vi_shape,
                    xi[i].vi_rate,
                    beta.vi_shape,
                    beta.vi_rate,
                    delta[i].vi_shape,
                    delta[i].vi_rate,
                )
                xi[i].vi_rate = bp[i] + theta[i].e_x.sum(1)

            # record llh/percent change and check for convergence
            if t % check_freq == 0:

                # chech llh
                # vX = validation_data if validation_data is not None else X
                try:
                    curr = 0
                    for i in range(ndatasets):
                        temp = loss_function(
                            a=a,
                            ap=ap,
                            bp=bp[i],
                            c=c,
                            cp=cp,
                            dp=dp[i],
                            xi=xi[i],
                            eta=eta,
                            theta=theta[i],
                            beta=beta,
                            delta=delta[i],
                            datasetno=i,
                        )
                        curr += temp
                    loss.append(curr)
                except NameError as e:
                    print("Invalid loss function")
                    raise e

                # calculate percent change
                try:
                    prev = loss[-2]
                    pct_change.append(100 * (curr - prev) / np.abs(prev))
                except IndexError:
                    pct_change.append(100)
                if verbose:
                    msg = "[Iter. {0: >4}]  loss:{1:.6f}  pct:{2:.9f}".format(
                        t, curr, pct_change[-1]
                    )
                    print(msg)
                if checkstep_function is not None:
                    checkstep_function(
                        bp=bp, dp=dp, xi=xi, eta=eta, theta=theta, beta=beta, t=t
                    )

                # check convergence
                if len(loss) > 3 and t >= min_iter:
                    # convergence conditions (all must be met)
                    current_small = np.abs(pct_change[-1]) < self.epsilon
                    prev_small = np.abs(pct_change[-2]) < self.epsilon
                    not_inflection = not (
                        (np.abs(loss[-3]) < np.abs(prev))
                        and (np.abs(prev) > np.abs(curr))
                    )
                    converged = current_small and prev_small and not_inflection
                    if converged:
                        if verbose:
                            print("converged")
                        break

                    # getting worse, and has been for better_than_n_ago checks
                    # (don't waste time on a bad run)
                    if len(loss) > self.better_than_n_ago and self.better_than_n_ago:
                        nprev = loss[-self.better_than_n_ago]
                        worse_than_n_ago = np.abs(nprev) < np.abs(curr)
                        getting_worse = np.abs(prev) < np.abs(curr)
                        if worse_than_n_ago and getting_worse:
                            if verbose:
                                print("getting worse break")
                            break

            # TODO message or warning or something
            if t >= self.max_iter:
                break

        return (bp, dp, xi, eta, zeta, theta, beta, delta, loss)

    def _setup(
        self, X, freeze_shared_genes=False, freeze_genes=False, reinit=True, clip=True
    ):
        """Setup variational distributions

        Parameters
        ----------
        X : list of coo_matrix
            List of data to fit
        freeze_genes: bool, optional (Default: False)
            Should we update gene variational distributions eta and beta
        reinit: bool, optional (Default: True)
            Randomly initialize variational distributions even if they
            already exist. Superseded by freeze_genes. Does not affect
            self.bp and self.dp (which will only be set empirically if
            they are None)
        clip : bool, optional (Default: True)
            If empirically calculating dp and bp > 1000 * dp, clip dp to
            bp / 1000.

        Returns
        -------
        bp : List of float
        dp : List of float
        xi : List of HPF_Gamma
        eta : HPF_Gamma
        zeta : List of HPF_Gamma
        theta : List of HPF_Gamma
        beta : HPF_Gamma
        delta: List of HPF_Gamma

        """
        # locals for convenience
        nfactors = self.nfactors
        ngenes = X[0].shape[1]
        ndatasets = len(X)
        a, ap, c, cp = self.a, self.ap, self.c, self.cp
        bp, dp = self.bp, self.dp

        xi, eta, zeta, theta, beta, delta = (
            self.xi,
            self.eta,
            self.zeta,
            self.theta,
            self.beta,
            self.delta,
        )

        # empirically set bp and dp
        bp, dp = self._get_empirical_hypers(X, freeze_genes, clip)

        if reinit or (xi is None):
            xi = [
                HPF_Gamma.random_gamma_factory(
                    (X[k].shape[0],), ap, bp[k], dtype=self.dtype
                )
                for k in range(len(X))
            ]

        if reinit or (theta is None):
            theta = [
                HPF_Gamma.random_gamma_factory(
                    (X[k].shape[0], nfactors), a, bp[k], dtype=self.dtype
                )
                for k in range(len(X))
            ]

        # Check if variational distributions for genes exist, create if not
        # Error if freeze_genes and eta and beta don't exists
        if freeze_genes:
            if eta is None or beta is None:
                msg = (
                    "To fit with frozen gene variational distributions "
                    + "(`freeze_genes`==True), eta and beta must be set to "
                    + "valid HPF_Gamma instances."
                )
                raise ValueError(msg)

            if reinit or (zeta is None):
                zeta = [
                    HPF_Gamma.random_gamma_factory(
                        (ngenes,), cp, dp[i], dtype=self.dtype
                    )
                    for i in range(ndatasets)
                ]

            if reinit or (delta is None):
                delta = [
                    HPF_Gamma.random_gamma_factory(
                        (ngenes, nfactors), c, dp[i], dtype=self.dtype
                    )
                    for i in range(ndatasets)
                ]
        else:

            if reinit or (eta is None):
                eta = HPF_Gamma.random_gamma_factory(
                    (ngenes,), cp, np.mean(dp), dtype=self.dtype
                )

            if reinit or (beta is None):
                beta = HPF_Gamma.random_gamma_factory(
                    (ngenes, nfactors), c, np.mean(dp), dtype=self.dtype
                )

            if reinit or (zeta is None):
                zeta = [
                    HPF_Gamma.random_gamma_factory(
                        (ngenes,), cp, dp[i], dtype=self.dtype
                    )
                    for i in range(ndatasets)
                ]

            if reinit or (delta is None):
                delta = [
                    HPF_Gamma.random_gamma_factory(
                        (ngenes, nfactors), c, dp[i], dtype=self.dtype
                    )
                    for i in range(ndatasets)
                ]

        return (bp, dp, xi, eta, zeta, theta, beta, delta)

    def _get_empirical_hypers(self, X, freeze_genes=False, clip=False):
        """Get empirical values for bp, dp

        Parameters
        ----------
        X : list of coo_matrix
            List of data to fit

        Returns
        -------
        bp : list of float
        dp : list of float
        """
        bp, dp = self.bp, self.dp

        # empirically set bp and dp
        def mean_var_ratio(X, axis):
            axis_sum = X.sum(axis=axis)
            return np.mean(axis_sum) / np.var(axis_sum)

        if bp is None:
            bp = [self.ap * mean_var_ratio(X_data, axis=1) for X_data in X]

        if dp is None:  # dp first in case of error
            if freeze_genes:
                msg = "dp is None and cannot be set"
                msg += " when freeze_genes is True."
                raise ValueError(msg)
            else:
                dp = [self.cp * mean_var_ratio(X_data, axis=0) for X_data in X]

                if clip and bp > 1000 * dp:
                    old_val = dp
                    dp = [bpc / 1000.0 for bpc in bp]
                    print("Clipping dp: was {} now {}".format(old_val, dp))

        return bp, dp

    def _initialize(self, X, freeze_genes=False):
        """Shortcut to setup random distributions & set variables"""
        bp, dp, xi, eta, zeta, theta, beta, delta = self._setup(
            X, freeze_genes, reinit=True
        )
        self.bp = bp
        self.dp = dp
        self.xi = xi
        self.eta = eta
        self.zeta = zeta
        self.theta = theta
        self.beta = beta
        self.delta = delta


def combine_across_cells(x, y, y_ixs):
    """Combine theta & xi from two scHPF instance with the same beta & eta

    Intended to be used combining variational distributions for local
    variables (theta,xi) from training data with variational distributions
    for local variables from validation or other data that was projected
    onto the same global variational distributions (beta,eta)

    If `x.bp` != `y.bp`, returned model `xy.bp` is set to None. All other
    attributes (except for the merged xi and eta) are inherited from `x`.

    Parameters
    ----------
    x : `scHPF`
    y : `scHPF`
        The scHPF instance whose rows in the output should be at the
        corresponding indices `y_ixs`
    y_ixs : ndarray
        Row indices of `y` in the returned distributions. Must be 1-d and
        have same number of rows as `y`, have no repeats, and have no index
        greater than or equal to x.ncells + y.ncells.


    Returns
    -------
    ab : `scHPF`

    """
    assert x.dp == y.dp
    assert x.eta == y.eta
    assert x.beta == y.beta

    xy = deepcopy(x)
    if y.bp != x.bp:
        xy.bp = None
    xy.xi = x.xi.combine(y.xi, y_ixs)
    xy.theta = x.theta.combine(y.theta, y_ixs)
    return xy


def run_trials(
    X,
    nfactors,
    ntrials=15,
    min_iter=30,
    max_iter=500,
    check_freq=10,
    epsilon=0.001,
    better_than_n_ago=5,
    dtype=np.float64,
    verbose=True,
    vcells=None,
    vX=None,
    loss_function=None,
    model_kwargs={},
):
    """
    Train with multiple random initializations, selecting model with best loss

    As scHPF uses non-convex optimization, it benefits from training with
    multiple random initializations to avoid local minima.

    Parameters
    ----------
    X: coo_matrix
        Data to fit
    nfactors: int
        Number of factors (K)
    ntrials : int,  optional (Default 5)
        Number of random initializations for training
    min_iter: int, optional (Default 30)
        Minimum number of interations for training.
    max_iter: int, optional (Default 1000):
        Maximum number of interations for training.
    check_freq: int, optional (Default 10)
        Number of training iterations between calculating loss.
    epsilon: float, optional (Default 0.001)
        Percent change of loss for convergence.
    better_than_n_ago: int, optional (Default 5)
        Stop condition if loss is getting worse.  Stops training if loss
        is worse than `better_than_n_ago`*`check_freq` training steps
        ago and getting worse.
    dtype : datatype, optional (Default np.float64)
        np.float64 or np.float32
    verbose: bool, optional (Default True)
        verbose
    vcells : coo_matrix, optional (Default None)
        cells to use in a validation loss function
    vX : coo_matrix, optional (Default None)
        nonzero entries from the cells in vX
    loss_function : function, optional (Default None)
        A loss function that accepts data, model variational parameters,
        and model hyperparameters.  Note this is distinct from the
        `loss_function` argument in scHPF._fit (called by scHPF.fit and
        scHPF.project), which assumes a fixed reference to data is included
        in the function and *does not* accept data as an argument.
    model_kwargs: dict, optional (Default {})
        dictionary of additional keyword arguments for model
        initialization


    Returns
    -------
    best_model: scHPF
        The model with the best loss facter `ntrials` random initializations
        and training runs
    """

    # run trials
    best_loss, best_model, best_t = np.finfo(np.float64).max, None, None
    for t in range(ntrials):
        # make a new model
        np.random.seed(t)
        print("scIHPF running with seed {}".format(t))
        model = scIHPF(
            nfactors=nfactors,
            min_iter=min_iter,
            max_iter=max_iter,
            check_freq=check_freq,
            epsilon=epsilon,
            better_than_n_ago=better_than_n_ago,
            verbose=verbose,
            dtype=dtype,
            **model_kwargs
        )

        # fit the model
        model.fit(X, **model_kwargs)

        loss = model.loss[-1]
        if loss < best_loss:
            best_model = model
            best_loss = loss
            best_t = t
            if verbose:
                print("New best!".format(t))
        if verbose:
            print("Trial {0} loss: {1:.6f}".format(t, loss))
            print("Best loss: {0:.6f} (trial {1})".format(best_loss, best_t))

    return best_model
