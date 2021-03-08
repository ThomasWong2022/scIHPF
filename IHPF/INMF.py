""" Integrative Non-negative matrix factorization
"""
#  Author: Thomas Wong
#         Mauricio Barahona
# License: BSD 3 clause

# Author: Vlad Niculae
#         Lars Buitinck
#         Mathieu Blondel <mathieu@mblondel.org>
#         Tom Dupre la Tour
# License: BSD 3 clause


import numbers
import numpy as np
import scipy.sparse as sp
import time
import warnings
from math import sqrt


from sklearn.decomposition._cdnmf_fast import _update_cdnmf_fast
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.validation import check_is_fitted, check_non_negative
from sklearn.utils.validation import _deprecate_positional_args




EPSILON = np.finfo(np.float32).eps


def norm(x):
    """Dot product-based Euclidean norm implementation

    See: http://fseoane.net/blog/2011/computing-the-vector-norm/

    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm
    """
    return sqrt(squared_norm(x))


def trace_dot(X, Y):
    """Trace of np.dot(X, Y.T).

    Parameters
    ----------
    X : array-like
        First matrix
    Y : array-like
        Second matrix
    """
    return np.dot(X.ravel(), Y.ravel())


def _check_init(A, shape, whom):
    A = check_array(A)
    if np.shape(A) != shape:
        raise ValueError('Array with wrong shape passed to %s. Expected %s, '
                         'but got %s ' % (whom, shape, np.shape(A)))
    check_non_negative(A, whom)
    if np.max(A) == 0:
        raise ValueError('Array passed to %s is full of zeros.' % whom)


def _beta_divergence(X, W, H, V, beta, square_root=False):
    """Compute the beta-divergence of X and dot(W, H+V).

    Parameters
    ----------
    X : float or array-like, shape (n_samples, n_features)

    W : float or dense array-like, shape (n_samples, n_components)

    H : float or dense array-like, shape (n_components, n_features)

    V:  float or dense array-like, shape (n_components, n_features)

    beta : float, string in {'frobenius', 'kullback-leibler', 'itakura-saito'}
        Parameter of the beta-divergence.
        If beta == 2, this is half the Frobenius *squared* norm.
        If beta == 1, this is the generalized Kullback-Leibler divergence.
        If beta == 0, this is the Itakura-Saito divergence.
        Else, this is the general beta-divergence.

    square_root : boolean, default False
        If True, return np.sqrt(2 * res)
        For beta == 2, it corresponds to the Frobenius norm.

    Returns
    -------
        res : float
            Beta divergence of X and np.dot(W, H+V)
    """
    beta = _beta_loss_to_float(beta)

    # The method can be called with scalars
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)
    V = np.atleast_2d(V)

    # Frobenius norm
    if beta == 2:
        # Avoid the creation of the dense np.dot(W, H+V) if X is sparse.
        if sp.issparse(X):
            norm_X = np.dot(X.data, X.data)
            norm_WH = trace_dot(np.dot(np.dot(W.T, W), H+V), H+V)
            cross_prod = trace_dot((X * (H+V).T), W)
            res = (norm_X + norm_WH - 2. * cross_prod) / 2.
        else:
            res = squared_norm(X - np.dot(W, H+V)) / 2.

        if square_root:
            return np.sqrt(res * 2)
        else:
            return res

    if sp.issparse(X):
        # compute np.dot(W, H) only where X is nonzero
        WH_data = _special_sparse_dot(W, H, V, X).data
        X_data = X.data
    else:
        WH = np.dot(W, H+V)
        WH_data = WH.ravel()
        X_data = X.ravel()

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = X_data > EPSILON
    WH_data = WH_data[indices]
    X_data = X_data[indices]

    # used to avoid division by zero
    WH_data[WH_data == 0] = EPSILON

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H+V))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H+V, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]+V[:, i]) ** beta)
        else:
            sum_WH_beta = np.sum(WH ** beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data ** beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        return np.sqrt(2 * res)
    else:
        return res


def _special_sparse_dot(W, H, V, X):
    """Computes np.dot(W, H+V), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = W.shape[1]

        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(W[ii[batch], :],
                                          H.T[jj[batch], :] + V.T[jj[batch], :]).sum(axis=1)

        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()
    else:
        return np.dot(W, H+V)

def _check_string_param(solver, regularization, beta_loss, init):
    allowed_solver = ('cd', 'mu')
    if solver not in allowed_solver:
        raise ValueError(
            'Invalid solver parameter: got %r instead of one of %r' %
            (solver, allowed_solver))

    allowed_regularization = ('both', 'components', 'transformation', None)
    if regularization not in allowed_regularization:
        raise ValueError(
            'Invalid regularization parameter: got %r instead of one of %r' %
            (regularization, allowed_regularization))

    # 'mu' is the only solver that handles other beta losses than 'frobenius'
    if solver != 'mu' and beta_loss not in (2, 'frobenius'):
        raise ValueError(
            'Invalid beta_loss parameter: solver %r does not handle beta_loss'
            ' = %r' % (solver, beta_loss))

    if solver == 'mu' and init == 'nndsvd':
        warnings.warn("The multiplicative update ('mu') solver cannot update "
                      "zeros present in the initialization, and so leads to "
                      "poorer results when used jointly with init='nndsvd'. "
                      "You may try init='nndsvda' or init='nndsvdar' instead.",
                      UserWarning)

    beta_loss = _beta_loss_to_float(beta_loss)
    return beta_loss


def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float"""
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    if not isinstance(beta_loss, numbers.Number):
        raise ValueError('Invalid beta_loss parameter: got %r instead '
                         'of one of %r, or a float.' %
                         (beta_loss, allowed_beta_loss.keys()))
    return beta_loss


def _initialize_nmf(X, n_components, init=None, eps=1e-6,
                    random_state=None):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : integer
        The number of components desired in the approximation.

    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure.
        Default: None.
        Valid options:

        - None: 'nndsvd' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

    eps : float
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Initial guesses for solving X ~= WH

    H : array-like, shape (n_components, n_features)
        Initial guesses for solving X ~= WH

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if (init is not None and init != 'random'
            and n_components > min(n_samples, n_features)):
        raise ValueError("init = '{}' can only be used when "
                         "n_components <= min(n_samples, n_features)"
                         .format(init))

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = 'nndsvd'
        else:
            init = 'random'

    # Random initialization
    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features).astype(X.dtype,
                                                             copy=False)
        W = avg * rng.randn(n_samples, n_components).astype(X.dtype,
                                                            copy=False)
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    return W, H


def _update_coordinate_descent(X, W, Ht, l1_reg, l2_reg, shuffle,
                               random_state):
    """Helper function for _fit_coordinate_descent

    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...)

    """
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    XHt = safe_sparse_dot(X, Ht)

    # L2 regularization corresponds to increase of the diagonal of HHt
    if l2_reg != 0.:
        # adds l2_reg only on the diagonal
        HHt.flat[::n_components + 1] += l2_reg
    # L1 regularization corresponds to decrease of each element of XHt
    if l1_reg != 0.:
        XHt -= l1_reg

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    return _update_cdnmf_fast(W, HHt, XHt, permutation)


def _fit_coordinate_descent(X, W, H, tol=1e-4, max_iter=200, l1_reg_W=0,
                            l1_reg_H=0, l2_reg_W=0, l2_reg_H=0, update_H=True,
                            verbose=0, shuffle=False, random_state=None):
    """Compute Non-negative Matrix Factorization (NMF) with Coordinate Descent

    The objective function is minimized with an alternating minimization of W
    and H. Each minimization is done with a cyclic (up to a permutation of the
    features) Coordinate Descent.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant matrix.

    W : array-like, shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like, shape (n_components, n_features)
        Initial guess for the solution.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    l1_reg_W : double, default: 0.
        L1 regularization parameter for W.

    l1_reg_H : double, default: 0.
        L1 regularization parameter for H.

    l2_reg_W : double, default: 0.
        L2 regularization parameter for W.

    l2_reg_H : double, default: 0.
        L2 regularization parameter for H.

    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : integer, default: 0
        The verbosity level.

    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.

    random_state : int, RandomState instance, default=None
        Used to randomize the coordinates in the CD solver, when
        ``shuffle`` is set to ``True``. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    Cichocki, Andrzej, and Phan, Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.
    """
    # so W and Ht are both in C order in memory
    Ht = check_array(H.T, order='C')
    X = check_array(X, accept_sparse='csr')

    rng = check_random_state(random_state)

    for n_iter in range(1, max_iter + 1):
        violation = 0.

        # Update W
        violation += _update_coordinate_descent(X, W, Ht, l1_reg_W,
                                                l2_reg_W, shuffle, rng)
        # Update H
        if update_H:
            violation += _update_coordinate_descent(X.T, Ht, W, l1_reg_H,
                                                    l2_reg_H, shuffle, rng)

        if n_iter == 1:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, V, n_iter

# Update W for one of the datasets 
def _multiplicative_update_w(X, W, H, V, beta_loss, alpha, gamma, update_H=True):
    """update W in Multiplicative Update NMF"""
    
    regularisation = alpha * np.dot(W, np.dot(V,V.T))
    
    if beta_loss == 2:
        # Numerator
        numerator = safe_sparse_dot(X, H.T + V.T)
        # Denominator
        HHt = np.dot(H+V, H.T+V.T)
        denominator = np.dot(W, HHt) + regularisation

    else:
        # Numerator
        # if X is sparse, compute W(H+V) only where X is non zero
        WH_safe_X = _special_sparse_dot(W, H, V, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1. < 0:
                WH[WH == 0] = EPSILON

        # to avoid taking a negative power of zero
        if beta_loss - 2. < 0:
            WH_safe_X_data[WH_safe_X_data == 0] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
        numerator = safe_sparse_dot(WH_safe_X, H.T+V.T)

        # Denominator
        if beta_loss == 1:
            H_sum = np.sum(H+V, axis=1)  # shape(n_components, )
            denominator = H_sum[np.newaxis, :]  + regularisation

        else:
            # computation of WHHt = dot(dot(W, H) ** beta_loss - 1, H.T)
            if sp.issparse(X):
                # memory efficient computation
                # (compute row by row, avoiding the dense matrix WH)
                WHHt = np.empty(W.shape)
                for i in range(X.shape[0]):
                    WHi = np.dot(W[i, :], H)
                    if beta_loss - 1 < 0:
                        WHi[WHi == 0] = EPSILON
                    WHi **= beta_loss - 1
                    WHHt[i, :] = np.dot(WHi, H.T+V.T)
            else:
                WH **= beta_loss - 1
                WHHt = np.dot(WH, H.T + V.T)
            denominator = WHHt  + regularisation

    
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_W = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_W **= gamma

    return delta_W


# Update V for one of the datasets 
def _multiplicative_update_v(X, W, H, V, beta_loss, alpha, gamma, update_H=True):
    """update W in Multiplicative Update NMF"""
    
    regularisation = alpha * np.dot(np.dot(W.T,W),V)
    
    if beta_loss == 2:
        # Numerator
        numerator = safe_sparse_dot(W.T,X)
        # Denominator
        WHt = np.dot(W, H+V)
        denominator = np.dot(W.T, WHt) + regularisation

    else:
        # Numerator
        # if X is sparse, compute W(H+V) only where X is non zero
        WH_safe_X = _special_sparse_dot(W, H, V, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1. < 0:
                WH[WH == 0] = EPSILON

        # to avoid taking a negative power of zero
        if beta_loss - 2. < 0:
            WH_safe_X_data[WH_safe_X_data == 0] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
        numerator = safe_sparse_dot(W.T, WH_safe_X)

        # Denominator
        if beta_loss == 1:
            Wsum = np.sum(W, axis=0)  # shape(n_components, )
            denominator = Wsum[:, np.newaxis]  + regularisation

        else:
            # computation of WHHt = dot(dot(W, H) ** beta_loss - 1, H.T)
            if sp.issparse(X):
                # memory efficient computation
                # (compute row by row, avoiding the dense matrix WH)
                WHHt = np.empty(W.shape)
                for i in range(X.shape[0]):
                    WHi = np.dot(W[i, :], H+V)
                    if beta_loss - 1 < 0:
                        WHi[WHi == 0] = EPSILON
                    WHi **= beta_loss - 1
                    WHHt[i, :] = np.dot(W.T, WHi)
            else:
                WH **= beta_loss - 1
                WHHt = np.dot(W.T, WH)
            denominator = WHHt  + regularisation

    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_V = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_V **= gamma

    return delta_V


def _multiplicative_update_h(X, W, H, V, beta_loss, alpha, gamma):
    """update H in Multiplicative Update NMF"""
    if beta_loss == 2:
        numerator = 0
        denominator = 0
        for i in range(len(X)):
            numerator += safe_sparse_dot(W[i].T, X[i])
            denominator += np.dot(np.dot(W[i].T, W[i]), H+V[i])
    else:
        numerator = 0
        denominator = 0
        for i in range(len(X)):
            # Numerator
            WH_safe_X = _special_sparse_dot(W[i], H, V[i], X[i])
            if sp.issparse(X[i]):
                WH_safe_X_data = WH_safe_X.data
                X_data = X.data
            else:
                WH_safe_X_data = WH_safe_X
                X_data = X
                # copy used in the Denominator
                WH = WH_safe_X.copy()
                if beta_loss - 1. < 0:
                    WH[WH == 0] = EPSILON

            # to avoid division by zero
            if beta_loss - 2. < 0:
                WH_safe_X_data[WH_safe_X_data == 0] = EPSILON

            if beta_loss == 1:
                np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
            elif beta_loss == 0:
                # speeds up computation time
                # refer to /numpy/numpy/issues/9363
                WH_safe_X_data **= -1
                WH_safe_X_data **= 2
                # element-wise multiplication
                WH_safe_X_data *= X_data
            else:
                WH_safe_X_data **= beta_loss - 2
                # element-wise multiplication
                WH_safe_X_data *= X_data

            # here numerator = dot(W.T, (dot(W, H+V) ** (beta_loss - 2)) * X)
            numerator += safe_sparse_dot(W[i].T, WH_safe_X)

            # Denominator
            if beta_loss == 1:
                W_sum = np.sum(W[i], axis=0)  # shape(n_components, )
                W_sum[W_sum == 0] = 1.
                denominator += W_sum[:, np.newaxis]

            # beta_loss not in (1, 2)
            else:
                # computation of WtWH = dot(W.T, dot(W, H) ** beta_loss - 1)
                if sp.issparse(X):
                    # memory efficient computation
                    # (compute column by column, avoiding the dense matrix WH)
                    WtWH = np.empty(H.shape)
                    for i in range(X[i].shape[1]):
                        WHi = np.dot(W[i], H[:, i])
                        if beta_loss - 1 < 0:
                            WHi[WHi == 0] = EPSILON
                        WHi **= beta_loss - 1
                        WtWH[:, i] = np.dot(W[i].T, WHi)
                else:
                    WH **= beta_loss - 1
                    WtWH = np.dot(W[i].T, WH)
                denominator += WtWH

    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_H = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_H **= gamma

    return delta_H


def _fit_multiplicative_update(X, W, H, V, beta_loss='frobenius',
                               max_iter=200, tol=1e-4,
                               alpha=1.0, update_H=True, verbose=0):
    """Compute Non-negative Matrix Factorization with Multiplicative Update

    The objective function is _beta_divergence(X, WH) and is minimized with an
    alternating minimization of W and H. Each minimization is done with a
    Multiplicative Update.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant input matrix.

    W : array-like, shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like, shape (n_components, n_features)
        Initial guess for the solution.

    beta_loss : float or string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros.

    max_iter : integer, default: 200
        Number of iterations.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    alpha: float, default: 1
        Regularisation parameter for dataset specific components 

    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : integer, default: 0
        The verbosity level.

    Returns
    -------
    W : list of array, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : array, shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    V:  list of array, shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    start_time = time.time()

    beta_loss = _beta_loss_to_float(beta_loss)

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    if beta_loss < 1:
        gamma = 1. / (2. - beta_loss)
    elif beta_loss > 2:
        gamma = 1. / (beta_loss - 1.)
    else:
        gamma = 1.

    # used for the convergence criterion
    error_at_init = 0
    for i in range(len(X)):
        error_at_init += _beta_divergence(X[i], W[i], H, V[i], beta_loss, square_root=True)
    previous_error = error_at_init

   
    for n_iter in range(1, max_iter + 1):

        for i in range(len(X)):
            # update W
            delta_W = _multiplicative_update_w(X[i], W[i], H, V[i], beta_loss, alpha, gamma,)
            W[i] *= delta_W

            # necessary for stability with beta_loss < 1
            if beta_loss < 1:
                W[i][W[i] < np.finfo(np.float64).eps] = 0.

            # update V 
            delta_V =  _multiplicative_update_v(X[i], W[i], H, V[i], beta_loss, alpha, gamma,)
            V[i] *= delta_V

            # necessary for stability with beta_loss < 1
            if beta_loss < 1:
                V[i][V[i] < np.finfo(np.float64).eps] = 0.

        # update H
        if update_H:
            delta_H = _multiplicative_update_h(X, W, H, V, beta_loss, alpha, gamma)
            H *= delta_H

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = 0
            for i in range(len(X)):
                error += _beta_divergence(X[i], W[i], H, V[i], beta_loss, square_root=True)

            if verbose:
                iter_time = time.time()
                print("Epoch %02d reached after %.3f seconds, error: %f" %
                      (n_iter, iter_time - start_time, error))

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print("Epoch %02d reached after %.3f seconds." %
              (n_iter, end_time - start_time))

    return W, H, V, n_iter


def integrative_non_negative_factorization(X, W=None, H=None, V=None, n_components=None,
                               init=None, update_H=True, solver='mu',
                               beta_loss=2, tol=1e-4,
                               max_iter=200, alpha=1.0, random_state=None,
                               verbose=0, shuffle=False):
    
    
    r"""Integrative Compute Non-negative Matrix Factorization (iNMF)

    https://www.ncbi.nlm.nih.gov/pubmed/26377073


    Find non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is::

        0.5 * \sum_{i=1}^I ||X_i - W_i(H+V_i) ||_Fro^2 + \alpha \sum_{i=1}^I ||W_iV_i||

    Where::

        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)

    For multiplicative-update ('mu') solver, the Frobenius norm
    (0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
    by changing the beta_loss parameter.

    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.

    Parameters
    ----------
    X : list of array-like, shape (n_samples, n_features)
        Constant matrix.

    W : list of array-like, shape (n_samples, n_components)
        If init='custom', it is used as initial guess for the solution.

    H : array-like, shape (n_components, n_features)
        If init='custom', it is used as initial guess for the solution.
        If update_H=False, it is used as a constant, to solve for W only.

    V:  list of array-like, shape (n_components, n_features)
        If init='custom', it is used as initial guess for the solution.



    n_components : integer
        Number of components, if n_components is not set all features
        are kept.

    init : None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'custom'
        Method used to initialize the procedure.
        Default: None.

        Valid options:

        - None: 'nndsvd' if n_components < n_features, otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

        .. versionchanged:: 0.23
            The default value of `init` changed from 'random' to None in 0.23.

    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    solver : 'cd' | 'mu'
        Numerical solver to use:

        - 'cd' is a Coordinate Descent solver that uses Fast Hierarchical
            Alternating Least Squares (Fast HALS).

        - 'mu' is a Multiplicative Update solver.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float , default 2
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

        .. versionadded:: 0.19

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    alpha: float, default: 1 
        Regularisation for dataset specific factors

    random_state : int, RandomState instance, default=None
        Used for NMF initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : integer, default: 0
        The verbosity level.

    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.

    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        Actual number of iterations.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import non_negative_factorization
    >>> W, H, n_iter = non_negative_factorization(X, n_components=2,
    ... init='random', random_state=0)

    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """

    for i in range(len(X)):
        X[i] = check_array(X[i], accept_sparse=('csr', 'csc'), dtype=[np.float64, np.float32])
        check_non_negative(X[i], "NMF (input X)")

        # beta_loss = _check_string_param(solver, regularization, beta_loss, init)

        if X[i].min() == 0 and beta_loss <= 0:
            raise ValueError("When beta_loss <= 0 and X contains zeros, "
                         "the solver may diverge. Please add small values to "
                         "X, or use a positive beta_loss.")

    n_samples = [Xk.shape[0] for Xk in X]
    n_features = [Xk.shape[1] for Xk in X]

    if not np.array_equal(n_features,np.ones(len(X))*n_features[0]):
        raise ValueError("All matrices must have the same number of features")


    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, numbers.Integral) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, numbers.Integral) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    # check W and H, or initialize them
    if init == 'custom' and update_H:
        _check_init(H, (n_components, n_features[0]), "NMF (input H)")
        for count in range(len(X)):
            _check_init(W[count], (n_samples[count], n_components), "NMF (input W)")
            if H.dtype != X[count].dtype or W[count].dtype != X[count].dtype:
                raise TypeError("H and W should have the same dtype as X. Got "
                                "H.dtype = {} and W.dtype = {}."
                                .format(H.dtype, W[count].dtype))
    elif not update_H:
        _check_init(H, (n_components, n_features[0]), "NMF (input H)")
        for count in range(len(X)):
            if H.dtype != X[count].dtype:
                raise TypeError("H should have the same dtype as X. Got "
                                "H.dtype = {}.".format(H.dtype,))
        # 'mu' solver should not be initialized by zeros
        if solver == 'mu':
            W = list()
            for i in range(len(X)):
                avg = np.sqrt(X[i].mean() / n_components)
                W.append(np.full((n_samples[i], n_components), avg, dtype=X[i].dtype))
        else:
            W = [np.zeros((n_samples[i], n_components), dtype=X.dtype) for i in range(len(X))]
    
    else:
        W = list()
        V = list()
        for i in range(len(X)):
            Wk, Vk = _initialize_nmf(X[i], n_components, init=init, random_state=random_state)
            W.append(Wk)
            V.append(Vk/2.0)
        # Estimate H by the average of dataset factorisation 
        H = sum(V)/(len(V)*2.0)

    if solver == 'cd':
        W, H, V, n_iter  = _fit_coordinate_descent(X, W, H, V, tol, max_iter, alpha,
                                               update_H=update_H,
                                               verbose=verbose,
                                               shuffle=shuffle,
                                               random_state=random_state)
    elif solver == 'mu':
        W, H, V, n_iter = _fit_multiplicative_update(X, W, H, V, beta_loss, max_iter,
                                                  tol, alpha, update_H,
                                                  verbose)

    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        warnings.warn("Maximum number of iterations %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)

    return W, H, V, n_iter


class INMF(TransformerMixin, BaseEstimator):
    r"""Integrative Non-Negative Matrix Factorization (INMF)

    Where::

        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)

    For multiplicative-update ('mu') solver, the Frobenius norm
    (0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
    by changing the beta_loss parameter.

    The objective function is minimized with an alternating minimization of W
    and H.

    Read more in the :ref:`User Guide <NMF>`.

    Parameters
    ----------
    n_components : int or None
        Number of components, if n_components is not set all features
        are kept.

    init : None | 'random' | 'nndsvd' |  'nndsvda' | 'nndsvdar' | 'custom'
        Method used to initialize the procedure.
        Default: None.
        Valid options:

        - None: 'nndsvd' if n_components <= min(n_samples, n_features),
            otherwise random.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

    solver : 'cd' | 'mu'
        Numerical solver to use:
        'cd' is a Coordinate Descent solver.
        'mu' is a Multiplicative Update solver.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float or string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

        .. versionadded:: 0.19

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    random_state : int, RandomState instance, default=None
        Used for initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : bool, default=False
        Whether to be verbose.

    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.

        .. versionadded:: 0.17
           *shuffle* parameter used in the Coordinate Descent solver.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Factorization matrix, sometimes called 'dictionary'.

    n_components_ : integer
        The number of components. It is same as the `n_components` parameter
        if it was given. Otherwise, it will be same as the number of
        features.

    reconstruction_err_ : number
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data ``X`` and the reconstructed data ``WH`` from
        the fitted model.

    n_iter_ : int
        Actual number of iterations.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> Y = np.array([[2, 5], [5, 2], [3, 4.2], [4, 2], [5, 1.8], [6, 3]])
    >>> from sklearn.decomposition import INMF
    >>> model = INMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform([X,Y,X+Y])
    >>> H = model.components_

    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    @_deprecate_positional_args
    def __init__(self, n_components=None, *, init=None, solver='mu',
                 beta_loss=2, tol=1e-4, max_iter=2000,
                 random_state=None, alpha=1., verbose=0,
                 shuffle=False):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.shuffle = shuffle

    def _more_tags(self):
        return {'requires_positive_X': True}

    def fit_transform(self, X, y=None, W=None, H=None, V=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : list of {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        W : list of array-like, shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        H : array-like, shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.
        
        V: list of array-like, shape (n_components, n_features)
           If init='custom', it is used as initial guess for the solution.
        
        Returns
        -------
        W : list of array, shape (n_samples, n_components)
            Transformed data.
        H: array-like, shape (n_components, n_features)
           shared factors
        V: list of array, shape (n_components, n_features)
           dataset specific factors
        
        """

        for i in range(len(X)):
            X[i] = self._validate_data(X[i], accept_sparse=('csr', 'csc'), dtype=[np.float64, np.float32])

        W, H, V, n_iter_ = integrative_non_negative_factorization(
            X=X, W=W, H=H, n_components=self.n_components, init=self.init,
            update_H=True, solver=self.solver, beta_loss=self.beta_loss,
            tol=self.tol, max_iter=self.max_iter, alpha=self.alpha,
            random_state=self.random_state, verbose=self.verbose,
            shuffle=self.shuffle)

        self.reconstruction_err_ = 0
        for i in range(len(X)):
            self.reconstruction_err_ += _beta_divergence(X[i], W[i], H, V[i], self.beta_loss, square_root=True)

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter_

        print('Reconstruction error {}'.format(self.reconstruction_err_))
        
        return W, H, V

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        """Transform the data X according to the fitted NMF model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be transformed by the model

        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data
        """

        W, _, _, n_iter_ = integrative_non_negative_factorization(
            X=X, W=None, H=self.components_, V=None, n_components=self.n_components_,
            init=self.init, update_H=False, solver=self.solver,
            beta_loss=self.beta_loss, tol=self.tol, max_iter=self.max_iter,
            alpha=self.alpha, random_state=self.random_state, verbose=self.verbose,
            shuffle=self.shuffle)

        return W

    def inverse_transform(self, W, V):
        """Transform data back to its original space.

        Parameters
        ----------
        W : list of {array-like, sparse matrix}, shape (n_samples, n_components)
            Transformed data matrix

        V: list of {array-like, sparse matrix}, shape (n_components,n_features)
           Dataset specific factors

        Returns
        -------
        X : list of {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix of original shape

        .. versionadded:: 0.18
        """
        return [np.dot(W[i], self.components_ + V[i]) for i in range(len(W))]
