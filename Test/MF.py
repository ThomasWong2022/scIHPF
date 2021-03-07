# Cross-validation for pCMF and HPF using data generated from pCMF

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.base import BaseEstimator


from multiprocessing import Process, Pool

import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score


from MulticoreTSNE import MulticoreTSNE as TSNE

# Numba calculations
from numba import njit, prange


##################################################################################
## Loss functions
######################################################################################


def _special_sparse_dot(W, H, X):
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = W.shape[1]

        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(W[ii[batch], :], H.T[jj[batch], :]).sum(
                axis=1
            )

        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()
    else:
        return np.dot(W, H)


def trace_dot(X, Y):
    """Trace of np.dot(X, Y.T)"""
    return np.dot(X.ravel(), Y.ravel())


def beta_divergence_ppc(X, X_rep, beta, square_root=False):
    """Compute the beta-divergence of X and dot(W, H).
    Parameters
    ----------
    X : float or array-like, shape (n_samples, n_features)
    X_rep: float or array-like, shape (n_samples, n_features)
    beta : float,
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
            Beta divergence of X and np.dot(X, H)
    """
    EPSILON = np.finfo(np.float32).eps

    # The method can be called with scalars
    if not sp.issparse(X):
        X = np.atleast_2d(X)

    # Frobenius norm
    if beta == 2:
        if sp.issparse(X):
            diff = X - X_rep
            res = np.dot(diff.data, diff.data) / 2
        else:
            diff = (X - X_rep).ravel()
            res = np.dot(diff, diff) / 2.0

        if square_root:
            return np.sqrt(res * 2)
        else:
            return res

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = X.data > EPSILON
    WH_data = X_rep.data[np.where(indices)]
    X_data = X.data[indices]

    WH_data[WH_data <= EPSILON] = EPSILON

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += X_rep.data.sum() - X.data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = np.divide(X.data, X_rep.data)
        res = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
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


def sparse_cv(filteredcountdata, axis=0):
    np.seterr(divide="ignore", invalid="ignore")
    gene_mean = np.mean(filteredcountdata, axis=axis)
    squared_count = filteredcountdata.power(2)
    gene_variances = np.sqrt(squared_count.mean(axis=axis) - np.square(gene_mean))
    gene_cv = np.divide(gene_variances, gene_mean)
    return np.ravel(gene_cv)


def KS_CV(X, X_rep):
    real_cell_cv = sparse_cv(X, axis=1)
    simu_cell_cv = sparse_cv(X_rep, axis=1)
    cell_ks = stats.ks_2samp(real_cell_cv, simu_cell_cv)[0]
    real_gene_cv = sparse_cv(X, axis=0)
    simu_gene_cv = sparse_cv(X_rep, axis=0)
    gene_ks = stats.ks_2samp(real_gene_cv, simu_gene_cv)[0]
    return cell_ks, gene_ks


###############################################
## Correlation matrices
##
##########################################################################


def correlation_factors(m1, m2):
    result = np.zeros((m1.shape[1], m2.shape[1]))
    for i in range(0, m1.shape[1]):
        for j in range(i, m2.shape[1]):
            result[i, j] = np.float(np.corrcoef(m1[:, i], m2[:, j], rowvar=False)[0, 1])
    print("Running correlation factors")
    return result


###############################################
## Clustering
##
##########################################################################


def clustering_factors(cellscores, noclusters, method="kmeans"):
    if method == "kmeans":
        kmeans = KMeans(n_clusters=noclusters, random_state=0).fit(cellscores)
    return kmeans.labels_


################################################
## Matrix factorisation methods
##
#################################################


## HPF as provided by Blei with helper functions for posterior predictive checks
##
## Accept coo_matrix


class scHPFI(schpf.scHPFI):
    
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

    def simulate_count_matrix(self):
        count_matrices = []
        ndatasets = len(self.theta)
        for i in range(ndatasets):
            theta_samples = self.theta[i].sample().reshape(self.theta[i].dims)
            beta_samples = self.beta.sample().reshape(self.beta.dims)
            delta_samples = self.delta[i].sample().reshape(self.delta[i].dims)
            rates = np.matmul(
                theta_samples, np.transpose(beta_samples) + np.transpose(delta_samples)
            )
            count_sample = np.random.poisson(rates)
            count_matrices.append(count_sample)
        return count_matrices

    def explained_deviance(self, X, X_rep, beta):
        try:
            X_avg = coo_matrix(
                np.matmul(np.ones((X.shape[0], 1)), X.mean(axis=0).reshape(1, -1))
            )
            average_divergence = beta_divergence_ppc(X, X_avg, beta)
            model_divergence = beta_divergence_ppc(X, X_rep, beta)
            ratio = (average_divergence - model_divergence) / average_divergence
            return ratio
        except:
            print("Error in computing explained deviance")
            return 0

    ## Average KS statistics over all the datasets and simulations

    def reconstruction_metrics(self, data, PPC_simulation=10):
        temp_KSCell = np.zeros((len(data), PPC_simulation))
        temp_KSGene = np.zeros((len(data), PPC_simulation))
        temp_Frob = np.zeros((len(data), PPC_simulation))
        temp_KL = np.zeros((len(data), PPC_simulation))
        for i in range(0, PPC_simulation):
            count_sample = self.simulate_count_matrix()
            for j in range(len(count_sample)):
                HPF_count = coo_matrix(count_sample[j])
                HPF_KS = KS_CV(data[j], HPF_count)
                temp_KSCell[j, i] = HPF_KS[0]
                temp_KSGene[j, i] = HPF_KS[1]
                temp_Frob[j, i] = self.explained_deviance(data[j], HPF_count, 2)
                temp_KL[j, i] = self.explained_deviance(data[j], HPF_count, 1)
        return (
            list(np.mean(temp_Frob, axis=1)),
            list(np.mean(temp_KL, axis=1)),
            list(np.mean(temp_KSCell, axis=1)),
            list(np.mean(temp_KSGene, axis=1)),
        )

    def copy_params(self, scHPFmodel):
        self.eta = scHPFmodel.eta
        self.zeta = scHPFmodel.zeta
        self.xi = scHPFmodel.xi
        self.beta = scHPFmodel.beta
        self.theta = scHPFmodel.theta
        self.delta = scHPFmodel.delta


class scHPF2(schpf.scHPF):
    def simulate_cell_factor(self, theta=None, n_samples=10):
        theta = self.theta if theta is None else theta
        theta_samples = np.mean(theta.sample(n_samples), axis=-1)
        return theta_samples

    def simulate_gene_factor(self, beta=None, n_samples=10):
        beta = self.beta if beta is None else beta
        beta_samples = np.mean(beta.sample(n_samples), axis=-1)
        return beta_samples

    def simulate_count_matrix(self):
        theta_samples = self.theta.sample().reshape(self.theta.dims)
        beta_samples = self.beta.sample().reshape(self.beta.dims)
        rates = np.matmul(theta_samples, np.transpose(beta_samples))
        count_sample = np.random.poisson(rates)
        return count_sample

    def explained_deviance(self, X, X_rep, beta):
        try:
            X_avg = coo_matrix(
                np.matmul(np.ones((X.shape[0], 1)), X.mean(axis=0).reshape(1, -1))
            )
            average_divergence = beta_divergence_ppc(X, X_avg, beta)
            model_divergence = beta_divergence_ppc(X, X_rep, beta)
            ratio = (average_divergence - model_divergence) / average_divergence
            return ratio
        except:
            print("Error in calculating deviance")
            return 0

    def reconstruction_metrics(self, data, PPC_simulation=10):
        temp_Frob = []
        temp_KL = []
        temp_KSCell = []
        temp_KSGene = []
        for i in range(0, PPC_simulation):
            count_sample = self.simulate_count_matrix()
            HPF_count = coo_matrix(count_sample)
            HPF_KS = KS_CV(data, HPF_count)
            temp_KSCell.append(HPF_KS[0])
            temp_KSGene.append(HPF_KS[1])
            temp_Frob.append(self.explained_deviance(data, HPF_count, 2))
            temp_KL.append(self.explained_deviance(data, HPF_count, 1))
        return (
            np.mean(temp_Frob),
            np.mean(temp_KL),
            np.mean(temp_KSCell),
            np.mean(temp_KSGene),
        )


# sklearn default ensures zero mean but not scaling variance
# USd truncated SVD in sklearn as PCA as it accepts sparse matrix as input
class scPCA:
    def __init__(self, k):
        from sklearn.decomposition import TruncatedSVD

        np.random.seed(0)
        self.n_components = k
        self.method = TruncatedSVD(n_components=self.n_components)

    def fit(self, X):
        self.data = X
        self.method.fit(self.data)
        self.cell_score = self.method.transform(self.data)
        self.gene_score = self.method.components_.transpose()

    # TODO sparsePCA does not have inverse transform
    def count_matrix(self):
        original = self.method.inverse_transform(self.method.transform(self.data))
        return original

    def explained_deviance(self, X, beta):
        X_rep = coo_matrix(self.count_matrix())
        X_avg = coo_matrix(
            np.matmul(np.ones((X.shape[0], 1)), X.mean(axis=0).reshape(1, -1))
        )
        average_divergence = beta_divergence_ppc(X, X_avg, beta)
        model_divergence = beta_divergence_ppc(X, X_rep, beta)
        ratio = (average_divergence - model_divergence) / average_divergence
        return ratio


# Acccept any sparse
class scNMFP:
    def __init__(self, k, **kwargs):
        from sklearn.decomposition import NMF

        np.random.seed(0)
        self.n_components = k
        self.method = NMF(
            n_components=self.n_components,
            solver="mu",
            beta_loss="kullback-leibler",
            **kwargs
        )

    def fit(self, X):
        self.data = X
        self.method.fit(self.data)
        self.cell_score = self.method.transform(self.data)
        self.gene_score = self.method.components_.transpose()

    def count_matrix(self):
        original = self.method.inverse_transform(self.method.transform(self.data))
        return original

    def explained_deviance(self, X, beta):
        X_rep = coo_matrix(self.count_matrix())
        X_avg = coo_matrix(
            np.matmul(np.ones((X.shape[0], 1)), X.mean(axis=0).reshape(1, -1))
        )
        average_divergence = beta_divergence_ppc(X, X_avg, beta)
        model_divergence = beta_divergence_ppc(X, X_rep, beta)
        ratio = (average_divergence - model_divergence) / average_divergence
        return ratio


class scNMF:
    def __init__(self, k, **kwargs):
        from sklearn.decomposition import NMF

        np.random.seed(0)
        self.n_components = k
        self.method = NMF(
            n_components=self.n_components, solver="mu", beta_loss="frobenius", **kwargs
        )

    def fit(self, X):
        self.data = X
        self.method.fit(self.data)
        self.cell_score = self.method.transform(self.data)
        self.gene_score = self.method.components_.transpose()

    def count_matrix(self):
        original = self.method.inverse_transform(self.method.transform(self.data))
        return original

    def explained_deviance(self, X, beta):
        X_rep = coo_matrix(self.count_matrix())
        X_avg = coo_matrix(
            np.matmul(np.ones((X.shape[0], 1)), X.mean(axis=0).reshape(1, -1))
        )
        average_divergence = beta_divergence_ppc(X, X_avg, beta)
        model_divergence = beta_divergence_ppc(X, X_rep, beta)
        ratio = (average_divergence - model_divergence) / average_divergence
        return ratio


class scINMF:
    def __init__(self, k, alpha=1, **kwargs):
        from sklearn.decomposition import INMF

        np.random.seed(0)
        self.n_components = k
        self.method = INMF(
            n_components=self.n_components, solver="mu", alpha=alpha, **kwargs
        )

    def fit(self, X):
        self.data = X
        (
            self.cell_score,
            self.shared_gene_score,
            self.dataset_gene_score,
        ) = self.method.fit_transform(self.data)

    def count_matrix(self):
        # print(self.cell_score[0].shape)
        # print(self.dataset_gene_score[0].shape)
        # print(self.shared_gene_score.shape)
        original = [
            np.dot(
                self.cell_score[i], self.shared_gene_score + self.dataset_gene_score[i]
            )
            for i in range(len(self.cell_score))
        ]
        return original

    def explained_deviance(self, X, X_rep, beta):
        try:
            X_avg = coo_matrix(
                np.matmul(np.ones((X.shape[0], 1)), X.mean(axis=0).reshape(1, -1))
            )
            average_divergence = beta_divergence_ppc(X, X_avg, beta)
            model_divergence = beta_divergence_ppc(X, X_rep, beta)
            ratio = (average_divergence - model_divergence) / average_divergence
            return ratio
        except:
            print("Error in calculating deviance")
            return 0


###############################################################
## Cross validation tasks for matrix factorisation methods
##
##############################################################


def MF_hyperparameter(data, debug=True, a=0.3, c=0.3, ap=1.0, cp=1.0, **kwargs):

    print("Start MF hyper-parameters for data size", data.shape)

    startK = kwargs.get("startK", 2)
    endK = kwargs.get("endK", 10)
    step = kwargs.get("step", 2)
    PPC_simulation = 10

    Explained_deviance = dict()
    Explained_deviance["Frob"] = dict()
    Explained_deviance["KL"] = dict()
    Explained_deviance["KS_cell"] = dict()
    Explained_deviance["KS_gene"] = dict()
    Explained_deviance["Matrix"] = data.shape

    sparsity = np.sum(np.sum(data > 0)) / data.size
    Explained_deviance["Sparsity"] = sparsity

    for k in range(startK, endK + 1, step):

        HPF_model = scHPF2(nfactors=k, a=a, ap=ap, c=c, cp=cp)
        sparseX = coo_matrix(data)
        HPF_model.fit(sparseX, verbose=False)

        HPF_U = HPF_model.cell_score()
        HPF_V = HPF_model.gene_score()

        if debug:
            print("HPF running for {} Factors".format(k))

        # Compute stats
        temp_Frob = []
        temp_KL = []
        temp_KSCell = []
        temp_KSGene = []
        for i in range(0, PPC_simulation):
            HPF_count = coo_matrix(HPF_model.simulate_count_matrix())
            HPF_KS = KS_CV(data, HPF_count)
            temp_Frob.append(HPF_model.explained_deviance(data, HPF_count, 2))
            temp_KL.append(HPF_model.explained_deviance(data, HPF_count, 1))
            temp_KSCell.append(HPF_KS[0])
            temp_KSGene.append(HPF_KS[1])

        Explained_deviance["Frob"]["HPF_{}".format(k)] = np.mean(temp_Frob)
        Explained_deviance["KL"]["HPF_{}".format(k)] = np.mean(temp_KL)
        Explained_deviance["KS_cell"]["HPF_{}".format(k)] = np.mean(temp_KSCell)
        Explained_deviance["KS_gene"]["HPF_{}".format(k)] = np.mean(temp_KSGene)

    return Explained_deviance


def MF_genes(data, topgenes=10000, debug=True, **kwargs):

    startK = kwargs.get("startK", 2)
    endK = kwargs.get("endK", 10)
    step = kwargs.get("step", 2)
    PPC_simulation = 10

    Results = dict()
    Results["CountMatrix"] = data

    Explained_deviance = dict()
    Explained_deviance["Frob"] = dict()
    Explained_deviance["KL"] = dict()
    Explained_deviance["KS_cell"] = dict()
    Explained_deviance["KS_gene"] = dict()
    Explained_deviance["Matrix"] = data.shape
    sparsity = np.sum(np.sum(data > 0)) / data.size
    Explained_deviance["Sparsity"] = sparsity

    print("Start MF genes for {} features data size {}".format(topgenes, data.shape))

    # Run analysis

    for k in range(startK, endK + 1, step):

        PCA_model = scPCA(k)
        PCA_model.fit(data)
        PCA_cell = PCA_model.cell_score
        PCA_gene = PCA_model.gene_score
        PCA_count = coo_matrix(PCA_model.count_matrix())
        if debug:
            print("PCA running for {} Factors".format(k))

        Results["PCA_{}".format(k)] = {"cell_score": PCA_cell, "gene_score": PCA_gene}
        Explained_deviance["Frob"]["PCA_{}".format(k)] = PCA_model.explained_deviance(
            data, 2
        )
        Explained_deviance["KL"]["PCA_{}".format(k)] = PCA_model.explained_deviance(
            data, 1
        )
        PCA_KS = KS_CV(data, PCA_count)
        Explained_deviance["KS_cell"]["PCA_{}".format(k)] = PCA_KS[0]
        Explained_deviance["KS_gene"]["PCA_{}".format(k)] = PCA_KS[1]

        NMF_model = scNMF(k)
        NMF_model.fit(data)
        NMF_cell = NMF_model.cell_score
        NMF_gene = NMF_model.gene_score
        NMF_count = coo_matrix(NMF_model.count_matrix())
        if debug:
            print("NMF running for {} Factors".format(k))

        Results["NMF_{}".format(k)] = {"cell_score": NMF_cell, "gene_score": NMF_gene}
        Explained_deviance["Frob"]["NMF_{}".format(k)] = NMF_model.explained_deviance(
            data, 2
        )
        Explained_deviance["KL"]["NMF_{}".format(k)] = NMF_model.explained_deviance(
            data, 1
        )
        NMF_KS = KS_CV(data, NMF_count)
        Explained_deviance["KS_cell"]["NMF_{}".format(k)] = NMF_KS[0]
        Explained_deviance["KS_gene"]["NMF_{}".format(k)] = NMF_KS[1]

        NMF_model = scNMFP(k)
        NMF_model.fit(data)
        NMF_cell = NMF_model.cell_score
        NMF_gene = NMF_model.gene_score
        NMF_count = coo_matrix(NMF_model.count_matrix())
        if debug:
            print("NMFP running for {} Factors".format(k))

        Results["NMFP_{}".format(k)] = {"cell_score": NMF_cell, "gene_score": NMF_gene}
        Explained_deviance["Frob"]["NMFP_{}".format(k)] = NMF_model.explained_deviance(
            data, 2
        )
        Explained_deviance["KL"]["NMFP_{}".format(k)] = NMF_model.explained_deviance(
            data, 1
        )
        NMF_KS = KS_CV(data, NMF_count)
        Explained_deviance["KS_cell"]["NMFP_{}".format(k)] = NMF_KS[0]
        Explained_deviance["KS_gene"]["NMFP_{}".format(k)] = NMF_KS[1]

        HPF_model = scHPF2(k)
        sparseX = coo_matrix(data)
        HPF_model.fit(sparseX, verbose=True)
        HPF_U = HPF_model.cell_score()
        HPF_V = HPF_model.gene_score()
        if debug:
            print("HPF running for {} Factors".format(k))

        Results["HPF_{}".format(k)] = {
            "cell_score": HPF_U,
            "gene_score": HPF_V,
        }
        # Compute stats
        temp_Frob = []
        temp_KL = []
        temp_KSCell = []
        temp_KSGene = []
        for i in range(0, PPC_simulation):
            HPF_count = coo_matrix(HPF_model.simulate_count_matrix())
            HPF_KS = KS_CV(data, HPF_count)
            temp_Frob.append(HPF_model.explained_deviance(data, HPF_count, 2))
            temp_KL.append(HPF_model.explained_deviance(data, HPF_count, 1))
            temp_KSCell.append(HPF_KS[0])
            temp_KSGene.append(HPF_KS[1])

        Explained_deviance["Frob"]["HPF_{}".format(k)] = np.mean(temp_Frob)
        Explained_deviance["KL"]["HPF_{}".format(k)] = np.mean(temp_KL)
        Explained_deviance["KS_cell"]["HPF_{}".format(k)] = np.mean(temp_KSCell)
        Explained_deviance["KS_gene"]["HPF_{}".format(k)] = np.mean(temp_KSGene)

    return Results, Explained_deviance


def MF_factors(data, debug=True, **kwargs):

    startK = kwargs.get("startK", 2)
    endK = kwargs.get("endK", 10)
    step = kwargs.get("step", 2)
    PPC_simulation = 10

    Results = dict()
    Results["CountMatrix"] = data

    Explained_deviance = dict()
    Explained_deviance["Frob"] = dict()
    Explained_deviance["KL"] = dict()
    Explained_deviance["KS_cell"] = dict()
    Explained_deviance["KS_gene"] = dict()
    Explained_deviance["Matrix"] = data.shape

    sparsity = np.sum(np.sum(data > 0)) / data.size
    Explained_deviance["Sparsity"] = sparsity

    Clustering = dict()
    Clustering["kmeans"] = dict()

    for k in range(startK, endK + 1, step):

        HPF_model = scHPF2(k)
        sparseX = coo_matrix(data)
        HPF_model.fit(sparseX, verbose=False)
        HPF_U = HPF_model.cell_score()
        HPF_V = HPF_model.gene_score()
        if debug:
            print("HPF running for {} Factors".format(k))

        Results["HPF_{}".format(k)] = {
            "cell_score": HPF_U,
            "gene_score": HPF_V,
        }
        # Compute stats
        temp_Frob = []
        temp_KL = []
        temp_KSCell = []
        temp_KSGene = []
        for i in range(0, PPC_simulation):
            HPF_count = coo_matrix(HPF_model.simulate_count_matrix())
            HPF_KS = KS_CV(data, HPF_count)
            temp_Frob.append(HPF_model.explained_deviance(data, HPF_count, 2))
            temp_KL.append(HPF_model.explained_deviance(data, HPF_count, 1))
            temp_KSCell.append(HPF_KS[0])
            temp_KSGene.append(HPF_KS[1])

        Explained_deviance["Frob"]["HPF_{}".format(k)] = np.mean(temp_Frob)
        Explained_deviance["KL"]["HPF_{}".format(k)] = np.mean(temp_KL)
        Explained_deviance["KS_cell"]["HPF_{}".format(k)] = np.mean(temp_KSCell)
        Explained_deviance["KS_gene"]["HPF_{}".format(k)] = np.mean(temp_KSGene)

        # TSNE
        Results["HPF_{}".format(k)]["tsne_cell"] = TSNE(n_jobs=20).fit_transform(HPF_U)
        Results["HPF_{}".format(k)]["tsne_gene"] = TSNE(n_jobs=20).fit_transform(HPF_V)

    return Results, Explained_deviance, Clustering


def MF_integration(data, genes_list, cell_labels, debug=True, **kwargs):

    startK = kwargs.get("startK", 2)
    endK = kwargs.get("endK", 10)
    step = kwargs.get("step", 2)
    noise_ratio = kwargs.get("noise", 0.1)

    PPC_simulation = 10

    no_cell_types = len(np.unique(cell_labels))

    print("There are {} cell types".format(no_cell_types))

    Results = dict()
    Results["CountMatrix"] = data
    Results["GenesList"] = genes_list

    print("There are genes ", genes_list)

    Explained_deviance = dict()
    Explained_deviance["Frob"] = dict()
    Explained_deviance["KL"] = dict()
    Explained_deviance["KS_cell"] = dict()
    Explained_deviance["KS_gene"] = dict()
    Explained_deviance["AMI_kmeans_normalised"] = dict()
    Explained_deviance["AMI_kmeans"] = dict()
    Explained_deviance["AMI_max"] = dict()
    Explained_deviance["AMI_scanorama"] = dict()
    Explained_deviance["SI"] = dict()
    Explained_deviance["SI_normalised"] = dict()
    Explained_deviance["SI_scanorama"] = dict()

    Clustering = dict()
    Clustering["kmeans_scanorama"] = dict()
    Clustering["kmeans_normalised"] = dict()
    Clustering["kmeans"] = dict()
    Clustering["max"] = dict()
    Clustering["actual"] = cell_labels

    ## Integrative HPF NMF

    for k in range(startK, endK + 1, step):

        ## Integrative HPF

        print("Integrative HPF {}".format(k))
        coo_data = [coo_matrix(d) for d in data]
        HPF_model = scHPFI(k, verbose=True)
        HPF_model.fit(coo_data, dataset_ratio=noise_ratio)

        Results["IHPF_{}".format(k)] = {
            "cell_score": HPF_model.cell_scores(),
            "shared_gene_score": HPF_model.shared_gene_scores(),
            "dataset_gene_score": HPF_model.gene_scores(),
        }

        # Clustering
        # consider the max entry for each cell in reduced dimension as cluster labels
        # TODO: other methods to standard data

        integrated_cells = np.concatenate(
            Results["IHPF_{}".format(k)]["cell_score"], axis=0
        )
        from sklearn.preprocessing import normalize

        normalised_integrated_cells = normalize(integrated_cells)

        Clustering["max"]["IHPF_{}".format(k)] = np.argmax(integrated_cells, axis=1)
        Clustering["kmeans"]["IHPF_{}".format(k)] = clustering_factors(
            integrated_cells, no_cell_types
        )
        Clustering["kmeans_normalised"]["IHPF_{}".format(k)] = clustering_factors(
            normalised_integrated_cells, no_cell_types
        )

        from sklearn import metrics

        Explained_deviance["AMI_max"][
            "IHPF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["max"]["IHPF_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans"][
            "IHPF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans"]["IHPF_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans_normalised"][
            "IHPF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans_normalised"]["IHPF_{}".format(k)], cell_labels
        )

        Explained_deviance["SI"]["IHPF_{}".format(k)] = metrics.silhouette_score(
            integrated_cells, cell_labels
        )
        Explained_deviance["SI_normalised"][
            "IHPF_{}".format(k)
        ] = metrics.silhouette_score(normalised_integrated_cells, cell_labels)

        # TODO: Normalise scores before TSNE?
        Results["IHPF_{}".format(k)]["tsne_cell"] = TSNE(n_jobs=20).fit_transform(
            normalised_integrated_cells
        )

        Frob, KLdiv, KSCell, KSGene = HPF_model.reconstruction_metrics(
            coo_data, PPC_simulation=5
        )
        Explained_deviance["Frob"]["IHPF_{}".format(k)] = Frob
        Explained_deviance["KL"]["IHPF_{}".format(k)] = KLdiv
        Explained_deviance["KS_cell"]["IHPF_{}".format(k)] = KSCell
        Explained_deviance["KS_gene"]["IHPF_{}".format(k)] = KSGene

        # Hierarchical clustering on the shared gene space

        print(Explained_deviance)

        #### Integrative NMF with normalised count data

        print("Integrative NMF {}".format(k))

        NMF_model = scINMF(k, alpha=1, verbose=True)

        # Fit normalised data
        from sklearn.preprocessing import normalize

        datasets = [normalize(csr_matrix(m)) for m in data]
        NMF_model.fit(datasets)

        Results["INMF_{}".format(k)] = {
            "cell_score": NMF_model.cell_score,
            "shared_gene_score": NMF_model.shared_gene_score,
            "dataset_gene_score": NMF_model.dataset_gene_score,
        }

        # Clustering
        # consider the max entry for each cell in reduced dimension as cluster labels
        # TODO: other methods to standard data
        integrated_cells = np.concatenate(NMF_model.cell_score, axis=0)
        from sklearn.preprocessing import normalize

        normalised_integrated_cells = normalize(integrated_cells)

        Clustering["max"]["INMF_{}".format(k)] = np.argmax(integrated_cells, axis=1)
        Clustering["kmeans"]["INMF_{}".format(k)] = clustering_factors(
            integrated_cells, no_cell_types
        )
        Clustering["kmeans_normalised"]["INMF_{}".format(k)] = clustering_factors(
            normalised_integrated_cells, no_cell_types
        )

        from sklearn import metrics

        Explained_deviance["AMI_max"][
            "INMF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["max"]["INMF_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans"][
            "INMF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans"]["INMF_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans_normalised"][
            "INMF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans_normalised"]["INMF_{}".format(k)], cell_labels
        )
        Explained_deviance["SI"]["INMF_{}".format(k)] = metrics.silhouette_score(
            integrated_cells, cell_labels
        )
        Explained_deviance["SI_normalised"][
            "INMF_{}".format(k)
        ] = metrics.silhouette_score(normalised_integrated_cells, cell_labels)

        # TODO: Normalise scores before TSNE?
        Results["INMF_{}".format(k)]["tsne_cell"] = TSNE(n_jobs=20).fit_transform(
            normalised_integrated_cells
        )

        ## Reconstruction metrics
        reconstructions = NMF_model.count_matrix()
        Frob_scores = []
        KL_scores = []
        KSCell_scores = []
        KSGene_scores = []
        for i in range(0, len(data)):
            sparseX = coo_matrix(data[i])
            reconstruct = coo_matrix(reconstructions[i])
            KC, KG = KS_CV(sparseX, reconstruct)
            KSCell_scores.append(KC)
            KSGene_scores.append(KG)
            Frob_scores.append(NMF_model.explained_deviance(sparseX, reconstruct, 2))
            KL_scores.append(NMF_model.explained_deviance(sparseX, reconstruct, 1))

        Explained_deviance["Frob"]["INMF_{}".format(k)] = Frob_scores
        Explained_deviance["KL"]["INMF_{}".format(k)] = KL_scores
        Explained_deviance["KS_cell"]["INMF_{}".format(k)] = KSCell_scores
        Explained_deviance["KS_gene"]["INMF_{}".format(k)] = KSGene_scores

        print(Explained_deviance)

        print("IPCA {}".format(k))

        # Train PCA model as usual
        PCA_model = scPCA(k)
        coo_data = [coo_matrix(d) for d in data]
        from scipy.sparse import vstack

        sparseX = vstack(coo_data)
        PCA_model.fit(
            sparseX,
        )
        Results["IPCA_{}".format(k)] = {
            "cell_score": PCA_model.cell_score,
            "shared_gene_score": PCA_model.gene_score,
        }

        integrated_cells = Results["IPCA_{}".format(k)]["cell_score"]
        from sklearn.preprocessing import normalize

        normalised_integrated_cells = normalize(integrated_cells)

        Clustering["max"]["IPCA_{}".format(k)] = np.argmax(integrated_cells, axis=1)
        Clustering["kmeans"]["IPCA_{}".format(k)] = clustering_factors(
            integrated_cells, no_cell_types
        )
        Clustering["kmeans_normalised"]["IPCA_{}".format(k)] = clustering_factors(
            normalised_integrated_cells, no_cell_types
        )

        from sklearn import metrics

        Explained_deviance["AMI_max"][
            "IPCA_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["max"]["IPCA_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans"][
            "IPCA_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans"]["IPCA_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans_normalised"][
            "IPCA_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans_normalised"]["IPCA_{}".format(k)], cell_labels
        )
        Explained_deviance["SI"]["IPCA_{}".format(k)] = metrics.silhouette_score(
            integrated_cells, cell_labels
        )
        Explained_deviance["SI_normalised"][
            "IPCA_{}".format(k)
        ] = metrics.silhouette_score(normalised_integrated_cells, cell_labels)

        # TODO: Normalise scores before TSNE?
        Results["IPCA_{}".format(k)]["tsne_cell"] = TSNE(n_jobs=20).fit_transform(
            normalised_integrated_cells
        )

        PCA_count = coo_matrix(PCA_model.count_matrix())
        Explained_deviance["Frob"]["IPCA_{}".format(k)] = PCA_model.explained_deviance(
            sparseX, 2
        )
        Explained_deviance["KL"]["IPCA_{}".format(k)] = PCA_model.explained_deviance(
            sparseX, 1
        )
        PCA_KS = KS_CV(sparseX, PCA_count)
        Explained_deviance["KS_cell"]["PCA_{}".format(k)] = PCA_KS[0]
        Explained_deviance["KS_gene"]["PCA_{}".format(k)] = PCA_KS[1]

        HPF_model = scHPF2(k)
        coo_data = [coo_matrix(d) for d in data]
        from scipy.sparse import vstack

        sparseX = vstack(coo_data)
        HPF_model.fit(sparseX, verbose=True)
        HPF_U = HPF_model.cell_score()
        HPF_V = HPF_model.gene_score()
        if debug:
            print("HPF running for {} Factors".format(k))

        Results["HPF_{}".format(k)] = {
            "cell_score": HPF_U,
            "gene_score": HPF_V,
        }

        integrated_cells = Results["HPF_{}".format(k)]["cell_score"]
        from sklearn.preprocessing import normalize

        normalised_integrated_cells = normalize(integrated_cells)

        Clustering["max"]["HPF_{}".format(k)] = np.argmax(integrated_cells, axis=1)
        Clustering["kmeans"]["HPF_{}".format(k)] = clustering_factors(
            integrated_cells, no_cell_types
        )
        Clustering["kmeans_normalised"]["HPF_{}".format(k)] = clustering_factors(
            normalised_integrated_cells, no_cell_types
        )

        from sklearn import metrics

        Explained_deviance["AMI_max"][
            "HPF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["max"]["HPF_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans"][
            "HPF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans"]["HPF_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans_normalised"][
            "HPF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans_normalised"]["HPF_{}".format(k)], cell_labels
        )
        Explained_deviance["SI"]["HPF_{}".format(k)] = metrics.silhouette_score(
            integrated_cells, cell_labels
        )
        Explained_deviance["SI_normalised"][
            "HPF_{}".format(k)
        ] = metrics.silhouette_score(normalised_integrated_cells, cell_labels)

        # TODO: Normalise scores before TSNE?
        Results["HPF_{}".format(k)]["tsne_cell"] = TSNE(n_jobs=20).fit_transform(
            normalised_integrated_cells
        )

        ## Integrative HPF with shared gene space from the usual HPF

        print("Integrative HPF 2 {}".format(k))
        coo_data = [coo_matrix(d) for d in data]
        HPF_model2 = scHPFI(k, verbose=True, beta=HPF_model.beta, eta=HPF_model.eta)
        HPF_model2.fit(coo_data, dataset_ratio=noise_ratio, freeze_shared_genes=True)

        Results["IHPF2_{}".format(k)] = {
            "cell_score": HPF_model2.cell_scores(),
            "shared_gene_score": HPF_model2.shared_gene_scores(),
            "dataset_gene_score": HPF_model2.gene_scores(),
        }

        # Clustering
        # consider the max entry for each cell in reduced dimension as cluster labels
        # TODO: other methods to standard data

        integrated_cells = np.concatenate(
            Results["IHPF2_{}".format(k)]["cell_score"], axis=0
        )
        from sklearn.preprocessing import normalize

        normalised_integrated_cells = normalize(integrated_cells)

        Clustering["max"]["IHPF2_{}".format(k)] = np.argmax(integrated_cells, axis=1)
        Clustering["kmeans"]["IHPF2_{}".format(k)] = clustering_factors(
            integrated_cells, no_cell_types
        )
        Clustering["kmeans_normalised"]["IHPF2_{}".format(k)] = clustering_factors(
            normalised_integrated_cells, no_cell_types
        )

        from sklearn import metrics

        Explained_deviance["AMI_max"][
            "IHPF2_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["max"]["IHPF2_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans"][
            "IHPF2_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans"]["IHPF2_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans_normalised"][
            "IHPF2_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans_normalised"]["IHPF2_{}".format(k)], cell_labels
        )

        Explained_deviance["SI"]["IHPF2_{}".format(k)] = metrics.silhouette_score(
            integrated_cells, cell_labels
        )
        Explained_deviance["SI_normalised"][
            "IHPF2_{}".format(k)
        ] = metrics.silhouette_score(normalised_integrated_cells, cell_labels)

        # TODO: Normalise scores before TSNE?
        Results["IHPF2_{}".format(k)]["tsne_cell"] = TSNE(n_jobs=20).fit_transform(
            normalised_integrated_cells
        )

        Frob, KLdiv, KSCell, KSGene = HPF_model2.reconstruction_metrics(
            coo_data, PPC_simulation=5
        )
        Explained_deviance["Frob"]["IHPF2_{}".format(k)] = Frob
        Explained_deviance["KL"]["IHPF2_{}".format(k)] = KLdiv
        Explained_deviance["KS_cell"]["IHPF2_{}".format(k)] = KSCell
        Explained_deviance["KS_gene"]["IHPF2_{}".format(k)] = KSGene

    return Results, Explained_deviance, Clustering


def MF_integration_hyper(data, genes_list, cell_labels, debug=True, **kwargs):

    startK = kwargs.get("startK", 10)
    endK = kwargs.get("endK", 10)
    step = kwargs.get("step", 2)
    PPC_simulation = 10

    no_cell_types = len(np.unique(cell_labels))

    print("There are {} cell types".format(no_cell_types))

    Results = dict()
    Results["CountMatrix"] = data
    Results["GenesList"] = genes_list

    print("There are genes ", genes_list)

    Explained_deviance = dict()
    Explained_deviance["Frob"] = dict()
    Explained_deviance["KL"] = dict()
    Explained_deviance["KS_cell"] = dict()
    Explained_deviance["KS_gene"] = dict()
    Explained_deviance["AMI_kmeans_normalised"] = dict()
    Explained_deviance["AMI_kmeans"] = dict()
    Explained_deviance["AMI_max"] = dict()
    Explained_deviance["AMI_scanorama"] = dict()
    Explained_deviance["SI"] = dict()
    Explained_deviance["SI_normalised"] = dict()
    Explained_deviance["SI_scanorama"] = dict()

    Clustering = dict()
    Clustering["kmeans_scanorama"] = dict()
    Clustering["kmeans_normalised"] = dict()
    Clustering["kmeans"] = dict()
    Clustering["max"] = dict()
    Clustering["actual"] = cell_labels

    for k in range(startK, endK + 1, step):

        # a = np.linspace(0.1, 1, num=2)
        a = np.array([0.1, 0.5, 1, 10])
        b = np.array([0.00001, 0.0001, 0.001, 0.01])

        hyper_parameter_space = np.concatenate((a, b), axis=None)

        for l in hyper_parameter_space:

            ## Integrative HPF
            print("Integrative HPF {}".format(k))
            # Use old HPF results to initilaise
            HPF_model = scHPFI(k, verbose=False)
            coo_data = [coo_matrix(d) for d in data]
            HPF_model.fit(coo_data, dataset_ratio=l)
            Results["IHPF_{}".format(l)] = {
                "cell_score": HPF_model.cell_scores(),
                "shared_gene_score": HPF_model.shared_gene_scores(),
                "dataset_gene_score": HPF_model.gene_scores(),
            }

            # Clustering
            # consider the max entry for each cell in reduced dimension as cluster labels
            # TODO: other methods to standard data

            integrated_cells = np.concatenate(
                Results["IHPF_{}".format(l)]["cell_score"], axis=0
            )
            from sklearn.preprocessing import normalize

            normalised_integrated_cells = normalize(integrated_cells)

            Clustering["max"]["IHPF_{}".format(l)] = np.argmax(integrated_cells, axis=1)
            Clustering["kmeans"]["IHPF_{}".format(l)] = clustering_factors(
                integrated_cells, no_cell_types
            )
            Clustering["kmeans_normalised"]["IHPF_{}".format(l)] = clustering_factors(
                normalised_integrated_cells, no_cell_types
            )

            from sklearn import metrics

            Explained_deviance["AMI_max"][
                "IHPF_{}".format(l)
            ] = metrics.adjusted_mutual_info_score(
                Clustering["max"]["IHPF_{}".format(l)], cell_labels
            )
            Explained_deviance["AMI_kmeans"][
                "IHPF_{}".format(l)
            ] = metrics.adjusted_mutual_info_score(
                Clustering["kmeans"]["IHPF_{}".format(l)], cell_labels
            )
            Explained_deviance["AMI_kmeans_normalised"][
                "IHPF_{}".format(l)
            ] = metrics.adjusted_mutual_info_score(
                Clustering["kmeans_normalised"]["IHPF_{}".format(l)], cell_labels
            )

            Explained_deviance["SI"]["IHPF_{}".format(l)] = metrics.silhouette_score(
                integrated_cells, cell_labels
            )
            Explained_deviance["SI_normalised"][
                "IHPF_{}".format(l)
            ] = metrics.silhouette_score(normalised_integrated_cells, cell_labels)

            # TSNE-clustering
            Results["IHPF_{}".format(l)]["tsne_cell"] = TSNE(n_jobs=40).fit_transform(
                normalised_integrated_cells
            )

            #### Integrative NMF with normalised count data

            print("Integrative NMF {}".format(k))

            NMF_model = scINMF(k, alpha=1 / l, verbose=True)

            # Fit normalised data
            from sklearn.preprocessing import normalize

            datasets = [normalize(csr_matrix(m)) for m in data]
            NMF_model.fit(datasets)

            Results["INMF_{}".format(l)] = {
                "cell_score": NMF_model.cell_score,
                "shared_gene_score": NMF_model.shared_gene_score,
                "dataset_gene_score": NMF_model.dataset_gene_score,
            }

            # Clustering
            # consider the max entry for each cell in reduced dimension as cluster labels
            # TODO: other methods to standard data
            integrated_cells = np.concatenate(NMF_model.cell_score, axis=0)
            from sklearn.preprocessing import normalize

            normalised_integrated_cells = normalize(integrated_cells)

            Clustering["max"]["INMF_{}".format(l)] = np.argmax(integrated_cells, axis=1)
            Clustering["kmeans"]["INMF_{}".format(l)] = clustering_factors(
                integrated_cells, no_cell_types
            )
            Clustering["kmeans_normalised"]["INMF_{}".format(l)] = clustering_factors(
                normalised_integrated_cells, no_cell_types
            )

            from sklearn import metrics

            Explained_deviance["AMI_max"][
                "INMF_{}".format(l)
            ] = metrics.adjusted_mutual_info_score(
                Clustering["max"]["INMF_{}".format(l)], cell_labels
            )
            Explained_deviance["AMI_kmeans"][
                "INMF_{}".format(l)
            ] = metrics.adjusted_mutual_info_score(
                Clustering["kmeans"]["INMF_{}".format(l)], cell_labels
            )
            Explained_deviance["AMI_kmeans_normalised"][
                "INMF_{}".format(l)
            ] = metrics.adjusted_mutual_info_score(
                Clustering["kmeans_normalised"]["INMF_{}".format(l)], cell_labels
            )
            Explained_deviance["SI"]["INMF_{}".format(l)] = metrics.silhouette_score(
                integrated_cells, cell_labels
            )
            Explained_deviance["SI_normalised"][
                "INMF_{}".format(l)
            ] = metrics.silhouette_score(normalised_integrated_cells, cell_labels)

    return Results, Explained_deviance, Clustering


def MF_integration_simulation(data, genes_list, cell_labels, debug=True, **kwargs):

    startK = kwargs.get("startK", 2)
    endK = kwargs.get("endK", 10)
    step = kwargs.get("step", 2)
    PPC_simulation = 10

    no_cell_types = len(np.unique(cell_labels))

    print("There are {} cell types".format(no_cell_types))

    Results = dict()
    Results["CountMatrix"] = data
    Results["GenesList"] = genes_list

    Explained_deviance = dict()
    Explained_deviance["Frob"] = dict()
    Explained_deviance["KL"] = dict()
    Explained_deviance["KS_cell"] = dict()
    Explained_deviance["KS_gene"] = dict()
    Explained_deviance["AMI_kmeans_normalised"] = dict()
    Explained_deviance["AMI_kmeans"] = dict()
    Explained_deviance["AMI_max"] = dict()
    Explained_deviance["AMI_scanorama"] = dict()
    Explained_deviance["SI"] = dict()
    Explained_deviance["SI_normalised"] = dict()
    Explained_deviance["SI_scanorama"] = dict()

    Clustering = dict()
    Clustering["kmeans_scanorama"] = dict()
    Clustering["kmeans_normalised"] = dict()
    Clustering["kmeans"] = dict()
    Clustering["max"] = dict()
    Clustering["actual"] = cell_labels

    for k in range(startK, endK + 1, step):

        ## Integrative HPF

        print("Integrative HPF {}".format(k))

        HPF_model = scHPFI(k, verbose=True)
        coo_data = [coo_matrix(d) for d in data]
        HPF_model.fit(coo_data, dataset_ratio=0.1)

        Results["IHPF_{}".format(k)] = {
            "cell_score": HPF_model.cell_scores(),
            "shared_gene_score": HPF_model.shared_gene_scores(),
            "dataset_gene_score": HPF_model.gene_scores(),
        }

        # Clustering
        # consider the max entry for each cell in reduced dimension as cluster labels
        # TODO: other methods to standard data

        integrated_cells = np.concatenate(
            Results["IHPF_{}".format(k)]["cell_score"], axis=0
        )
        from sklearn.preprocessing import normalize

        normalised_integrated_cells = normalize(integrated_cells)

        # import scanorama
        # datasets_dimred = scanorama.assemble(Results['IHPF_{}'.format(k)]['cell_score'], verbose=True, knn=20, sigma=15, approx=True, alpha=0.1, batch_size=5000,)
        # scanorama_integrated_cells = np.concatenate(datasets_dimred,axis=0)

        Clustering["max"]["IHPF_{}".format(k)] = np.argmax(integrated_cells, axis=1)
        Clustering["kmeans"]["IHPF_{}".format(k)] = clustering_factors(
            integrated_cells, no_cell_types
        )
        Clustering["kmeans_normalised"]["IHPF_{}".format(k)] = clustering_factors(
            normalised_integrated_cells, no_cell_types
        )
        # Clustering['kmeans_scanorama']['IHPF_{}'.format(k)] = clustering_factors(scanorama_integrated_cells,no_cell_types)

        from sklearn import metrics

        Explained_deviance["AMI_max"][
            "IHPF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["max"]["IHPF_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans"][
            "IHPF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans"]["IHPF_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans_normalised"][
            "IHPF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans_normalised"]["IHPF_{}".format(k)], cell_labels
        )
        # Explained_deviance['AMI_scanorama']['IHPF_{}'.format(k)] = metrics.adjusted_mutual_info_score(Clustering['kmeans_scanorama']['IHPF_{}'.format(k)],cell_labels)

        Explained_deviance["SI"]["IHPF_{}".format(k)] = metrics.silhouette_score(
            integrated_cells, cell_labels
        )
        Explained_deviance["SI_normalised"][
            "IHPF_{}".format(k)
        ] = metrics.silhouette_score(normalised_integrated_cells, cell_labels)
        # Explained_deviance['SI_scanorama']['IHPF_{}'.format(k)] = metrics.silhouette_score(scanorama_integrated_cells,cell_labels)

        # TSNE-clustering
        Results["IHPF_{}".format(k)]["tsne_cell"] = TSNE(n_jobs=20).fit_transform(
            normalised_integrated_cells
        )

        ## Gene factors

        #### Integrative NMF with normalised count data

        print("Integrative NMF {}".format(k))

        NMF_model = scINMF(k, alpha=1, verbose=True)

        # Fit normalised data
        from sklearn.preprocessing import normalize

        datasets = [normalize(csr_matrix(m)) for m in data]
        NMF_model.fit(datasets)

        Results["INMF_{}".format(k)] = {
            "cell_score": NMF_model.cell_score,
            "shared_gene_score": NMF_model.shared_gene_score,
            "dataset_gene_score": NMF_model.dataset_gene_score,
        }

        # Clustering
        # consider the max entry for each cell in reduced dimension as cluster labels
        # TODO: other methods to standard data
        integrated_cells = np.concatenate(NMF_model.cell_score, axis=0)
        from sklearn.preprocessing import normalize

        normalised_integrated_cells = normalize(integrated_cells)

        # import scanorama
        # datasets_dimred = scanorama.assemble(Results['INMF_{}'.format(k)]['cell_score'], verbose=True, knn=20, sigma=15, approx=True, alpha=0.1, batch_size=5000,)
        # scanorama_integrated_cells = np.concatenate(datasets_dimred,axis=0)

        Clustering["max"]["INMF_{}".format(k)] = np.argmax(integrated_cells, axis=1)
        Clustering["kmeans"]["INMF_{}".format(k)] = clustering_factors(
            integrated_cells, no_cell_types
        )
        Clustering["kmeans_normalised"]["INMF_{}".format(k)] = clustering_factors(
            normalised_integrated_cells, no_cell_types
        )
        # Clustering['kmeans_scanorama']['INMF_{}'.format(k)] = clustering_factors(scanorama_integrated_cells,no_cell_types)

        from sklearn import metrics

        Explained_deviance["AMI_max"][
            "INMF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["max"]["INMF_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans"][
            "INMF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans"]["INMF_{}".format(k)], cell_labels
        )
        Explained_deviance["AMI_kmeans_normalised"][
            "INMF_{}".format(k)
        ] = metrics.adjusted_mutual_info_score(
            Clustering["kmeans_normalised"]["INMF_{}".format(k)], cell_labels
        )

        Explained_deviance["SI"]["INMF_{}".format(k)] = metrics.silhouette_score(
            integrated_cells, cell_labels
        )
        Explained_deviance["SI_normalised"][
            "INMF_{}".format(k)
        ] = metrics.silhouette_score(normalised_integrated_cells, cell_labels)

        # Explained_deviance['AMI_scanorama']['INMF_{}'.format(k)] = metrics.adjusted_mutual_info_score(Clustering['kmeans_scanorama']['INMF_{}'.format(k)],cell_labels)
        # Explained_deviance['SI_scanorama']['INMF_{}'.format(k)] = metrics.silhouette_score(scanorama_integrated_cells,cell_labels)

    return Results, Explained_deviance, Clustering


if __name__ == "__main__":
    import likelihood

    X = np.random.poisson(5, size=(1000, 400))
    var1 = MF_factors(X, topgenes=200)
    print(var1)
    var1 = MF_hyperparameter(X, a=0.1, ap=2.0)
    print(var1)
