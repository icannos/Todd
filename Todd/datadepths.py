import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet as MCD


def reduction_matrix(X, estimator="classic", method="cholesky", cov_precomputed=None):

    if estimator == "classic":
        Sigma = np.cov(X.T)

    elif estimator == "MCD":
        cov = MCD().fit(X)
        Sigma = cov.covariance_
    elif estimator == "precomputed":
        Sigma = cov_precomputed

    # Cholesky tends when the dimension is high.
    if method == "cholesky":
        C = np.linalg.cholesky(Sigma)
        return np.linalg.inv(C.T)

    elif method == "SVD":
        u, s, _ = np.linalg.svd(Sigma)
        return u @ np.diag(np.sqrt(1 / s))


def cov_matrix(X, robust=False):
    """Compute the covariance matrix of X."""
    if robust:
        cov = MCD().fit(X)
        sigma = cov.covariance_
    else:
        sigma = np.cov(X.T)

    return sigma


def standardize(X, robust=False):
    """Compute the square inverse of the covariance matrix of X."""

    sigma = cov_matrix(X, robust)
    n_samples, n_features = X.shape
    rank = np.linalg.matrix_rank(X)

    if rank < n_features:
        pca = PCA(rank)
        pca.fit(X)
        X_transf = pca.transform(X)
        sigma = cov_matrix(X_transf)
    else:
        pca = None
        X_transf = X.copy()

    u, s, _ = np.linalg.svd(sigma)
    square_inv_matrix = u / np.sqrt(s)

    return X_transf @ square_inv_matrix, square_inv_matrix, pca


def sampled_sphere(K, d):
    np.random.seed(0)
    mean = np.zeros(d)
    identity = np.identity(d)
    U = np.random.multivariate_normal(mean=mean, cov=identity, size=K)
    return normalize(U)


class DataDepth:
    def __init__(self, K):
        self.K = K  # as a starter you can set it to 10 times the dimension

    def AI_IRW_ai(self, X, estimator="classic", method="SVD", X_test=None, U=None):
        return self.AI_IRW(X, True, estimator, method, X_test, U)

    def AI_IRW(
        self, X, AI=False, estimator="classic", method="cholesky", X_test=None, U=None
    ):
        """Compute the score of the average halfspace depth of X_test w.r.t. X
        Parameters
        ----------
        X : Array-like (n_samples, dimension)
                The training set.
        AI: str
            To choose the Affine-Invariant or the original formulation of IRW.
        n_directions : int
            The number of random directions to compute the score.
        X_test : The testing set where the depth is computed.
        U: Array-like (n_directions, dimension)
           If None, it sample directions from the unit sphere.
        Returns
        -------
        Array of float
            Depth score of each delement of X_test.
        """
        if AI:
            Y, sigma_square_inv, pca = standardize(X, False)
            if pca is not None:
                Y_test = pca.transform(X_test)
            else:
                Y_test = X_test

        else:
            Y_test = X_test.copy()
            Y = X.copy()

        # Simulate random directions on the unit sphere.
        n_samples, dim = Y.shape

        if U is None:
            U = sampled_sphere(self.K, dim)

        ################################################
        """
        Sigma_square_inv = reduction_matrix(X, estimator, method)


        if AI == True:
            if X_test is None:
                Y = X @ Sigma_square_inv
            else:
                Y = X @ Sigma_square_inv
                Y_test = X_test @ Sigma_square_inv
        else:
            if X_test is None:
                Y = X.copy()
            else:
                Y = X.copy()
                Y_test = X_test.copy()
        """

        # A faster implementation is given if one want to compute the depth of the training set.

        if X_test is None:
            z = np.arange(1, n_samples + 1)
            Depth = np.zeros((n_samples, self.K))

            Z = np.matmul(Y, U.T)
            A = np.matrix.argsort(Z, axis=0)

            for k in tqdm(range(self.K), "Projection"):
                Depth[A[:, k], k] = z

            Depth = Depth / (n_samples * 1.0)

            Depth_score = np.minimum(Depth, 1 - Depth)

        # The general implementation.
        else:
            n_samples_test, dim_test = Y_test.shape  ## TODO demander a guigui
            if dim_test != dim:
                print("error: dimension of X and X_test must be the same")

            Depth = np.zeros((n_samples_test, self.K))
            z = np.arange(1, n_samples_test + 1)  # TODO z is not used ?
            A = np.zeros((n_samples_test, self.K), dtype=int)
            Z = np.matmul(Y, U.T)

            Z2 = np.matmul(Y_test, U.T)

            Z.sort(axis=0)

            for k in tqdm(range(self.K), "Projection"):
                A[:, k] = np.searchsorted(a=Z[:, k], v=Z2[:, k], side="left")
                # print(z.shape)
                # print(Depth.shape)
                # print(A[:,k])
                Depth[:, k] = A[:, k]  # A[:,k]##Depth[:,k] =

            Depth = Depth / (n_samples * 1.0)

            Depth_score = np.minimum(Depth, 1 - Depth)

        return np.mean(Depth_score, axis=1)

    def halfspace_mass(self, X, psi=32, lamb=0.5, X_test=None, U=None):
        n, d = X.shape
        Score = np.zeros(n)

        mass_left = np.zeros(self.K)
        mass_right = np.zeros(self.K)
        s = np.zeros(self.K)

        if U is None:
            U = sampled_sphere(self.K, d)
        M = X @ U.T

        for i in tqdm(range(self.K), "Projection"):
            try:
                subsample = np.random.choice(np.arange(n), size=psi, replace=False)
            except:
                subsample = np.random.choice(np.arange(n), size=n - 1, replace=False)
            SP = M[subsample, i]
            max_i = np.max(SP)
            min_i = np.min(SP)
            mid_i = (max_i + min_i) / 2
            s[i] = (
                lamb * (max_i - min_i) * np.random.uniform()
                + mid_i
                - lamb * (max_i - min_i) / 2
            )
            mass_left[i] = (SP < s[i]).sum() / psi
            mass_right[i] = (SP > s[i]).sum() / psi
            Score += mass_left[i] * (M[:, i] < s[i]) + mass_right[i] * (M[:, i] > s[i])

        if X_test is None:
            return Score / self.K
        else:
            Score_test = np.zeros(len(X_test))
            M_test = X_test @ U.T
            for i in tqdm(range(self.K), "Projection"):
                Score_test += mass_left[i] * (M_test[:, i] < s[i]) + mass_right[i] * (
                    M_test[:, i] > s[i]
                )
            return Score_test / self.K

    def halfspace_depth(self, X, X_test=None, U=None):
        """Compute the score of the classical tukey depth of X_test w.r.t. X
        Parameters
        ----------
        X : Array-like
                The training set.
        n : int
            The number of random directions to compute the score.
        X_test : The testing set where the depth is computed.
        Returns
        -------
        Array of float
            Depth score of each delement of X_test.
        """

        # Simulate random directions on the unit sphere.

        if U is None:
            U = sampled_sphere(self.K, X.shape[1])
        ################################################

        # A faster implementation is given if one want to compute the depth of the training set.

        if X_test is None:
            z = np.arange(1, X.shape[0] + 1)
            Depth = np.zeros((X.shape[0], self.K))

            Z = np.matmul(X, U.T)
            A = np.matrix.argsort(Z, axis=0)

            for k in tqdm(range(self.K), "Projection"):
                Depth[A[:, k], k] = z

            Depth = Depth / (X.shape[0] * 1.0)

            Depth_score = np.minimum(Depth, 1 - Depth)

        # The general implementation.
        else:
            Depth = np.zeros((X_test.shape[0], self.K))
            z = np.arange(1, X_test.shape[0] + 1)
            A = np.zeros((X_test.shape[0], self.K), dtype=int)
            Z = np.matmul(X, U.T)

            Z2 = np.matmul(X_test, U.T)

            Z.sort(axis=0)

            for k in tqdm(range(self.K), "Projection"):
                A[:, k] = np.searchsorted(a=Z[:, k], v=Z2[:, k], side="left")
                Depth[:, k] = A[:, k]

            Depth = Depth / (X.shape[0] * 1.0)

            Depth_score = np.minimum(Depth, 1 - Depth)

        return np.amin(Depth, axis=1)

    def projection_depth(self, X, X_test=None):
        # Simulate random directions on the unit sphere.
        mean = np.zeros((X.shape[1]))
        cov = np.identity(X.shape[1])
        U = np.random.multivariate_normal(mean=mean, cov=cov, size=self.K)
        norm = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / norm
        ################################################

        Z = np.matmul(X, U.T)
        Z2 = np.matmul(X_test, U.T)
        Depth = np.zeros((X_test.shape[0], self.K))
        for d in tqdm(range(self.K), "Projection"):
            a = np.median(Z[:, d])
            for i in range(X_test.shape[0]):
                Depth[i, d] = np.absolute(Z2[i, d] - a)

        for i in range(X_test.shape[0]):
            Depth[i] = Depth[i] / np.median(Depth[i])

        return 1 / (1 + np.amax(Depth, axis=1))

    def compute_depths(self, X, X_test, depth_choice):
        assert depth_choice in [
            "half_space",
            "proj_depth",
            "int_w_halfs_pace",
            "int_w_halfs_pace_ai",
            "halfspace_mass",
        ]
        logger.info("Choosen depths %s", depth_choice)
        if depth_choice == "half_space":
            depth_method = self.halfspace_depth
        elif depth_choice == "proj_depth":
            depth_method = self.projection_depth
        elif depth_choice == "int_w_halfs_pace":
            depth_method = self.AI_IRW
        elif depth_choice == "int_w_halfs_pace_ai":
            depth_method = self.AI_IRW_ai
        elif depth_choice == "halfspace_mass":
            depth_method = self.halfspace_mass
        else:
            raise NotImplementedError
        return depth_method(X=X, X_test=X_test)


if __name__ == "__main__":
    import time
    import torch

    X_train = torch.randn(25000, 768).numpy()
    X_test = torch.randn(25000, 768).numpy()  # n_samples * latent dimention
    K = 10000  # number of direction 10/20 times the dimension
    depth = DataDepth(K)
    print("Starting")
    t0 = time.perf_counter()
    depth.AI_IRW(X=X_train, AI=True, X_test=X_test)
    t1 = time.perf_counter() - t0
    print("Time elapsed: ", t1)
