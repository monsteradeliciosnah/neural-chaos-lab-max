import numpy as np


class ESN:
    def __init__(self, n_inputs, n_res=500, spectral_radius=0.9, alpha=0.3, seed=42):
        rng = np.random.default_rng(seed)
        self.Win = rng.uniform(-0.5, 0.5, size=(n_res, n_inputs))
        W = rng.uniform(-0.5, 0.5, size=(n_res, n_res))
        # scale to spectral radius
        eig = max(abs(np.linalg.eigvals(W)))
        self.W = (W / eig) * spectral_radius
        self.alpha = alpha
        self.Wout = None

    def fit(self, U, Y, washout=100, reg=1e-6):
        N = U.shape[0]
        X = np.zeros((N, self.W.shape[0]))
        for t in range(1, N):
            X[t] = (1 - self.alpha) * X[t - 1] + self.alpha * np.tanh(
                self.Win @ U[t] + self.W @ X[t - 1]
            )
        Xtr = X[washout:]
        Ytr = Y[washout:]
        # ridge
        self.Wout = np.linalg.solve(
            Xtr.T @ Xtr + reg * np.eye(Xtr.shape[1]), Xtr.T @ Ytr
        )
        return self

    def predict(self, U, x0=None):
        N = U.shape[0]
        X = np.zeros((N, self.W.shape[0])) if x0 is None else x0
        Yhat = np.zeros((N, self.Wout.shape[1]))
        for t in range(1, N):
            X[t] = (1 - self.alpha) * X[t - 1] + self.alpha * np.tanh(
                self.Win @ U[t] + self.W @ X[t - 1]
            )
            Yhat[t] = X[t] @ self.Wout
        return Yhat
