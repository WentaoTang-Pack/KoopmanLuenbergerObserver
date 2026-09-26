"""
kernel.py
=========

Defines the MercerKernel class for RKHS-based learning, together with
a fitted kernel ridge regression (KRR) model and a collection of
standard kernel factory functions.

Mathematical background
-----------------------
A Mercer kernel is a symmetric, positive semi-definite function
k : X × X → ℝ.  By the Moore–Aronszajn theorem it uniquely defines a
reproducing kernel Hilbert space (RKHS)  H_k  with inner product ⟨·,·⟩_k
satisfying the reproducing property  f(x) = ⟨f, k(·,x)⟩_k.

Kernel ridge regression (KRR) solves the regularised empirical risk

    min_{f ∈ H_k}  ‖Y − K α‖_F²  +  λ ‖f‖²_{H_k}

whose unique minimiser is  α = (K + λI)⁻¹ Y,  with predictions

    ŷ(x*) = k(x*, X_train) α.

Supported kernel arithmetic
---------------------------
Given kernels k₁ and k₂ and a scalar c > 0:

    k₁ + k₂      — sum kernel
    c * k₁        — scaled kernel
    k₁ * k₂      — product kernel
    k₁ ** n       — power kernel  (k(x,x')ⁿ; valid Mercer for integer n≥1)
    k₁ + c        — constant-shifted kernel
    c₁*k₁ + c₂*k₂ — linear combination (compose the above)

All arithmetic operations propagate vectorised Gram-matrix computations
where available, so there is no performance penalty for combining kernels.

Built-in kernels
----------------
    linear_kernel()
    polynomial_kernel(degree, gamma, coef0)
    rbf_kernel(sigma)
    laplacian_kernel(sigma)
    matern_kernel(nu, length_scale)          nu ∈ {0.5, 1.5, 2.5}

Example
-------
>>> import numpy as np
>>> from kernel import rbf_kernel, linear_kernel, polynomial_kernel
>>>
>>> # Build a composite kernel
>>> k = 2.0 * rbf_kernel(sigma=0.5) + polynomial_kernel(degree=2)
>>>
>>> # Gram matrix
>>> X = np.random.randn(20, 3)
>>> K = k.matrix(X)               # (20, 20), symmetric PSD
>>>
>>> # Kernel ridge regression
>>> Y = np.sin(X[:, 0]) + 0.1 * np.random.randn(20)
>>> model = k.fit(X, Y, regularization=1e-4)
>>> model.predict(np.zeros((5, 3)))
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Callable, Optional, Union
from scipy.linalg import cho_factor, cho_solve
from scipy.special import kv as _kv, gamma as _gamma


# ---------------------------------------------------------------------------
# KRR fitted model
# ---------------------------------------------------------------------------

class KRRModel:
    """
    Fitted kernel ridge regression model returned by :meth:`MercerKernel.fit`.

    Attributes
    ----------
    kernel : MercerKernel
        The kernel used for fitting.
    X_train : np.ndarray, shape (N, d)
        Training inputs.
    Y_train : np.ndarray, shape (N,) or (N, p)
        Training targets.
    alpha : np.ndarray, shape (N,) or (N, p)
        Dual coefficients α = (K + λI)⁻¹ Y.
    regularization : float
        Ridge parameter λ used during fitting.
    gram_matrix : np.ndarray, shape (N, N)
        Training Gram matrix K (cached from fitting).
    """

    def __init__(
        self,
        kernel:         'MercerKernel',
        X_train:        np.ndarray,
        Y_train:        np.ndarray,
        alpha:          np.ndarray,
        regularization: float,
        gram_matrix:    np.ndarray,
        _cho_factor:    tuple,              # Cholesky of (K + λI), private
    ) -> None:
        self.kernel         = kernel
        self.X_train        = X_train
        self.Y_train        = Y_train
        self.alpha          = alpha
        self.regularization = regularization
        self.gram_matrix    = gram_matrix
        self._cho           = _cho_factor

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predict at new inputs.

            ŷ(x*) = k(x*, X_train) α

        Parameters
        ----------
        X_new : array_like, shape (M, d) or (d,)

        Returns
        -------
        Y_pred : np.ndarray, shape (M,) or (M, p)
        """
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        K_cross = self.kernel.matrix(X_new, self.X_train)   # (M, N)
        return K_cross @ self.alpha

    # ------------------------------------------------------------------
    # Residuals and diagnostics
    # ------------------------------------------------------------------

    def training_residuals(self) -> np.ndarray:
        """
        In-sample residuals  r = Y_train − K α.

        Returns
        -------
        r : np.ndarray, same shape as Y_train
        """
        return self.Y_train - self.gram_matrix @ self.alpha

    def loo_residuals(self) -> np.ndarray:
        """
        Leave-one-out (LOO) residuals via the closed-form shortcut.

        For KRR the LOO residual at point i avoids re-fitting by using

            e_i^{LOO} = r_i / (1 − H_{ii})

        where r_i is the in-sample residual and H = K(K + λI)⁻¹ is the
        hat matrix.  Its diagonal satisfies

            H_{ii} = 1 − λ [(K + λI)⁻¹]_{ii}

        which is extracted from the cached Cholesky factorisation, so no
        additional O(N³) solve is needed.

        Returns
        -------
        loo_res : np.ndarray, same shape as Y_train
        """
        N   = self.X_train.shape[0]
        lam = self.regularization

        # (K + λI)⁻¹ via the stored Cholesky factor
        C       = cho_solve(self._cho, np.eye(N), check_finite=False)  # (N, N)
        H_diag  = 1.0 - lam * np.diag(C)                              # (N,)

        r = self.training_residuals()            # (N,) or (N, p)
        if r.ndim == 2:
            return r / H_diag[:, np.newaxis]
        return r / H_diag

    def loo_mse(self) -> float:
        """
        Mean-squared LOO prediction error (averaged over training points).

        Useful for selecting the regularisation parameter λ without a
        separate validation set.

        Returns
        -------
        float
        """
        e = self.loo_residuals()
        return float(np.mean(e ** 2))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    @property
    def n_train(self) -> int:
        """Number of training samples N."""
        return self.X_train.shape[0]

    def __repr__(self) -> str:
        out_dim = "scalar" if self.Y_train.ndim == 1 else f"p={self.Y_train.shape[1]}"
        return (
            f"KRRModel(kernel={self.kernel.name!r}, "
            f"N={self.n_train}, output={out_dim}, "
            f"λ={self.regularization:.2e})"
        )


# ---------------------------------------------------------------------------
# MercerKernel
# ---------------------------------------------------------------------------

class MercerKernel:
    """
    A Mercer (positive semi-definite) kernel function  k(x, x').

    Parameters
    ----------
    func : callable
        Pointwise evaluation.  Signature: ``func(x, x') -> float``
        where ``x``, ``x'`` are 1-D ``np.ndarray`` objects of equal length.
    name : str, optional
        Human-readable label (used in ``repr`` and arithmetic names).
    _matrix_func : callable, optional
        Vectorised Gram-matrix override.  Signature:
        ``_matrix_func(X, X') -> np.ndarray`` of shape ``(N, M)``,
        where rows of ``X`` and ``X'`` are data points.  When supplied,
        :meth:`matrix` dispatches to this function instead of the generic
        ``O(N·M)`` pointwise loop.  All built-in factory functions provide
        this, and kernel arithmetic propagates it automatically.

    Notes
    -----
    * The user is responsible for supplying a function that defines a valid
      Mercer kernel (symmetric and positive semi-definite).  The class does
      not verify positive semi-definiteness.
    * Kernel arithmetic (``+``, ``*``, scalar ``*``, ``**``) returns a new
      ``MercerKernel`` and composes the vectorised Gram-matrix functions so
      combined kernels remain efficient.
    """

    def __init__(
        self,
        func:          Callable,
        name:          Optional[str] = None,
        _matrix_func:  Optional[Callable] = None,
    ) -> None:
        if not callable(func):
            raise TypeError("'func' must be callable.")
        self._func         = func
        self.name          = name or "MercerKernel"
        self._matrix_func  = _matrix_func

    # ------------------------------------------------------------------
    # Pointwise evaluation
    # ------------------------------------------------------------------

    def __call__(
        self,
        x:       np.ndarray,
        x_prime: np.ndarray,
    ) -> float:
        """
        Evaluate  k(x, x').

        Parameters
        ----------
        x, x_prime : array_like, shape (d,)

        Returns
        -------
        float
        """
        x  = np.asarray(x,       dtype=float).ravel()
        xp = np.asarray(x_prime, dtype=float).ravel()
        return float(self._func(x, xp))

    # ------------------------------------------------------------------
    # Gram matrix
    # ------------------------------------------------------------------

    def matrix(
        self,
        X:        np.ndarray,
        X_prime:  Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the Gram (kernel) matrix.

        Parameters
        ----------
        X : array_like, shape (N, d)
            Row-major data matrix; each row is one sample.
        X_prime : array_like, shape (M, d), optional
            If provided, compute the cross-kernel matrix
            ``K[i, j] = k(X[i], X_prime[j])``.
            If ``None``, compute the symmetric matrix
            ``K[i, j] = k(X[i], X[j])`` exploiting symmetry for speed.

        Returns
        -------
        K : np.ndarray, shape (N, M)  or  (N, N)
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        symmetric = X_prime is None
        if symmetric:
            X_prime_arr = X
        else:
            X_prime_arr = np.atleast_2d(np.asarray(X_prime, dtype=float))

        # Fast path: use the vectorised matrix function when available
        if self._matrix_func is not None:
            return self._matrix_func(X, X_prime_arr)

        # Generic path: O(N·M) pointwise loop
        N = X.shape[0]
        M = X_prime_arr.shape[0]
        K = np.empty((N, M))
        f = self._func

        if symmetric:
            for i in range(N):
                for j in range(i, N):
                    v        = float(f(X[i], X[j]))
                    K[i, j]  = v
                    K[j, i]  = v
        else:
            for i in range(N):
                for j in range(M):
                    K[i, j] = float(f(X[i], X_prime_arr[j]))

        return K

    # ------------------------------------------------------------------
    # Kernel arithmetic
    # ------------------------------------------------------------------

    def __add__(
        self,
        other: Union['MercerKernel', float, int],
    ) -> 'MercerKernel':
        """
        Sum of two kernels  (k₁ + k₂)(x, x') = k₁(x, x') + k₂(x, x').
        A scalar c is treated as the constant kernel k(x,x') = c.
        """
        if isinstance(other, MercerKernel):
            f1, f2 = self._func, other._func
            k1, k2 = self, other
            name   = f"({self.name} + {other.name})"

            def _f(x, xp, _f1=f1, _f2=f2):
                return _f1(x, xp) + _f2(x, xp)

            def _m(X, Xp, _k1=k1, _k2=k2):
                return _k1.matrix(X, Xp) + _k2.matrix(X, Xp)

            return MercerKernel(_f, name=name, _matrix_func=_m)

        if isinstance(other, (int, float)):
            c = float(other)
            f, k = self._func, self
            name  = f"({self.name} + {c})"

            def _f(x, xp, _f=f, _c=c):
                return _f(x, xp) + _c

            def _m(X, Xp, _k=k, _c=c):
                return _k.matrix(X, Xp) + _c

            return MercerKernel(_f, name=name, _matrix_func=_m)

        return NotImplemented

    def __radd__(self, other: Union[float, int]) -> 'MercerKernel':
        """Support  c + k  (scalar on the left)."""
        return self.__add__(other)

    def __mul__(
        self,
        other: Union['MercerKernel', float, int],
    ) -> 'MercerKernel':
        """
        Product of two kernels  (k₁ · k₂)(x, x') = k₁(x, x') · k₂(x, x').
        Multiplying by a scalar c scales all kernel values by c.
        """
        if isinstance(other, MercerKernel):
            f1, f2 = self._func, other._func
            k1, k2 = self, other
            name   = f"({self.name} × {other.name})"

            def _f(x, xp, _f1=f1, _f2=f2):
                return _f1(x, xp) * _f2(x, xp)

            def _m(X, Xp, _k1=k1, _k2=k2):
                return _k1.matrix(X, Xp) * _k2.matrix(X, Xp)

            return MercerKernel(_f, name=name, _matrix_func=_m)

        if isinstance(other, (int, float)):
            c = float(other)
            f, k = self._func, self
            name  = f"({c} · {self.name})"

            def _f(x, xp, _f=f, _c=c):
                return _c * _f(x, xp)

            def _m(X, Xp, _k=k, _c=c):
                return _c * _k.matrix(X, Xp)

            return MercerKernel(_f, name=name, _matrix_func=_m)

        return NotImplemented

    def __rmul__(self, scalar: Union[float, int]) -> 'MercerKernel':
        """Support  c * k  (scalar on the left)."""
        return self.__mul__(scalar)

    def __pow__(self, n: Union[int, float]) -> 'MercerKernel':
        """
        Power kernel  (k ** n)(x, x') = k(x, x')ⁿ.

        For positive integer n this is a valid Mercer kernel (n-fold
        elementwise product of k with itself).  For non-integer n,
        positive semi-definiteness is not guaranteed in general.

        Parameters
        ----------
        n : int or float
            Must be positive.
        """
        if not isinstance(n, (int, float)) or n <= 0:
            raise ValueError("Kernel power must be a positive number.")
        f, k  = self._func, self
        name  = f"({self.name})^{n}"

        def _f(x, xp, _f=f, _n=n):
            return _f(x, xp) ** _n

        def _m(X, Xp, _k=k, _n=n):
            return _k.matrix(X, Xp) ** _n

        return MercerKernel(_f, name=name, _matrix_func=_m)

    # ------------------------------------------------------------------
    # Kernel ridge regression
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train:        np.ndarray,
        Y_train:        np.ndarray,
        regularization: float = 1e-6,
    ) -> KRRModel:
        """
        Fit kernel ridge regression on training data.

        Solves

            (K + λ I) α = Y,      K[i,j] = k(X_train[i], X_train[j])

        for the dual coefficients  α,  then wraps everything in a
        :class:`KRRModel` whose :meth:`~KRRModel.predict` method evaluates

            ŷ(x*) = k(x*, X_train) α.

        Parameters
        ----------
        X_train : array_like, shape (N, d)
            Training inputs.  Each row is one sample.
        Y_train : array_like, shape (N,) or (N, p)
            Training targets.  Multi-dimensional output (p columns) is
            handled by solving the p problems simultaneously.
        regularization : float, optional
            Ridge parameter λ > 0.  Larger values → smoother prediction
            with higher bias.  Use :meth:`~KRRModel.loo_mse` to guide
            selection.  Default: 1e-6.

        Returns
        -------
        model : KRRModel

        Notes
        -----
        The regularised system (K + λI) α = Y is solved via Cholesky
        factorisation, which is O(N³) and numerically stable for all
        λ > 0.  The factorisation is cached inside the returned model
        for efficient computation of :meth:`~KRRModel.loo_residuals`.

        Raises
        ------
        ValueError
            If dimensions are inconsistent or λ ≤ 0.
        RuntimeError
            If the Cholesky factorisation fails (e.g. kernel matrix
            is not numerically PSD; try a larger regularization).
        """
        X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
        Y_train = np.asarray(Y_train, dtype=float)

        if regularization <= 0.0:
            raise ValueError("'regularization' (λ) must be strictly positive.")
        N = X_train.shape[0]
        if Y_train.shape[0] != N:
            raise ValueError(
                f"X_train has {N} rows but Y_train has {Y_train.shape[0]} rows."
            )

        K = self.matrix(X_train)                    # (N, N)
        A = K + regularization * np.eye(N)           # symmetric PD

        try:
            cho = cho_factor(A, lower=False, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Cholesky factorisation of (K + λI) failed — the kernel "
                "matrix may not be numerically positive definite.  "
                f"Try a larger regularization value.  Error: {exc}"
            ) from exc

        alpha = cho_solve(cho, Y_train, check_finite=False)   # (N,) or (N, p)

        return KRRModel(
            kernel         = self,
            X_train        = X_train,
            Y_train        = Y_train,
            alpha          = alpha,
            regularization = regularization,
            gram_matrix    = K,
            _cho_factor    = cho,
        )

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"MercerKernel(name={self.name!r})"

    def __str__(self) -> str:
        return f"Mercer kernel: {self.name}"


# ---------------------------------------------------------------------------
# Built-in kernel factory functions
# ---------------------------------------------------------------------------

def linear_kernel() -> MercerKernel:
    """
    Linear kernel:  k(x, x') = x^T x'.

    The RKHS consists of all linear functions f(x) = w^T x.
    KRR with this kernel is equivalent to ridge regression in the primal.
    The Gram matrix is computed as  K = X X^T  (a single BLAS call).
    """
    def _f(x, xp):
        return float(np.dot(x, xp))

    def _m(X, Xp):
        return X @ Xp.T

    return MercerKernel(_f, name="Linear", _matrix_func=_m)


def polynomial_kernel(
    degree: int   = 2,
    gamma:  float = 1.0,
    coef0:  float = 1.0,
) -> MercerKernel:
    """
    Polynomial kernel:  k(x, x') = (γ x^T x' + c)^d.

    Parameters
    ----------
    degree : int ≥ 1
        Polynomial degree d.
    gamma : float
        Scale of the inner product (γ > 0).
    coef0 : float
        Inhomogeneity offset c (c > 0 mixes monomials of all degrees
        up to d; c = 0 gives a homogeneous polynomial kernel).
    """
    if degree < 1:
        raise ValueError("'degree' must be ≥ 1.")
    if gamma <= 0:
        raise ValueError("'gamma' must be positive.")

    d, g, c = int(degree), float(gamma), float(coef0)

    def _f(x, xp, _d=d, _g=g, _c=c):
        return float((_g * np.dot(x, xp) + _c) ** _d)

    def _m(X, Xp, _d=d, _g=g, _c=c):
        return (_g * (X @ Xp.T) + _c) ** _d

    return MercerKernel(
        _f,
        name=f"Polynomial(d={d}, γ={g}, c={c})",
        _matrix_func=_m,
    )


def rbf_kernel(sigma: float = 1.0) -> MercerKernel:
    """
    Radial basis function (Gaussian / squared-exponential) kernel:

        k(x, x') = exp( −‖x − x'‖² / (2σ²) )

    Parameters
    ----------
    sigma : float > 0
        Length-scale σ.  Larger σ → smoother functions in the RKHS.

    Notes
    -----
    The Gram matrix is computed via the identity
    ‖x_i − x_j‖² = ‖x_i‖² − 2 x_i^T x_j + ‖x_j‖²
    using a single matrix product (O(N² d)) instead of an O(N² d) loop.
    """
    if sigma <= 0:
        raise ValueError("'sigma' must be positive.")

    inv2s2 = 0.5 / (sigma ** 2)

    def _f(x, xp, _c=inv2s2):
        d = x - xp
        return float(np.exp(-np.dot(d, d) * _c))

    def _m(X, Xp, _c=inv2s2):
        X_sq   = np.einsum('ij,ij->i', X,  X)[:, np.newaxis]   # (N, 1)
        Xp_sq  = np.einsum('ij,ij->i', Xp, Xp)[np.newaxis, :]  # (1, M)
        sq_dist = X_sq + Xp_sq - 2.0 * (X @ Xp.T)
        return np.exp(-np.maximum(sq_dist, 0.0) * _c)           # clip for float safety

    return MercerKernel(_f, name=f"RBF(σ={sigma})", _matrix_func=_m)


def laplacian_kernel(sigma: float = 1.0) -> MercerKernel:
    """
    Laplacian (exponential) kernel using the L1 (Manhattan) norm:

        k(x, x') = exp( −‖x − x'‖₁ / σ )

    This kernel is less smooth than the RBF kernel and corresponds to
    functions in a Matérn RKHS with ν = 0.5 per coordinate.

    Parameters
    ----------
    sigma : float > 0
        Length-scale σ.
    """
    if sigma <= 0:
        raise ValueError("'sigma' must be positive.")

    def _f(x, xp, _s=sigma):
        return float(np.exp(-np.sum(np.abs(x - xp)) / _s))

    def _m(X, Xp, _s=sigma):
        # Broadcast: (N, 1, d) − (1, M, d)  → (N, M, d)  → sum over d
        L1 = np.sum(np.abs(X[:, np.newaxis, :] - Xp[np.newaxis, :, :]), axis=2)
        return np.exp(-L1 / _s)

    return MercerKernel(_f, name=f"Laplacian(σ={sigma})", _matrix_func=_m)


def matern_kernel(
    nu:           float = 1.5,
    length_scale: float = 1.0,
) -> MercerKernel:
    """
    Matérn kernel with half-integer smoothness parameter ν.

    Closed-form expressions are available for ν ∈ {0.5, 1.5, 2.5}:

    * ν = 0.5:  k(r) = exp(−r/l)
    * ν = 1.5:  k(r) = (1 + √3 r/l) exp(−√3 r/l)
    * ν = 2.5:  k(r) = (1 + √5 r/l + 5r²/(3l²)) exp(−√5 r/l)

    where r = ‖x − x'‖₂  and  l = length_scale.  As ν → ∞ the kernel
    converges to the RBF kernel.

    Parameters
    ----------
    nu : float
        Smoothness: must be 0.5, 1.5, or 2.5.
    length_scale : float > 0
        Length-scale l.
    """
    supported = {0.5, 1.5, 2.5}
    if nu not in supported:
        raise ValueError(
            f"Matérn ν must be in {supported}; got {nu}.  "
            "For other values use a custom MercerKernel."
        )
    if length_scale <= 0:
        raise ValueError("'length_scale' must be positive.")

    l = float(length_scale)

    if nu == 0.5:
        def _phi(r, _l=l): return np.exp(-r / _l)
    elif nu == 1.5:
        def _phi(r, _l=l):
            s = 1.7320508075688772 * r / _l  # √3
            return (1.0 + s) * np.exp(-s)
    else:  # nu == 2.5
        def _phi(r, _l=l):
            s = 2.23606797749979 * r / _l    # √5
            return (1.0 + s + s ** 2 / 3.0) * np.exp(-s)

    def _f(x, xp, _phi=_phi):
        return float(_phi(np.linalg.norm(x - xp)))

    def _m(X, Xp, _phi=_phi):
        X_sq   = np.einsum('ij,ij->i', X,  X)[:, np.newaxis]
        Xp_sq  = np.einsum('ij,ij->i', Xp, Xp)[np.newaxis, :]
        sq_dist = X_sq + Xp_sq - 2.0 * (X @ Xp.T)
        r = np.sqrt(np.maximum(sq_dist, 0.0))
        return _phi(r)

    return MercerKernel(
        _f,
        name=f"Matern(ν={nu}, l={l})",
        _matrix_func=_m,
    )


# ---------------------------------------------------------------------------
# Sobolev kernel
# ---------------------------------------------------------------------------

def sobolev_kernel(
    d:            int,
    s:            float,
    length_scale: float = 1.0,
) -> MercerKernel:
    """
    Sobolev kernel whose RKHS is (isomorphic to) the Sobolev space H^s(ℝ^d).

    The reproducing kernel of H^s(ℝ^d) — the space of L² functions whose
    weak derivatives up to order s are in L² — is

        k(x, x') = [2^{1−ν}/Γ(ν)] · (r/ℓ)^ν · K_ν(r/ℓ),

    where  r = ‖x − x'‖₂,   ν = s − d/2,   ℓ = length_scale,  and
    K_ν  is the modified Bessel function of the second kind
    (computed by scipy.special.kv for any real ν > 0).

    By the Sobolev embedding theorem, continuous point evaluation is
    bounded in H^s(ℝ^d) if and only if  s > d/2,  i.e. ν > 0.
    Larger s means smoother functions in the RKHS.

    The spectral (power) density of this kernel is

        Ŝ(ω) ∝ (ℓ⁻² + ‖ω‖²)^{−s},

    a power-law decay whose exponent s controls the smoothness.

    Relation to Matérn kernels
    --------------------------
    When ν = n + 1/2 for non-negative integer n, K_ν has a closed-form
    rational–exponential expression and the kernel coincides with the
    Matérn-ν kernel:

        d=1, s=1.0  → ν=0.5  Ornstein–Uhlenbeck  k(r) = exp(−r/ℓ)
        d=1, s=2.0  → ν=1.5  Matérn-3/2          k(r) = (1+r/ℓ)exp(−r/ℓ)
        d=1, s=3.0  → ν=2.5  Matérn-5/2          k(r) = (1+r/ℓ+r²/(3ℓ²))exp(−r/ℓ)
        d=2, s=2.0  → ν=1.0  (general Bessel)
        d=2, s=3.0  → ν=2.0  (general Bessel)
        d=3, s=2.0  → ν=0.5  same Bessel order as d=1, s=1

    As s → ∞ the kernel converges pointwise to the RBF/Gaussian kernel.

    Parameters
    ----------
    d : int
        Dimension of the input space ℝ^d.  Together with s it determines
        the Bessel order  ν = s − d/2.
    s : float
        Sobolev smoothness index.  Must satisfy  s > d/2  (equivalently
        ν > 0).  Does not need to be an integer or a half-integer.
    length_scale : float, optional
        Length-scale  ℓ > 0.  Controls the correlation range; larger ℓ
        means the kernel decays more slowly with distance.  Default 1.0.

    Returns
    -------
    MercerKernel
        A MercerKernel object with pointwise and vectorised Gram-matrix
        evaluation.  k(x, x) = 1 for all x (the kernel is normalised).

    Raises
    ------
    ValueError
        If d < 1, s ≤ d/2, or length_scale ≤ 0.

    Examples
    --------
    >>> import numpy as np
    >>> from kernel import sobolev_kernel
    >>>
    >>> # RKHS = H^2(R^1)  (twice weakly differentiable functions)
    >>> k = sobolev_kernel(d=1, s=2.0)
    >>> k(np.array([0.0]), np.array([0.0]))   # → 1.0
    1.0
    >>> k(np.array([0.0]), np.array([1.0]))   # k(0, 1) with ν=1.5
    0.5837...
    >>>
    >>> # KRR on H^3(R^2)
    >>> X = np.random.randn(50, 2)
    >>> model = sobolev_kernel(d=2, s=3.0).fit(X, np.sin(X[:, 0]))
    """
    if not (isinstance(d, int) and d >= 1):
        raise ValueError("'d' must be a positive integer (input dimension).")
    s   = float(s)
    nu  = s - d / 2.0
    if nu <= 0.0:
        raise ValueError(
            f"Smoothness index s={s} must exceed d/2 = {d / 2:.4g} so that "
            f"H^s(R^d) embeds into C(R^d).  Got ν = s − d/2 = {nu:.4g} ≤ 0."
        )
    if length_scale <= 0.0:
        raise ValueError("'length_scale' must be positive.")

    ell = float(length_scale)
    _c  = 2.0 ** (1.0 - nu) / _gamma(nu)   # normalisation: k(0) = 1

    # ------------------------------------------------------------------ #
    # Pointwise evaluation                                                 #
    # ------------------------------------------------------------------ #
    def _func(x, xp, _nu=nu, _ell=ell, _c=_c):
        r = np.linalg.norm(
            np.asarray(x, dtype=float) - np.asarray(xp, dtype=float)
        ) / _ell
        if r == 0.0:
            return 1.0
        return float(_c * r ** _nu * _kv(_nu, r))

    # ------------------------------------------------------------------ #
    # Vectorised Gram-matrix evaluation                                    #
    # ------------------------------------------------------------------ #
    def _m(X, Xp, _nu=nu, _ell=ell, _c=_c):
        # Squared Euclidean distances via the identity
        #   ‖xᵢ − xⱼ'‖² = ‖xᵢ‖² + ‖xⱼ'‖² − 2 xᵢ·xⱼ'
        X_sq  = np.einsum('ij,ij->i', X,  X)[:, np.newaxis]
        Xp_sq = np.einsum('ij,ij->i', Xp, Xp)[np.newaxis, :]
        sq    = np.maximum(X_sq + Xp_sq - 2.0 * (X @ Xp.T), 0.0)
        R     = np.sqrt(sq) / _ell            # (N, M) scaled distances

        # kv(ν, 0) = ∞ and 0^ν = 0 for ν > 0; handle diagonal (r=0) via mask.
        zero  = (R == 0.0)
        R_s   = np.where(zero, 1.0, R)        # safe substitute avoids kv(ν, 0)
        return np.where(zero, 1.0, _c * R_s ** _nu * _kv(_nu, R_s))

    return MercerKernel(
        _func,
        name=f"Sobolev(d={d}, s={s:.4g}, ℓ={ell:.4g})",
        _matrix_func=_m,
    )
