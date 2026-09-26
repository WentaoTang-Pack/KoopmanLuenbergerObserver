"""
dynamical_system.py
===================

Autonomous, multi-dimensional, continuous-time dynamical system with
an output measurement map:

    dx/dt = f(x)
        y = h(x)

The class is designed for both non-stiff and stiff systems; it defaults
to the Radau implicit Runge–Kutta solver so that stiff problems work out
of the box without any solver-switching boilerplate.

Example
-------
>>> import numpy as np
>>> from dynamical_system import DynamicalSystem
>>>
>>> # Van der Pol oscillator (stiff for large mu)
>>> mu = 1000.0
>>> def f(x):
...     return np.array([x[1],
...                      mu * (1 - x[0]**2) * x[1] - x[0]])
>>>
>>> def h(x):
...     return np.array([x[0]])       # observe position only
>>>
>>> sys = DynamicalSystem(f, h, n_states=2, n_outputs=1, name="VanDerPol")
>>> result = sys.simulate(x0=[2.0, 0.0], t_span=(0.0, 3000.0))
>>> print(result.t[-1], result.y[0, -1])
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult   # for type hint only


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Container for the output of :meth:`DynamicalSystem.simulate`.

    Attributes
    ----------
    t : np.ndarray, shape (N,)
        Time points at which the solution was evaluated.
    x : np.ndarray, shape (n_states, N)
        State trajectory.
    y : np.ndarray, shape (n_outputs, N)
        Output trajectory (h evaluated at each column of x).
    success : bool
        True if the solver completed the integration without error.
    message : str
        Human-readable description of the termination condition.
    solver : str
        Name of the ODE solver that was used.
    _raw : OdeResult
        The raw :class:`scipy.integrate.OdeResult` object, available
        when you need the dense solution or event information.
    """
    t:       np.ndarray
    x:       np.ndarray
    y:       np.ndarray
    success: bool
    message: str
    solver:  str
    _raw:    OdeResult = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"SimulationResult(solver={self.solver!r}, "
            f"t=[{self.t[0]:.4g}, {self.t[-1]:.4g}], "
            f"steps={len(self.t)}, success={self.success})"
        )


# ---------------------------------------------------------------------------
# Data collection result container
# ---------------------------------------------------------------------------

@dataclass
class DataCollectionResult:
    """
    Container for the output of :meth:`DynamicalSystem.collect_data`.

    Attributes
    ----------
    X_data : np.ndarray, shape (n_states, n)
        Sampled initial states.  Each **column** is one sample.
    Y_data : np.ndarray, shape (n_outputs, n)
        Output values  h(x)  evaluated at each column of X_data.
    trajectories : list of SimulationResult, length n
        One trajectory per initial state, simulated for duration tau.
    tau : float
        Simulation duration used for every trajectory.
    side_lengths : np.ndarray, shape (n_states,)
        Side lengths L_i of the sampling hypercube
        (state dimension i drawn uniformly from [−L_i/2, L_i/2]).
    solver : str
        ODE solver used for all trajectories.
    t_eval : np.ndarray or None
        Time grid passed to the solver.  None when the solver used its
        own adaptive grid.  Required for X_traj / Y_traj.

    Derived properties
    ------------------
    n_samples     : int
    X_final       : np.ndarray, shape (n_states, n) — states at t = τ
    Y_final       : np.ndarray, shape (n_outputs, n) — outputs at t = τ
    snapshot_pairs: (X_data, X_final) — Koopman snapshot pair (Xk, Xk+1)
    X_traj        : np.ndarray, shape (n_states, N_t, n)  [requires t_eval]
    Y_traj        : np.ndarray, shape (n_outputs, N_t, n) [requires t_eval]
    """
    X_data:       np.ndarray
    Y_data:       np.ndarray
    trajectories: list                               # list[SimulationResult]
    tau:          float
    side_lengths: np.ndarray
    solver:       str
    t_eval:       Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        n  = self.X_data.shape[1]
        ns = self.X_data.shape[0]
        if np.all(self.side_lengths == self.side_lengths[0]):
            sl_str = f"[{self.side_lengths[0]:.4g}] × {ns}"
        else:
            sl_str = "[" + ", ".join(f"{v:.4g}" for v in self.side_lengths) + "]"
        grid = f", N_t={len(self.t_eval)}" if self.t_eval is not None else ""
        return (
            f"DataCollectionResult("
            f"n={n}, n_states={ns}, tau={self.tau:.4g}, "
            f"side_lengths={sl_str}, solver={self.solver!r}{grid})"
        )

    # ------------------------------------------------------------------ #
    # Basic accessors                                                      #
    # ------------------------------------------------------------------ #

    @property
    def n_samples(self) -> int:
        """Number of sampled trajectories."""
        return self.X_data.shape[1]

    # ------------------------------------------------------------------ #
    # Terminal snapshots                                                   #
    # ------------------------------------------------------------------ #

    @property
    def X_final(self) -> np.ndarray:
        """State at  t = τ  for every trajectory.  Shape (n_states, n)."""
        return np.column_stack([tr.x[:, -1] for tr in self.trajectories])

    @property
    def Y_final(self) -> np.ndarray:
        """Output  h(x(τ))  for every trajectory.  Shape (n_outputs, n)."""
        return np.column_stack([tr.y[:, -1] for tr in self.trajectories])

    @property
    def snapshot_pairs(self):
        """Koopman snapshot pair **(X_data, X_final)**.

        Returns the tuple  ``(Xk, Xk1)``  where
        ``Xk[:, i] = x_i``  and  ``Xk1[:, i] = Φ_τ(x_i)``
        (Φ_τ is the flow map at time τ).  These are the standard inputs
        for Extended DMD and related Koopman learning methods.

        Returns
        -------
        Xk  : np.ndarray, shape (n_states, n)
        Xk1 : np.ndarray, shape (n_states, n)
        """
        return self.X_data, self.X_final

    # ------------------------------------------------------------------ #
    # Full trajectory arrays  (available only with a common t_eval)        #
    # ------------------------------------------------------------------ #

    @property
    def X_traj(self) -> np.ndarray:
        """Full state trajectories on the common time grid.

        Shape: **(n_states, N_t, n)**.

        Requires that ``t_eval`` was passed to
        :meth:`DynamicalSystem.collect_data`.
        Access as ``data.X_traj[:, k, i]`` for state component k of
        trajectory i, or ``data.X_traj[:, :, i]`` for the full
        trajectory of sample i.
        """
        if self.t_eval is None:
            raise AttributeError(
                "X_traj is only available when t_eval was passed to "
                "collect_data().  Pass a t_eval array to enable this."
            )
        return np.stack([tr.x for tr in self.trajectories], axis=2)

    @property
    def Y_traj(self) -> np.ndarray:
        """Full output trajectories on the common time grid.

        Shape: **(n_outputs, N_t, n)**.

        Requires that ``t_eval`` was passed to
        :meth:`DynamicalSystem.collect_data`.
        """
        if self.t_eval is None:
            raise AttributeError(
                "Y_traj is only available when t_eval was passed to "
                "collect_data().  Pass a t_eval array to enable this."
            )
        return np.stack([tr.y for tr in self.trajectories], axis=2)

    # ------------------------------------------------------------------ #
    # Kernel matrix computation                                            #
    # ------------------------------------------------------------------ #

    def compute_kernel_matrices(
        self,
        kernel,
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the two kernel matrices for Koopman–Luenberger synthesis.

        Parameters
        ----------
        kernel : MercerKernel (or any object with a ``.matrix(X, Xp)`` method)
            The Mercer kernel κ.  ``kernel.matrix(X)`` must return the square
            Gram matrix for rows of *X*; ``kernel.matrix(X, Xp)`` must return
            the cross-kernel matrix.
        lam : float > 0
            The large positive constant λ.

        Returns
        -------
        K0 : np.ndarray, shape (n, n)
            Standard (Gram) kernel matrix::

                K0[i, j] = κ(x_i, x_j)

        M : np.ndarray, shape (n, n)
            Finite-time Yosida-approximated generator matrix (G' in the paper)::

                M[i, j] = λ² ∫₀^τ e^{−λt} κ(x_i(t), x_j) dt − λ κ(x_i, x_j)

            where ``x_i(t)`` is the simulated trajectory starting from
            ``x_i = X_data[:, i]``.  This is the matrix representation of
            the finite-time Yosida approximation
            ``A_{λ,τ} = λ² ∫₀^τ e^{−λt} T_t dt − λI`` of the Koopman
            generator, evaluated at the data points.

        Notes
        -----
        The integral is approximated via the trapezoidal rule using
        each trajectory's stored time grid.  When a common *t_eval* was
        provided, all trajectories share the same grid and the inner loop
        is vectorised over all *n* trajectories simultaneously (O(N_t × n²)
        kernel evaluations, O(n²) working memory).  Without *t_eval* the
        integration falls back to a per-trajectory loop that handles
        adaptive solver grids.
        """
        lam = float(lam)
        if lam <= 0.0:
            raise ValueError("'lam' must be positive.")

        n = self.n_samples
        X = self.X_data.T   # (n, n_states) — rows are samples

        # Standard kernel matrix  G[i,j] = κ(x_i, x_j)
        K0 = kernel.matrix(X)   # (n, n)

        # ∫₀^τ e^{−λt} κ(x_i(t), x_j) dt  for all (i, j)
        # Each kernel slice K_k is weighted by e^{−λ t_k} before integration.
        if self.t_eval is not None:
            # Shared time grid → vectorised over all n trajectories at once.
            # X_traj[:, k, :].T  is (n, n_states) at time t_k.
            t      = self.t_eval
            N_t    = len(t)
            intgr  = np.zeros((n, n))
            w_prev = np.exp(-lam * t[0])
            K_prev = w_prev * kernel.matrix(self.X_traj[:, 0, :].T, X)
            for k in range(1, N_t):
                w_curr  = np.exp(-lam * t[k])
                K_curr  = w_curr * kernel.matrix(self.X_traj[:, k, :].T, X)
                intgr  += 0.5 * (t[k] - t[k - 1]) * (K_prev + K_curr)
                K_prev  = K_curr
        else:
            # Per-trajectory integration — each trajectory may have a
            # different (adaptive) time grid.
            intgr = np.zeros((n, n))
            for i, tr in enumerate(self.trajectories):
                w   = np.exp(-lam * tr.t)            # (N_i,) exponential weights
                K_i = kernel.matrix(tr.x.T, X)       # (N_i, n): K_i[k,j]=κ(x_i(t_k),x_j)
                intgr[i, :] = np.trapz(w[:, None] * K_i, x=tr.t, axis=0)

        M = lam ** 2 * intgr - lam * K0
        return K0, M


# ---------------------------------------------------------------------------
# Observer gain synthesis
# ---------------------------------------------------------------------------

def synthesize_observer_gain(
    data,
    kernel,
    lam: float,
    k: float = 1.0,
    eps: float = 1e-4,
    delta: float = 1e-4,
    margin: float = 1e-3,
    solver=None,
    solver_opts: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Solve the LMI (Eq. 19) and compute the observer gain (Eq. 22).

    Parameters
    ----------
    data : DataCollectionResult
        Simulation data collected with trajectory storage enabled.
    kernel : MercerKernel
        The kernel κ used for the RKHS lifting.
    lam : float
        Large positive constant λ for the Yosida approximation.
    k : float, optional
        Positive scalar weight on the state-reconstruction term X^T X in the
        LMI (Eq. 19).  Default 1.0.
    eps : float, optional
        Tikhonov regularisation added to the Gram matrix G, i.e.
        G_reg = G + ε I.  Increase if the solver reports infeasibility or
        returns inaccurate results.  Default 1e-4.
    delta : float, optional
        Lower bound on the smallest eigenvalue of P̃ = GPG, enforced as
        ``Ptilde >> delta * I``.  Keeps P̃ strictly positive definite so that
        the gain recovery ``L = P̃^{-1} Ỹ`` is well-conditioned.
        Default 1e-4.
    margin : float, optional
        Strict stability margin added to the LMI:
        ``lmi_expr << -margin * I``.  Ensures A_cl is strictly Hurwitz
        despite finite solver precision.  Increase if A_cl is found to
        have near-zero or positive eigenvalues after synthesis.
        Default 1e-3.
    solver : str or None, optional
        CVXPY solver name (e.g. ``'SCS'``, ``'MOSEK'``).  ``None`` lets
        CVXPY choose automatically (typically SCS).
    solver_opts : dict or None, optional
        Extra keyword arguments forwarded verbatim to ``prob.solve()``.
        Example: ``{'max_iters': 50000, 'eps': 1e-5}`` for SCS.
    verbose : bool, optional
        Pass ``verbose=True`` to have the solver print its log.

    Notes on MOSEK robustness
    -------------------------
    If MOSEK fails or returns suspicious results, try the following in order:

    1. **Increase** ``eps`` (e.g. 1e-2): better-conditioned G → better-scaled A.
    2. **Increase** ``margin`` (e.g. 1e-2): more slack from the constraint boundary.
    3. **Reduce** ``k`` or ``n``: smaller problem, smaller solution norms.
    4. **Use SCS as fallback**: ``solver='SCS'`` is less accurate but much
       more robust; pass ``solver_opts={'eps': 1e-4, 'max_iters': 100000}``.
    The function already retries MOSEK automatically with tol=1e-4 when the
    first attempt returns ``'unknown'``.

    Returns
    -------
    result : dict with keys
        ``'Ptilde'`` – (n, n) Lyapunov matrix P̃ = GPG (SDP variable)
        ``'Ytilde'`` – (n, m) LMI gain variable Ỹ = GY (SDP variable)
        ``'L'``      – (n, m) innovation gain L = P̃^{-1}Ỹ = G^{-1}P^{-1}Y
        ``'A'``      – (n, n) drift matrix A = G^{-1}G'^T (precomputed)
        ``'K0'``     – (n, n) Gram matrix (un-regularised)
        ``'M'``      – (n, n) Yosida generator matrix G'
        ``'G_reg'``  – (n, n) regularised Gram matrix G + ε I
        ``'status'`` – solver status string

    Notes
    -----
    The LMI (Eq. 19) is solved via the substitution P̃ = GPG, Ỹ = GY, which
    transforms it into the standard Lyapunov observer form with A = G^{-1}G'^T:

    **Reformulated LMI**::

        P̃ A + A^T P̃ − Ỹ H − H^T Ỹ^T + k X^T X ⪯ 0

    **Reformulated constraint (Eq. 20)**::

        X̃ P̃ X̃^T ⪰ I_d,   P̃ ⪰ δ I     (X̃ = X G^{-1})

    **Objective**::

        minimise  tr(P̃)   over P̃ ∈ ℝ^{n×n}, Ỹ ∈ ℝ^{n×m}

    **Innovation gain**::

        L = P̃^{-1} Ỹ = G^{-1} P^{-1} Y     (n × m)

    The Lyapunov function v = η^T P̃ η = η^T GPG η guarantees stability of
    A_cl = A − L H = G^{-1}(G'^T − P^{-1}Y H).

    **Notation mapping between paper and code**

    ============  ===========  ==========================================
    Paper symbol  Code name    Shape / meaning
    ============  ===========  ==========================================
    G             K0 / G_reg   (n, n) Gram matrix (regularised in LMI)
    G'            M            (n, n) Yosida generator matrix
    A = G^{-1}G'  A            (n, n) drift matrix (precomputed)
    X             X_data       (d, n) state snapshots
    H             Y_data       (m, n) output snapshots
    P̃ = GPG      Ptilde       (n, n) Lyapunov matrix (SDP variable)
    Ỹ = GY       Ytilde       (n, m) LMI gain variable (SDP variable)
    L = P̃^{-1}Ỹ  L            (n, m) innovation gain
    ============  ===========  ==========================================
    """
    import warnings as _warnings

    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "CVXPY is required for observer gain synthesis.  "
            "Install it with:  pip install cvxpy"
        ) from e

    K0, M = data.compute_kernel_matrices(kernel, lam)

    n = data.n_samples
    X = data.X_data          # (d, n) — paper's X
    H = data.Y_data          # (m, n) — paper's H
    d, m = X.shape[0], H.shape[0]

    G_reg = K0 + eps * np.eye(n)   # regularised Gram matrix, paper's G

    # Pre-compute fixed matrices (outside the SDP, done once).
    # Substitution P̃ = GPG, Ỹ = GY transforms the outer-G LMI (Eq. 19) into
    # the standard Lyapunov observer form  P̃ A + A^T P̃ − Ỹ H − H^T Ỹ^T + k X^TX ⪯ 0
    # where A = G^{-1}G'^T is the Koopman drift matrix.
    A  = np.linalg.solve(G_reg, M.T)      # (n, n)  G^{-1} G'^T  — drift
    Xt = np.linalg.solve(G_reg, X.T).T   # (d, n)  X G^{-1}     — for Eq. 20

    # LMI variables: P̃ = GPG (n×n symmetric) and Ỹ = GY (n×m).
    Ptilde = cp.Variable((n, n), symmetric=True)
    Ytilde = cp.Variable((n, m))

    # Standard Lyapunov observer LMI (equivalent to paper's Eq. 19):
    #   P̃ A + A^T P̃ − Ỹ H − H^T Ỹ^T + k X^T X  ⪯  0
    # Lyapunov function v = η^T P̃ η = η^T (GPG) η, innovation gain L = P̃^{-1}Ỹ.
    lmi_expr = (
        Ptilde @ A
        + A.T @ Ptilde
        - Ytilde @ H
        - H.T @ Ytilde.T
        + k * (X.T @ X)
    )

    constraints = [
        lmi_expr << -margin * np.eye(n),       # stability + strict margin
        Xt @ Ptilde @ Xt.T >> np.eye(d),       # Eq. 20: X P X^T ⪰ I_d (equiv.)
        Ptilde >> delta * np.eye(n),            # keeps P̃ strictly PD for gain recovery
    ]

    prob = cp.Problem(cp.Minimize(cp.trace(Ptilde)), constraints)

    _solver = solver if solver is not None else "MOSEK"

    def _mosek_params(tol: float) -> dict:
        return {
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS":   tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS":   tol,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
            "MSK_IPAR_INTPNT_MAX_ITERATIONS":  800,
        }

    solve_kwargs: dict = {"solver": _solver, "verbose": verbose}
    if _solver == "MOSEK":
        # The outer-G LMI is ill-scaled (cond ∝ ‖G‖²‖P‖); start with 1e-5.
        solve_kwargs["mosek_params"] = _mosek_params(1e-5)
    if solver_opts is not None:
        solve_kwargs.update(solver_opts)

    _first_attempt_failed = False
    try:
        prob.solve(**solve_kwargs)
    except cp.error.SolverError:
        _first_attempt_failed = True

    # Automatic retry: if MOSEK raises SolverError or returns UNKNOWN/solver_error,
    # relax tolerances to 1e-4 and double the iteration budget.
    if _solver == "MOSEK" and (
        _first_attempt_failed or prob.status in ("unknown", "solver_error")
    ):
        _warnings.warn(
            "synthesize_observer_gain: MOSEK failed on first attempt; "
            "retrying with tol=1e-4 and 1600 iterations.",
            RuntimeWarning,
            stacklevel=2,
        )
        retry_kwargs = dict(solve_kwargs)
        retry_kwargs["mosek_params"] = _mosek_params(1e-4)
        retry_kwargs["mosek_params"]["MSK_IPAR_INTPNT_MAX_ITERATIONS"] = 1600
        prob.solve(**retry_kwargs)

    Ptilde_val = Ptilde.value
    Ytilde_val = Ytilde.value

    if Ptilde_val is None or Ytilde_val is None:
        raise RuntimeError(
            f"LMI solver did not return a feasible solution (status: {prob.status!r}).  "
            "Remedies: increase 'eps' (e.g. 1e-2), increase 'margin' (e.g. 1e-2), "
            "reduce 'k', or reduce 'n'."
        )

    if prob.status not in ("optimal", "optimal_inaccurate"):
        _warnings.warn(
            f"synthesize_observer_gain: solver returned '{prob.status}'.  "
            "Verify A_cl eigenvalues before using this result.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif prob.status == "optimal_inaccurate":
        _warnings.warn(
            "synthesize_observer_gain: solver returned 'optimal_inaccurate'.  "
            "The solution is likely usable but may violate the LMI by a small amount.  "
            "Increase 'eps' or 'margin' if A_cl has positive eigenvalues.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Innovation gain L = P̃^{-1} Ỹ = G^{-1} P^{-1} Y  (n × m).
    # P̃ is strictly PD (enforced by Ptilde >> delta*I).
    L_val = np.linalg.solve(Ptilde_val, Ytilde_val)

    return {
        "Ptilde": Ptilde_val,   # (n, n) Lyapunov matrix P̃ = GPG
        "Ytilde": Ytilde_val,   # (n, m) LMI gain variable Ỹ = GY
        "L":      L_val,        # (n, m) innovation gain L = P̃^{-1}Ỹ = G^{-1}P^{-1}Y
        "A":      A,            # (n, n) drift matrix G^{-1}G'^T (precomputed)
        "K0":     K0,
        "M":      M,
        "G_reg":  G_reg,
        "status": prob.status,
    }


# ---------------------------------------------------------------------------
# Observer simulation
# ---------------------------------------------------------------------------

def simulate_koopman_observer(
    result: dict,
    data,
    y_func,
    z0: np.ndarray,
    t_span: tuple,
    t_eval=None,
    solver: str = "Radau",
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> dict:
    """Simulate the Koopman–Luenberger observer (Eq. 23).

    The observer ODE from Eq. 23 is::

        G ż(t) = G'^T z(t) + (P^{-1}Y)(y(t) − H z(t))

    Pre-multiplying by G^{-1} gives the standard stiff ODE::

        ż(t) = A_cl z(t) + L_mat y(t)

    where::

        A_obs = G^{-1} G'^T              (open-loop Koopman matrix)
        L_mat = G^{-1} (P^{-1} Y)        (Luenberger gain)
        A_cl  = A_obs − L_mat H           (= G^{-1}(G'^T − P^{-1}Y H), pre-computed)

    The state estimate is recovered as (Eq. 23)::

        x̂(t) = X z(t)

    Parameters
    ----------
    result : dict
        Output of :func:`synthesize_observer_gain`.
    data : DataCollectionResult
        The same data used for synthesis (provides X_data, Y_data).
    y_func : callable
        Measured output as a function of time: ``y_func(t) -> np.ndarray``
        of shape ``(m,)``.
    z0 : np.ndarray, shape (n,)
        Initial observer state in the kernel basis.
    t_span : (t0, tf)
        Integration interval.
    t_eval : array_like or None, optional
        Times at which to store the solution.  If ``None``, the solver
        chooses its own internal steps.
    solver : str, optional
        ODE solver passed to :func:`scipy.integrate.solve_ivp`.  ``'Radau'``
        (the default) is the recommended choice for stiff problems.  ``'BDF'``
        is a good alternative.
    rtol, atol : float, optional
        Relative and absolute tolerances for the solver.

    Returns
    -------
    result : dict with keys
        ``'t'``     – (N_t,) time points
        ``'z'``     – (n, N_t) observer state trajectory
        ``'x_hat'`` – (d, N_t) state estimate trajectory
        ``'sol'``   – full :class:`scipy.integrate.OdeSolution` object
    """
    from scipy.integrate import solve_ivp

    # A = G^{-1}G'^T and L = P̃^{-1}Ỹ are pre-computed by synthesize_observer_gain.
    A_obs = result["A"]        # (n, n)  G^{-1} G'^T  (drift)
    L_mat = result["L"]        # (n, m)  P̃^{-1} Ỹ = G^{-1} P^{-1} Y  (gain)
    X     = data.X_data        # (d, n)
    H     = data.Y_data        # (m, n)

    # Closed-loop matrix, pre-computed so each ODE evaluation is O(n^2).
    # Eq. 23: ż = (A − L H) z + L y(t)
    A_cl = A_obs - L_mat @ H   # (n, n)  G^{-1}(G'^T − P^{-1}Y H)

    def _rhs(t, z):
        return A_cl @ z + L_mat @ y_func(t)

    sol = solve_ivp(
        _rhs, t_span, z0,
        method=solver,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    x_hat = X @ sol.y   # (d, N_t)  Eq. 23: x̂ = X z

    return {
        "t":     sol.t,
        "z":     sol.y,
        "x_hat": x_hat,
        "sol":   sol,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DynamicalSystem:
    """
    Autonomous, multi-dimensional, continuous-time dynamical system.

        dx/dt = f(x)          (state equation)
            y = h(x)          (output measurement map)

    Parameters
    ----------
    f : callable
        Vector field.  Signature: ``f(x) -> array_like`` of shape
        ``(n_states,)``.  ``x`` is a 1-D ``np.ndarray``.
    h : callable
        Output map.  Signature: ``h(x) -> array_like`` of shape
        ``(n_outputs,)``.  ``x`` is a 1-D ``np.ndarray``.
    n_states : int
        Dimension of the state space (length of x).
    n_outputs : int
        Dimension of the output space (length of y).
    name : str, optional
        Human-readable identifier for the system.
    validate : bool, optional
        If True (default), evaluate f and h at the origin during
        construction to catch shape errors early.

    Notes
    -----
    * ``f`` and ``h`` must be *pure* functions of ``x`` (no external
      time argument) because the system is autonomous.
    * For stiff problems, use the default solver ``'Radau'`` or switch
      to ``'BDF'`` or ``'LSODA'``.
    * Both ``f`` and ``h`` should accept and return plain ``np.ndarray``
      objects.  Returning a list or tuple is also acceptable; the class
      will coerce the result.
    """

    #: Stiff-capable ODE solvers available in scipy.
    STIFF_SOLVERS: frozenset[str]   = frozenset({'Radau', 'BDF', 'LSODA'})
    #: Explicit (non-stiff) ODE solvers available in scipy.
    EXPLICIT_SOLVERS: frozenset[str] = frozenset({'RK45', 'RK23', 'DOP853'})
    #: Union of all supported solver names.
    ALL_SOLVERS: frozenset[str]     = STIFF_SOLVERS | EXPLICIT_SOLVERS

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        f:         Callable[[np.ndarray], np.ndarray],
        h:         Callable[[np.ndarray], np.ndarray],
        n_states:  int,
        n_outputs: int,
        name:      Optional[str] = None,
        validate:  bool = True,
    ) -> None:
        if not callable(f):
            raise TypeError("'f' must be callable.")
        if not callable(h):
            raise TypeError("'h' must be callable.")
        if not isinstance(n_states, int) or n_states < 1:
            raise ValueError("'n_states' must be a positive integer.")
        if not isinstance(n_outputs, int) or n_outputs < 1:
            raise ValueError("'n_outputs' must be a positive integer.")

        self._f        = f
        self._h        = h
        self._n_states  = n_states
        self._n_outputs = n_outputs
        self.name      = name or "DynamicalSystem"

        if validate:
            self._validate_maps()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_states(self) -> int:
        """Dimension of the state vector x."""
        return self._n_states

    @property
    def n_outputs(self) -> int:
        """Dimension of the output vector y."""
        return self._n_outputs

    # ------------------------------------------------------------------
    # Pointwise map evaluations
    # ------------------------------------------------------------------

    def vector_field(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the vector field  f(x).

        Parameters
        ----------
        x : array_like, shape (n_states,)

        Returns
        -------
        dxdt : np.ndarray, shape (n_states,)
        """
        x = self._coerce_state(x)
        dxdt = np.asarray(self._f(x), dtype=float).ravel()
        if dxdt.shape != (self._n_states,):
            raise ValueError(
                f"f(x) returned shape {dxdt.shape}; "
                f"expected ({self._n_states},)."
            )
        return dxdt

    def output(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the output map  h(x).

        Parameters
        ----------
        x : array_like, shape (n_states,)

        Returns
        -------
        y : np.ndarray, shape (n_outputs,)
        """
        x = self._coerce_state(x)
        y = np.asarray(self._h(x), dtype=float).ravel()
        if y.shape != (self._n_outputs,):
            raise ValueError(
                f"h(x) returned shape {y.shape}; "
                f"expected ({self._n_outputs},)."
            )
        return y

    # ------------------------------------------------------------------
    # Numerical Jacobians  (central-difference)
    # ------------------------------------------------------------------

    def jacobian_f(
        self,
        x:   np.ndarray,
        eps: float = 1e-7,
    ) -> np.ndarray:
        """
        Numerical Jacobian of  f  at  x  via central differences.

        Returns
        -------
        A : np.ndarray, shape (n_states, n_states)
            A[i, j] = ∂f_i / ∂x_j  evaluated at x.
        """
        x = self._coerce_state(x)
        n = self._n_states
        A = np.empty((n, n))
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = eps
            A[:, j] = (
                self.vector_field(x + ej) - self.vector_field(x - ej)
            ) / (2.0 * eps)
        return A

    def jacobian_h(
        self,
        x:   np.ndarray,
        eps: float = 1e-7,
    ) -> np.ndarray:
        """
        Numerical Jacobian of  h  at  x  via central differences.

        Returns
        -------
        C : np.ndarray, shape (n_outputs, n_states)
            C[i, j] = ∂h_i / ∂x_j  evaluated at x.
        """
        x = self._coerce_state(x)
        n = self._n_states
        p = self._n_outputs
        C = np.empty((p, n))
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = eps
            C[:, j] = (
                self.output(x + ej) - self.output(x - ej)
            ) / (2.0 * eps)
        return C

    def linearize(
        self,
        x_eq: np.ndarray,
        eps:  float = 1e-7,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Linearize the system about an equilibrium (or any operating point).

        Computes the pair (A, C) where

            A = Df|_{x_eq}   ∈ ℝ^{n_states × n_states}
            C = Dh|_{x_eq}   ∈ ℝ^{n_outputs × n_states}

        so that the linearized dynamics are

            dξ/dt ≈ A ξ
                η ≈ C ξ

        with  ξ = x − x_eq  and  η = y − h(x_eq).

        Parameters
        ----------
        x_eq : array_like, shape (n_states,)
            Point at which to linearize (typically an equilibrium).
        eps : float, optional
            Step size for central-difference Jacobians.

        Returns
        -------
        A : np.ndarray, shape (n_states, n_states)
        C : np.ndarray, shape (n_outputs, n_states)
        """
        x_eq = self._coerce_state(x_eq)
        A = self.jacobian_f(x_eq, eps=eps)
        C = self.jacobian_h(x_eq, eps=eps)
        return A, C

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        x0:           np.ndarray,
        t_span:       tuple[float, float],
        t_eval:       Optional[np.ndarray] = None,
        solver:       str   = 'Radau',
        rtol:         float = 1e-6,
        atol:         float = 1e-8,
        dense_output: bool  = False,
        **kwargs,
    ) -> SimulationResult:
        """
        Integrate the system from ``x0`` over the interval ``t_span``.

        By default the implicit Radau method is used, which handles both
        stiff and non-stiff problems.  For severely stiff problems (e.g.
        large Lipschitz constants, multi-scale dynamics) you may prefer
        ``'BDF'``.  For smooth non-stiff problems ``'RK45'`` or
        ``'DOP853'`` are faster.

        Parameters
        ----------
        x0 : array_like, shape (n_states,)
            Initial state  x(t_span[0]).
        t_span : (t0, tf)
            Integration interval.  Must satisfy tf > t0.
        t_eval : array_like of float, optional
            Ordered times at which to record the solution.
            If None the solver's own adaptive grid is returned.
        solver : {'Radau', 'BDF', 'LSODA', 'RK45', 'RK23', 'DOP853'}
            ODE method.  'Radau' (default) is a 5th-order implicit
            Runge–Kutta method suitable for stiff systems.
        rtol : float, optional
            Relative tolerance.  Default: 1e-6.
        atol : float, optional
            Absolute tolerance.  Default: 1e-8.
        dense_output : bool, optional
            If True, compute a continuous solution object accessible at
            ``result._raw.sol``.  Useful for evaluating x at arbitrary
            times after the fact.
        **kwargs
            Additional keyword arguments forwarded verbatim to
            :func:`scipy.integrate.solve_ivp` (e.g. ``events``,
            ``jac``, ``max_step``).

        Returns
        -------
        result : SimulationResult
            * ``result.t``      – time points, shape (N,)
            * ``result.x``      – state trajectory, shape (n_states, N)
            * ``result.y``      – output trajectory, shape (n_outputs, N)
            * ``result.success`` – True if integration succeeded
            * ``result.message`` – solver termination message
            * ``result._raw``   – raw :class:`scipy.integrate.OdeResult`

        Raises
        ------
        ValueError
            If ``x0`` has the wrong length, ``t_span`` is inverted, or
            ``solver`` is not one of the supported methods.

        Examples
        --------
        Lorenz system, observe x-coordinate only:

        >>> sigma, rho, beta = 10.0, 28.0, 8/3
        >>> def f(x): return np.array([
        ...     sigma*(x[1]-x[0]),
        ...     x[0]*(rho-x[2]) - x[1],
        ...     x[0]*x[1] - beta*x[2],
        ... ])
        >>> def h(x): return x[:1]
        >>> sys = DynamicalSystem(f, h, 3, 1, name="Lorenz")
        >>> t_eval = np.linspace(0, 50, 5000)
        >>> result = sys.simulate([1, 0, 0], (0, 50), t_eval=t_eval)
        >>> result.y.shape
        (1, 5000)
        """
        x0 = self._coerce_state(x0)

        t0, tf = float(t_span[0]), float(t_span[1])
        if tf <= t0:
            raise ValueError(
                f"t_span[1]={tf} must be strictly greater than t_span[0]={t0}."
            )
        if solver not in self.ALL_SOLVERS:
            raise ValueError(
                f"Unsupported solver '{solver}'. "
                f"Choose from {sorted(self.ALL_SOLVERS)}."
            )

        def _rhs(t: float, x: np.ndarray) -> np.ndarray:          # noqa: ANN001
            # scipy passes t even for autonomous systems; we discard it.
            return self._f(x)

        raw = solve_ivp(
            _rhs,
            (t0, tf),
            x0,
            method=solver,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=dense_output,
            **kwargs,
        )

        if not raw.success:
            import warnings
            warnings.warn(
                f"[{self.name}] Solver '{solver}' did not converge: {raw.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Evaluate h at every stored state column
        N = raw.y.shape[1]
        y_traj = np.column_stack(
            [self.output(raw.y[:, k]) for k in range(N)]
        )

        return SimulationResult(
            t       = raw.t,
            x       = raw.y,
            y       = y_traj,
            success = raw.success,
            message = raw.message,
            solver  = solver,
            _raw    = raw,
        )

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_data(
        self,
        side_lengths: Union[float, np.ndarray],
        n:            int,
        tau:          float,
        t_eval:       Optional[np.ndarray] = None,
        solver:       str   = 'Radau',
        rtol:         float = 1e-6,
        atol:         float = 1e-8,
        rng=None,
        burn_in:      float = 0.0,
        **kwargs,
    ) -> DataCollectionResult:
        """
        Sample n random states in a hypercube and simulate from each.

        Two complementary datasets are produced in one call:

        1. **Static snapshot** — n states sampled uniformly in the
           axis-aligned hypercube  [−L_i/2, L_i/2]^{n_states}  together
           with their output values  Y_data = h(X_data).
        2. **Trajectory bundle** — one trajectory of duration τ starting
           from each sampled state.

        Parameters
        ----------
        side_lengths : float or array_like, shape (n_states,)
            Side lengths  L_i > 0  of the sampling hypercube.
            A scalar is broadcast to all state dimensions.
        n : int
            Number of initial states to sample.
        tau : float > 0
            Duration of each trajectory (time interval [0, τ]).
        t_eval : array_like of float, optional
            Ordered times in [0, τ] at which to store every trajectory.
            When provided, ``result.X_traj`` and ``result.Y_traj`` are
            available (shape  (n_states, N_t, n)  and
            (n_outputs, N_t, n)).  If None the solver uses its own
            adaptive grid.
        solver : str, optional
            ODE method.  Default 'Radau'.
        rtol, atol : float
            Relative / absolute solver tolerances.
        rng : int, np.random.Generator, or None
            Seed or Generator for reproducible sampling.
            None → a fresh, unseeded Generator.
        **kwargs
            Extra keyword arguments forwarded to
            :func:`scipy.integrate.solve_ivp`.

        Returns
        -------
        result : DataCollectionResult
            result.X_data         shape (n_states, n)
            result.Y_data         shape (n_outputs, n)
            result.trajectories   list of n SimulationResult objects
            result.X_final        shape (n_states, n)   — states at t=τ
            result.Y_final        shape (n_outputs, n)  — outputs at t=τ
            result.snapshot_pairs (X_data, X_final) Koopman pair (Xk, Xk+1)
            result.X_traj         shape (n_states, N_t, n)  [if t_eval given]
            result.Y_traj         shape (n_outputs, N_t, n) [if t_eval given]

        Raises
        ------
        ValueError
            On invalid inputs.

        Examples
        --------
        >>> import numpy as np
        >>> from dynamical_system import DynamicalSystem
        >>>
        >>> sys = DynamicalSystem(
        ...     lambda x: np.array([x[1], -4*x[0]]),
        ...     lambda x: x[:1],
        ...     n_states=2, n_outputs=1, name="SHO",
        ... )
        >>> t_grid = np.linspace(0, 1, 50)
        >>> data = sys.collect_data(
        ...     side_lengths=2.0,   # each dim ∈ [−1, 1]
        ...     n=200, tau=1.0, t_eval=t_grid, rng=0,
        ... )
        >>> data.X_data.shape      # (2, 200)
        (2, 200)
        >>> data.X_traj.shape      # (2, 50, 200)
        (2, 50, 200)
        >>> Xk, Xk1 = data.snapshot_pairs
        """
        # ── Input validation ──────────────────────────────────────────
        n = int(n)
        if n < 1:
            raise ValueError("'n' must be a positive integer.")
        tau = float(tau)
        if tau <= 0.0:
            raise ValueError("'tau' must be positive.")

        sl = np.asarray(side_lengths, dtype=float).ravel()
        if sl.size == 1:
            sl = np.full(self._n_states, sl[0])
        elif sl.size != self._n_states:
            raise ValueError(
                f"'side_lengths' has {sl.size} element(s); "
                f"expected {self._n_states} (= n_states)."
            )
        if np.any(sl <= 0.0):
            raise ValueError("All entries of 'side_lengths' must be positive.")

        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)   # handles None and int seeds

        # ── Uniform sampling in the hypercube ─────────────────────────
        # samples: (n, n_states);  X_data: (n_states, n)
        X_data = rng.uniform(-sl / 2.0, sl / 2.0,
                             size=(n, self._n_states)).T

        # ── Optional burn-in: advance each IC to the attractor ────────
        if burn_in > 0.0:
            X_burned = np.empty_like(X_data)
            for i in range(n):
                tr_b = self.simulate(
                    X_data[:, i],
                    t_span=(0.0, float(burn_in)),
                    solver=solver, rtol=rtol, atol=atol,
                    **kwargs,
                )
                X_burned[:, i] = tr_b.x[:, -1]   # .x = state, .y = output
            X_data = X_burned

        # ── Output at each sampled state ──────────────────────────────
        Y_data = np.column_stack(
            [self.output(X_data[:, i]) for i in range(n)]
        )                                          # (n_outputs, n)

        # ── Simulate a trajectory from each sample ────────────────────
        trajectories = []
        for i in range(n):
            tr = self.simulate(
                X_data[:, i],
                t_span  = (0.0, tau),
                t_eval  = t_eval,
                solver  = solver,
                rtol    = rtol,
                atol    = atol,
                **kwargs,
            )
            trajectories.append(tr)

        return DataCollectionResult(
            X_data       = X_data,
            Y_data       = Y_data,
            trajectories = trajectories,
            tau          = tau,
            side_lengths = sl,
            solver       = solver,
            t_eval       = t_eval,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce_state(self, x: np.ndarray) -> np.ndarray:
        """Cast x to a float64 1-D array and check its length."""
        x = np.asarray(x, dtype=float).ravel()
        if x.shape != (self._n_states,):
            raise ValueError(
                f"State vector has shape {x.shape}; "
                f"expected ({self._n_states},)."
            )
        return x

    def _validate_maps(self) -> None:
        """Smoke-test f and h at the origin to catch shape errors early."""
        origin = np.zeros(self._n_states)

        try:
            fx = np.asarray(self._f(origin), dtype=float).ravel()
        except Exception as exc:
            raise ValueError(
                f"Calling f(zeros({self._n_states})) raised: {exc}"
            ) from exc
        if fx.shape != (self._n_states,):
            raise ValueError(
                f"f(zeros) returned shape {fx.shape}; "
                f"expected ({self._n_states},)."
            )

        try:
            hx = np.asarray(self._h(origin), dtype=float).ravel()
        except Exception as exc:
            raise ValueError(
                f"Calling h(zeros({self._n_states})) raised: {exc}"
            ) from exc
        if hx.shape != (self._n_outputs,):
            raise ValueError(
                f"h(zeros) returned shape {hx.shape}; "
                f"expected ({self._n_outputs},)."
            )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DynamicalSystem("
            f"name={self.name!r}, "
            f"n_states={self._n_states}, "
            f"n_outputs={self._n_outputs})"
        )

    def __str__(self) -> str:
        stiff_note = "(use solver='Radau'/'BDF' for stiff problems)"
        return (
            f"System : {self.name}\n"
            f"  States  : {self._n_states}\n"
            f"  Outputs : {self._n_outputs}\n"
            f"  Note    : {stiff_note}"
        )
