"""
luenberger_observer.py
======================

Defines the LuenbergerObserver class for state estimation of an
autonomous continuous-time dynamical system with output measurements.

Observer dynamics (coupled to an associated DynamicalSystem):

    dz/dt = A_z z(t) + L (y(t) − C_z z(t))
     x̂(t) = E_z z(t)

where  y(t) = h(x(t))  is the measured output produced by the plant.

The observer and the plant are integrated **simultaneously** as a single
enlarged ODE system

    d/dt [x; z] = [ f(x)                     ]
                  [ (A_z − L C_z) z + L h(x)  ]

so that a stiff solver (default: Radau) can adapt its step size to the
combined dynamics of both subsystems.

Dimension bookkeeping
---------------------
    n  = system.n_states   (dimension of x and x̂)
    p  = system.n_outputs  (dimension of y)
    q  = n_observer        (order of the observer; dimension of z)

    A_z : (q, q)  — observer state matrix
    C_z : (p, q)  — maps z to the output space for injection
    E_z : (n, q)  — decoding matrix; maps z to the state estimate
    L   : (q, p)  — observer gain (injection) matrix

Example
-------
Classical full-order Luenberger observer for a linear system
dx/dt = Ax, y = Cx, with observer gain L chosen by pole placement:

>>> import numpy as np
>>> from dynamical_system import DynamicalSystem
>>> from luenberger_observer import LuenbergerObserver
>>>
>>> A = np.array([[-1, 2], [-3, -4]])
>>> C = np.array([[1, 0]])
>>> L_gain = np.array([[5], [7]])   # from pole placement
>>>
>>> sys = DynamicalSystem(lambda x: A @ x, lambda x: C @ x,
...                       n_states=2, n_outputs=1, name="Linear2D")
>>>
>>> obs = LuenbergerObserver(
...     system = sys,
...     Az = A,          # A_z = A  (plant matrix in observer)
...     Cz = C,          # C_z = C  (output matrix in observer)
...     Ez = np.eye(2),  # x̂ = z   (full-state estimate)
...     L  = L_gain,
...     name = "FullOrderObs",
... )
>>> result = obs.simulate(x0=[1., -1.], z0=[0., 0.], t_span=(0, 5))
>>> result.error[:, -1]   # estimation error at final time
"""

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult   # for type hint only

from dynamical_system import DynamicalSystem


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_shape(name: str, arr: np.ndarray, expected: tuple[int, int]) -> None:
    """Raise a descriptive ValueError when arr does not have the expected shape."""
    if arr.ndim != 2 or arr.shape != expected:
        raise ValueError(
            f"Matrix '{name}' has shape {arr.shape}; expected {expected}."
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ObserverResult:
    """
    Container for the output of :meth:`LuenbergerObserver.simulate`.

    Attributes
    ----------
    t      : np.ndarray, shape (N,)
        Time points at which the solution was evaluated.
    x      : np.ndarray, shape (n_states, N)
        True system state trajectory.
    y      : np.ndarray, shape (n_outputs, N)
        System output trajectory  y = h(x).
    z      : np.ndarray, shape (n_observer, N)
        Observer state trajectory.
    x_hat  : np.ndarray, shape (n_states, N)
        State estimate  x̂ = E_z z.
    error  : np.ndarray, shape (n_states, N)
        Estimation error  e = x − x̂.
    success : bool
    message : str
    solver  : str
    _raw    : OdeResult
        Raw scipy result (gives access to dense solution, events, etc.).
    """
    t:       np.ndarray
    x:       np.ndarray
    y:       np.ndarray
    z:       np.ndarray
    x_hat:   np.ndarray
    error:   np.ndarray
    success: bool
    message: str
    solver:  str
    _raw:    OdeResult = field(repr=False)

    def __repr__(self) -> str:
        e_rms = float(np.sqrt(np.mean(self.error ** 2)))
        return (
            f"ObserverResult(solver={self.solver!r}, "
            f"t=[{self.t[0]:.4g}, {self.t[-1]:.4g}], "
            f"steps={len(self.t)}, "
            f"RMS_error={e_rms:.3e}, "
            f"success={self.success})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LuenbergerObserver:
    """
    Luenberger (or Koopman-Luenberger) observer for a DynamicalSystem.

    Observer dynamics:
        dz/dt = A_z z(t) + L (y(t) − C_z z(t))
         x̂(t) = E_z z(t)

    where  y(t) = h(x(t))  is the output of the associated plant.

    Parameters
    ----------
    system : DynamicalSystem
        The plant whose state is to be estimated.
    Az : array_like, shape (q, q)
        Observer state matrix.  q is the observer order.
    Cz : array_like, shape (p, q)
        Observer output matrix; maps the observer state z into the
        output space so that  C_z z  can be compared with y.
        p = system.n_outputs.
    Ez : array_like, shape (n, q)
        Decoding matrix; maps z to the state estimate x̂ = E_z z.
        n = system.n_states.
    L : array_like, shape (q, p)
        Observer gain (injection) matrix.
    name : str, optional
        Human-readable identifier.

    Notes
    -----
    * The observer RHS is re-written as
          dz/dt = (A_z − L C_z) z + L y
      and (A_z − L C_z) is pre-computed once at construction time.
    * For classical full-order linear design: A_z = A (plant matrix),
      C_z = C (plant output matrix), E_z = I_n.  The observer error
      then satisfies  ė = (A − LC) e,  so eigenvalues of (A_z − L C_z)
      determine convergence.
    * For Koopman-Luenberger design: q ≥ n is typical;  A_z, C_z, E_z
      come from a Koopman decomposition and may differ substantially
      from the plant matrices.
    """

    # Inherit solver catalogue from DynamicalSystem
    STIFF_SOLVERS    = DynamicalSystem.STIFF_SOLVERS
    EXPLICIT_SOLVERS = DynamicalSystem.EXPLICIT_SOLVERS
    ALL_SOLVERS      = DynamicalSystem.ALL_SOLVERS

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        system: DynamicalSystem,
        Az:     np.ndarray,
        Cz:     np.ndarray,
        Ez:     np.ndarray,
        L:      np.ndarray,
        name:   Optional[str] = None,
    ) -> None:
        if not isinstance(system, DynamicalSystem):
            raise TypeError("'system' must be a DynamicalSystem instance.")

        Az = np.asarray(Az, dtype=float)
        Cz = np.asarray(Cz, dtype=float)
        Ez = np.asarray(Ez, dtype=float)
        L  = np.asarray(L,  dtype=float)

        n = system.n_states    # state dimension
        p = system.n_outputs   # output dimension

        if Az.ndim != 2 or Az.shape[0] != Az.shape[1]:
            raise ValueError(
                f"Az must be square (2-D); got shape {Az.shape}."
            )
        q = Az.shape[0]        # observer order

        _require_shape("Cz", Cz, (p, q))
        _require_shape("Ez", Ez, (n, q))
        _require_shape("L",  L,  (q, p))

        self.system = system
        self.Az     = Az
        self.Cz     = Cz
        self.Ez     = Ez
        self.L      = L
        self.name   = name or f"LuenbergerObserver({system.name})"

        self._n_states   = n
        self._n_outputs  = p
        self._n_observer = q

        # Pre-compute once: (A_z - L C_z), the closed-loop observer matrix
        self._A_closed: np.ndarray = Az - L @ Cz   # shape (q, q)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_observer(self) -> int:
        """Order of the observer (dimension of z)."""
        return self._n_observer

    @property
    def error_dynamics_matrix(self) -> np.ndarray:
        """
        Return  A_z − L C_z  (a copy).

        In the linear Luenberger setting this matrix governs the error
        dynamics  ė = (A_z − L C_z) e;  its eigenvalues set the
        observer convergence rate.  A copy is returned so that the
        internally cached matrix is not accidentally mutated.
        """
        return self._A_closed.copy()

    # ------------------------------------------------------------------
    # Pointwise evaluations
    # ------------------------------------------------------------------

    def observer_vector_field(
        self,
        z: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the observer RHS at a single point.

            dz/dt = (A_z − L C_z) z + L y

        Parameters
        ----------
        z : array_like, shape (n_observer,)
        y : array_like, shape (n_outputs,)

        Returns
        -------
        dzdt : np.ndarray, shape (n_observer,)
        """
        z = np.asarray(z, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if z.shape != (self._n_observer,):
            raise ValueError(
                f"z has shape {z.shape}; expected ({self._n_observer},)."
            )
        if y.shape != (self._n_outputs,):
            raise ValueError(
                f"y has shape {y.shape}; expected ({self._n_outputs},)."
            )
        return self._A_closed @ z + self.L @ y

    def state_estimate(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the state estimate  x̂ = E_z z  at a single point.

        Parameters
        ----------
        z : array_like, shape (n_observer,)

        Returns
        -------
        x_hat : np.ndarray, shape (n_states,)
        """
        z = np.asarray(z, dtype=float).ravel()
        if z.shape != (self._n_observer,):
            raise ValueError(
                f"z has shape {z.shape}; expected ({self._n_observer},)."
            )
        return self.Ez @ z

    # ------------------------------------------------------------------
    # Coupled simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        x0:           np.ndarray,
        z0:           np.ndarray,
        t_span:       tuple[float, float],
        t_eval:       Optional[np.ndarray] = None,
        solver:       str   = 'Radau',
        rtol:         float = 1e-6,
        atol:         float = 1e-8,
        dense_output: bool  = False,
        **kwargs,
    ) -> ObserverResult:
        """
        Integrate the plant and the observer simultaneously.

        The enlarged state  ξ = [x ; z]  (length n + q)  is evolved by

            dξ/dt = [ f(x)                       ]
                    [ (A_z − L C_z) z + L h(x)   ]

        Parameters
        ----------
        x0 : array_like, shape (n_states,)
            Initial system state.
        z0 : array_like, shape (n_observer,)
            Initial observer state (the "prior" on z).
            Does not need to be consistent with x0; the observer will
            converge if the gain L is designed correctly.
        t_span : (t0, tf)
            Integration interval.
        t_eval : array_like of float, optional
            Times at which to store the solution.  If None the solver
            chooses its own adaptive grid.
        solver : str, optional
            ODE method.  Default 'Radau' (L-stable, handles stiff
            coupled systems including fast observer transients).
        rtol, atol : float
            Relative / absolute tolerances.
        dense_output : bool
            Compute a continuous solution (accessible via result._raw.sol).
        **kwargs
            Extra keyword arguments forwarded to scipy.integrate.solve_ivp
            (e.g. 'events', 'max_step', 'jac').

        Returns
        -------
        result : ObserverResult
            result.t      – time points, shape (N,)
            result.x      – true state,   shape (n_states, N)
            result.y      – output,       shape (n_outputs, N)
            result.z      – observer state, shape (n_observer, N)
            result.x_hat  – state estimate, shape (n_states, N)
            result.error  – e = x − x̂,   shape (n_states, N)

        Raises
        ------
        ValueError
            On dimension mismatches or invalid solver name.
        """
        n = self._n_states
        q = self._n_observer

        x0 = np.asarray(x0, dtype=float).ravel()
        z0 = np.asarray(z0, dtype=float).ravel()

        if x0.shape != (n,):
            raise ValueError(f"x0 has shape {x0.shape}; expected ({n},).")
        if z0.shape != (q,):
            raise ValueError(f"z0 has shape {z0.shape}; expected ({q},).")

        t0, tf = float(t_span[0]), float(t_span[1])
        if tf <= t0:
            raise ValueError(
                f"t_span[1]={tf} must be greater than t_span[0]={t0}."
            )
        if solver not in self.ALL_SOLVERS:
            raise ValueError(
                f"Unknown solver '{solver}'. "
                f"Choose from {sorted(self.ALL_SOLVERS)}."
            )

        xi0 = np.concatenate([x0, z0])   # combined initial state, length n+q

        # Cache references so the closure does not hold 'self' alive
        sys_f  = self.system._f
        sys_h  = self.system._h
        A_cl   = self._A_closed   # (q, q)
        L      = self.L           # (q, p)

        def _rhs(t: float, xi: np.ndarray) -> np.ndarray:
            x    = xi[:n]
            z    = xi[n:]
            y    = np.asarray(sys_h(x), dtype=float).ravel()
            dxdt = np.asarray(sys_f(x), dtype=float).ravel()
            dzdt = A_cl @ z + L @ y
            return np.concatenate([dxdt, dzdt])

        raw = solve_ivp(
            _rhs,
            (t0, tf),
            xi0,
            method       = solver,
            t_eval       = t_eval,
            rtol         = rtol,
            atol         = atol,
            dense_output = dense_output,
            **kwargs,
        )

        if not raw.success:
            warnings.warn(
                f"[{self.name}] Solver '{solver}' did not converge: {raw.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        N = raw.y.shape[1]
        x_traj    = raw.y[:n, :]                                          # (n, N)
        z_traj    = raw.y[n:, :]                                          # (q, N)
        y_traj    = np.column_stack(                                      # (p, N)
            [np.asarray(sys_h(x_traj[:, k]), dtype=float).ravel()
             for k in range(N)]
        )
        x_hat_traj = self.Ez @ z_traj                                     # (n, N)
        error_traj = x_traj - x_hat_traj                                  # (n, N)

        return ObserverResult(
            t       = raw.t,
            x       = x_traj,
            y       = y_traj,
            z       = z_traj,
            x_hat   = x_hat_traj,
            error   = error_traj,
            success = raw.success,
            message = raw.message,
            solver  = solver,
            _raw    = raw,
        )

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LuenbergerObserver("
            f"name={self.name!r}, "
            f"n_states={self._n_states}, "
            f"n_outputs={self._n_outputs}, "
            f"n_observer={self._n_observer})"
        )

    def __str__(self) -> str:
        eigs   = np.linalg.eigvals(self._A_closed)
        stable = all(e.real < 0 for e in eigs)
        eig_str = ", ".join(
            f"{e.real:+.3f}{e.imag:+.3f}j" for e in np.sort_complex(eigs)
        )
        return (
            f"Observer    : {self.name}\n"
            f"  Plant     : {self.system.name}  "
            f"(n_states={self._n_states}, n_outputs={self._n_outputs})\n"
            f"  Order  q  : {self._n_observer}\n"
            f"  eig(A_z − L·C_z) : [{eig_str}]\n"
            f"  Stable    : {stable}"
        )
