
"""
Lie-group inverse dynamics for serial manipulators
Based on A. Müller, "Recursive Second-Order Inverse Dynamics for Serial
Manipulators", ICRA 2017.

This file implements the ordinary (zeroth-order) inverse-dynamics recursion
from the paper, using the BODY-FIXED Lie-group representation:
    C_i = C_{i-1} B_i exp(X_i q_i)
    V_i = Ad_{C_{i,i-1}} V_{i-1} + X_i qdot_i
    Vdot_i = Ad_{C_{i,i-1}} Vdot_{i-1}
             - qdot_i ad_{X_i} V_i + X_i qddot_i
    W_i = Ad^T_{C_{i+1,i}} W_{i+1}
          + M_i Vdot_i - ad^T_{V_i} M_i V_i + W_app_i
    Q_i = X_i^T W_i

The implementation is deliberately kept in one file and is intended to have
the same execution pattern as a conventional RNEA implementation:

    robot = make_planar_robot(n)
    tau = robot.inverse_dynamics(q, qd, qdd)
    tau_all = robot.inverse_dynamics_trajectory(q, qd, qdd)

CSV input can contain columns:
    q1 ... qn qd1 ... qdn qdd1 ... qddn

or rows:
    q
    qd
    qdd
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import os

try:
    from scipy.linalg import expm
except ImportError as exc:
    raise ImportError(
        "This implementation requires SciPy for the SE(3) exponential map."
    ) from exc


EPS = 1e-12


# ---------------------------------------------------------------------------
# Basic Lie-group / screw algebra
# ---------------------------------------------------------------------------

def load_joint_data(npy_filename, n, time_int, filetype):
    if filetype == 'csv':
        data = np.loadtxt(npy_filename, delimiter=',', skiprows=1)
        data = data.T
        # print(np.shape(data))
        q = np.zeros((time_int, n))
        # print(np.shape(q))
        qd = np.zeros((time_int, n))
        qdd = np.zeros((time_int, n))
    elif filetype == 'npy':
        data = np.load(npy_filename, allow_pickle=True)
        data = data[0] if isinstance(data[0], list) else data
        q = np.vstack((data[0]))
        qd = np.vstack((data[1])) 
        qdd = np.vstack((data[2]))
        return q, qd, qdd

    for i in range(n):
        q[:, i] = data[i]

    for i in range(n, 2 * n):
        qd[:, i - n] = data[i]

    for i in range(2 * n, 3 * n):
        qdd[:, i - 2 * n] = data[i]

    return q, qd, qdd

def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew matrix such that skew(v) @ x == np.cross(v, x)."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def ad_matrix(V: np.ndarray) -> np.ndarray:
    """
    Lie bracket matrix ad_V for a twist V = [omega; v].

    [V1,V2] = ad_V1 V2
            = [omega1 x omega2;
               v1 x omega2 + omega1 x v2].
    """
    V = np.asarray(V, dtype=float).reshape(6)
    omega = V[:3]
    v = V[3:]
    return np.block([
        [skew(omega), np.zeros((3, 3))],
        [skew(v),      skew(omega)],
    ])


def se3_from_twist(X: np.ndarray, q: float) -> np.ndarray:
    """Return exp([X] q) represented as a 4x4 homogeneous transform."""
    X = np.asarray(X, dtype=float).reshape(6)
    xi_hat = np.zeros((4, 4))
    xi_hat[:3, :3] = skew(X[:3])
    xi_hat[:3, 3] = X[3:]
    return expm(xi_hat * q)


def make_transform(R: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Construct C = [[R,r],[0,1]] in SE(3)."""
    C = np.eye(4)
    C[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    C[:3, 3] = np.asarray(r, dtype=float).reshape(3)
    return C


def transform_inverse(C: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) homogeneous transformation."""
    R = C[:3, :3]
    r = C[:3, 3]
    Cinv = np.eye(4)
    Cinv[:3, :3] = R.T
    Cinv[:3, 3] = -R.T @ r
    return Cinv


def adjoint(C: np.ndarray) -> np.ndarray:
    """
    Ad_C from the paper, Eq. (26):

        Ad_C = [[R, 0],
                [r^ R, R]]
    """
    R = C[:3, :3]
    r = C[:3, 3]
    return np.block([
        [R, np.zeros((3, 3))],
        [skew(r) @ R, R],
    ])


def relative_transform(C_child: np.ndarray, C_parent: np.ndarray) -> np.ndarray:
    """
    C_{i,i-1} = C_i^{-1} C_{i-1}.

    This is the transform used by the body-fixed recursion in the paper.
    """
    return transform_inverse(C_child) @ C_parent


def spatial_force_from_wrench_at_origin(
    force: np.ndarray,
    point_from_origin: np.ndarray,
) -> np.ndarray:
    """
    Wrench [torque; force] at a frame origin caused by a force applied at a
    point whose vector from the frame origin is point_from_origin.
    """
    force = np.asarray(force, dtype=float).reshape(3)
    point_from_origin = np.asarray(point_from_origin, dtype=float).reshape(3)
    return np.concatenate([
        np.cross(point_from_origin, force),
        force,
    ])


# ---------------------------------------------------------------------------
# Robot data
# ---------------------------------------------------------------------------

@dataclass
class Link:
    """
    Link data for the Lie-group formulation.

    Parameters
    ----------
    B:
        Zero-reference relative configuration B_i = C_{i,i-1}(0).
    screw:
        Constant body-fixed joint screw X_i = [e; x^e + h e].
        For a revolute z joint at the frame origin:
            [0,0,1,0,0,0]
        For a prismatic x joint:
            [0,0,0,1,0,0]
    mass:
        Link mass.
    inertia_com:
        3x3 rotational inertia about the COM, expressed in the body frame.
    com:
        Vector from the body-frame origin to the COM, expressed in the
        body-fixed frame.
    damping:
        Optional viscous joint damping. This is an extension and is not part
        of the paper's rigid-link NE equations.
    """

    B: np.ndarray
    screw: np.ndarray
    mass: float
    inertia_com: np.ndarray
    com: np.ndarray
    damping: float = 0.0

    def __post_init__(self):
        self.B = np.asarray(self.B, dtype=float).reshape(4, 4)
        self.screw = np.asarray(self.screw, dtype=float).reshape(6)
        self.inertia_com = np.asarray(self.inertia_com, dtype=float).reshape(3, 3)
        self.com = np.asarray(self.com, dtype=float).reshape(3)
        self.mass = float(self.mass)
        self.damping = float(self.damping)

        if self.mass <= 0:
            raise ValueError("Link mass must be positive.")

        if not np.allclose(self.B[3], [0, 0, 0, 1]):
            raise ValueError("B must be a homogeneous SE(3) transform.")

    @property
    def inertia_origin(self) -> np.ndarray:
        """
        Rotational inertia about the link-frame origin.

        If d is the origin -> COM vector:
            I_O = I_C - m [d]^2
        """
        D = skew(self.com)
        return self.inertia_com - self.mass * (D @ D)

    @property
    def mass_matrix(self) -> np.ndarray:
        """
        Body-fixed 6x6 mass matrix, corresponding to Eq. (13).

        M = [[Theta,  m d^],
             [-m d^, m I]]
        """
        Dm = self.mass * skew(self.com)
        return np.block([
            [self.inertia_origin, Dm],
            [-Dm, self.mass * np.eye(3)],
        ])


@dataclass
class Robot:
    """Serial manipulator represented using Lie-group / screw coordinates."""

    links: Sequence[Link]

    def __post_init__(self):
        self.links = list(self.links)
        if len(self.links) == 0:
            raise ValueError("Robot must contain at least one link.")

    @property
    def n(self) -> int:
        return len(self.links)

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        q: Sequence[float],
        qd: Sequence[float],
        qdd: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Body-fixed forward recursion from the paper.

        Returns
        -------
        C:
            (n,4,4) link configurations.
        V:
            (n,6) body-fixed twists.
        Vdot:
            (n,6) body-fixed accelerations.
        Crel:
            (n,4,4) relative transforms C_{i,i-1}.
        """
        q = self._check_vector(q, "q")
        qd = self._check_vector(qd, "qd")
        qdd = self._check_vector(qdd, "qdd")

        C = np.zeros((self.n, 4, 4))
        Crel = np.zeros((self.n, 4, 4))
        V = np.zeros((self.n, 6))
        Vdot = np.zeros((self.n, 6))

        C_parent = np.eye(4)
        V_parent = np.zeros(6)
        Vdot_parent = np.zeros(6)

        for i, link in enumerate(self.links):
            # C_i = C_{i-1} B_i exp(X_i q_i)
            joint_motion = se3_from_twist(link.screw, q[i])
            C[i] = C_parent @ link.B @ joint_motion

            # C_{i,i-1} = C_i^{-1} C_{i-1}
            Crel[i] = relative_transform(C[i], C_parent)
            Ad_rel = adjoint(Crel[i])

            # Eq. (4)
            V[i] = Ad_rel @ V_parent + link.screw * qd[i]

            # Eq. (5), written in the compact body-fixed form
            Vdot[i] = (
                Ad_rel @ Vdot_parent
                - qd[i] * ad_matrix(link.screw) @ V[i]
                + link.screw * qdd[i]
            )

            C_parent = C[i]
            V_parent = V[i]
            Vdot_parent = Vdot[i]

        return C, V, Vdot, Crel

    # ------------------------------------------------------------------
    # Gravity / external wrench
    # ------------------------------------------------------------------

    def gravity_wrenches(
        self,
        C: np.ndarray,
        gravity: Sequence[float],
    ) -> np.ndarray:
        """
        Convert a constant gravity vector in the inertial frame into the
        body-fixed applied wrench for every link.

        The wrench is applied at the link COM and expressed at the body-frame
        origin:
            W_g = [d x (m g_body); m g_body].
        """
        gravity = np.asarray(gravity, dtype=float).reshape(3)
        Wg = np.zeros((self.n, 6))

        for i, link in enumerate(self.links):
            R = C[i, :3, :3]
            g_body = R.T @ gravity
            force = link.mass * g_body
            Wg[i] = spatial_force_from_wrench_at_origin(force, link.com)

        return Wg

    # ------------------------------------------------------------------
    # Inverse dynamics
    # ------------------------------------------------------------------

    def inverse_dynamics(
        self,
        q: Sequence[float],
        qd: Sequence[float],
        qdd: Sequence[float],
        gravity: Sequence[float] = (0.0, -9.81, 0.0),
        external_wrenches: Optional[np.ndarray] = None,
        include_damping: bool = True,
        return_intermediates: bool = False,
    ):
        """
        Compute joint generalized forces Q for one trajectory sample.

        Parameters
        ----------
        q, qd, qdd:
            Joint positions, velocities and accelerations, each length n.
        gravity:
            Gravity vector resolved in the inertial frame.
        external_wrenches:
            Optional (n,6) body-fixed applied wrenches at each link origin.
            These are added to the gravity wrench.
        include_damping:
            Add b_i*qdot_i if damping was supplied to Link.
        return_intermediates:
            If True, return tau plus the recursive quantities.

        Returns
        -------
        tau:
            (n,) generalized joint forces/torques.
        """
        C, V, Vdot, Crel = self.forward_kinematics(q, qd, qdd)

        M = np.array([link.mass_matrix for link in self.links])
        W = np.zeros((self.n, 6))

        W_app = self.gravity_wrenches(C, gravity)

        if external_wrenches is not None:
            external_wrenches = np.asarray(external_wrenches, dtype=float)
            if external_wrenches.shape != (self.n, 6):
                raise ValueError(
                    f"external_wrenches must have shape {(self.n, 6)}."
                )
            W_app += external_wrenches

        # Backward recursion:
        # W_i = Ad^T W_{i+1}
        #       + M_i Vdot_i
        #       - ad^T_{V_i} M_i V_i
        #       + W_app_i
        for i in reversed(range(self.n)):
            local = (
                M[i] @ Vdot[i]
                - ad_matrix(V[i]).T @ (M[i] @ V[i])
                + W_app[i]
            )

            if i == self.n - 1:
                W[i] = local
            else:
                # W_{i+1} is expressed in frame i+1. Transform it into frame i.
                # C_{i+1,i} is Crel[i+1].
                W[i] = adjoint(Crel[i + 1]).T @ W[i + 1] + local

        tau = np.array([
            self.links[i].screw @ W[i]
            for i in range(self.n)
        ])

        if include_damping:
            tau += np.array([
                self.links[i].damping * float(qd[i])
                for i in range(self.n)
            ])

        if return_intermediates:
            return {
                "tau": tau,
                "C": C,
                "Crel": Crel,
                "V": V,
                "Vdot": Vdot,
                "M": M,
                "W_app": W_app,
                "W": W,
            }

        return tau

    def inverse_dynamics_trajectory(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        gravity: Sequence[float] = (0.0, -9.81, 0.0),
        external_wrenches: Optional[np.ndarray] = None,
        include_damping: bool = True,
    ) -> np.ndarray:
        """
        Compute torques for any number of trajectory samples.

        Input shapes:
            q, qd, qdd = (N,n)

        Output:
            tau = (N,n)
        """
        q = self._check_trajectory(q, "q")
        qd = self._check_trajectory(qd, "qd")
        qdd = self._check_trajectory(qdd, "qdd")

        if not (q.shape == qd.shape == qdd.shape):
            raise ValueError("q, qd and qdd must have identical shapes.")

        if external_wrenches is not None:
            external_wrenches = np.asarray(external_wrenches, dtype=float)
            if external_wrenches.shape != (q.shape[0], self.n, 6):
                raise ValueError(
                    "external_wrenches must have shape (N,n,6)."
                )

        tau = np.zeros_like(q)

        for k in range(q.shape[0]):
            Wext = None if external_wrenches is None else external_wrenches[k]
            tau[k] = self.inverse_dynamics(
                q[k],
                qd[k],
                qdd[k],
                gravity=gravity,
                external_wrenches=Wext,
                include_damping=include_damping,
            )

        return tau

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def load_trajectory_csv(
        self,
        filename: str | Path,
        skiprows: int = 1,
        layout: str = "columns",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load q, qd, qdd from CSV.

        layout="columns":
            each row is one time sample:
            q1...qn, qd1...qdn, qdd1...qddn

        layout="rows":
            three rows (or equivalent transposed format):
            q
            qd
            qdd
        """
        data = np.loadtxt(filename, delimiter=",", skiprows=skiprows)
        data = np.atleast_2d(data)

        if layout == "columns":
            expected = 3 * self.n
            if data.shape[1] != expected:
                raise ValueError(
                    f"Expected {expected} columns for n={self.n}, "
                    f"got {data.shape[1]}."
                )
            q = data[:, :self.n]
            qd = data[:, self.n:2*self.n]
            qdd = data[:, 2*self.n:3*self.n]
            return q, qd, qdd

        if layout == "rows":
            if data.shape[0] != 3 or data.shape[1] != self.n:
                raise ValueError(
                    f"Expected a 3 x {self.n} array for row layout."
                )
            return data[0], data[1], data[2]

        raise ValueError("layout must be 'columns' or 'rows'.")

    @staticmethod
    def save_torques_csv(out_dir: str | Path, tau: np.ndarray) -> None:
        cols = tau.shape[1] if (hasattr(tau, "ndim") and tau.ndim > 1) else 1
        data = {f't{i+1}': (tau[:, i] if cols > 1 else tau[:]) for i in range(cols)}

        df = pd.DataFrame(data)

        torque_data = f"{out_dir}"
        # torque_data = f"{out_dir}/torquesFst{n}.csv"
        df.to_csv(torque_data, index=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_mass_matrices(self) -> dict:
        """Basic physical/numerical checks on the link mass matrices."""
        results = {}
        for i, link in enumerate(self.links):
            M = link.mass_matrix
            symmetry_error = np.linalg.norm(M - M.T)
            eigvals = np.linalg.eigvalsh(0.5 * (M + M.T))
            results[i + 1] = {
                "symmetry_error": symmetry_error,
                "minimum_eigenvalue": float(np.min(eigvals)),
                "positive_definite": bool(np.min(eigvals) > 0.0),
            }
        return results

    def _check_vector(self, x, name: str) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n:
            raise ValueError(
                f"{name} must contain {self.n} values; got {x.size}."
            )
        return x

    def _check_trajectory(self, x, name: str) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n:
            raise ValueError(
                f"{name} must have shape (N,{self.n}); got {x.shape}."
            )
        return x


# ---------------------------------------------------------------------------
# Convenient robot generators
# ---------------------------------------------------------------------------

def make_planar_robot(
    n: int,
    lengths: float | Sequence[float] = 1.0,
    masses: float | Sequence[float] = 1.0,
    damping: float | Sequence[float] = 0.0,
    com_fraction: float = 0.5,
) -> Robot:
    """
    Create an n-link planar revolute manipulator.

    Frames are at the joint axes. At q=0 the next frame is translated along
    +x by the link length, matching the simple planar serial-chain geometry.

    Every joint axis is +z:
        X_i = [0,0,1,0,0,0]

    Each link is a slender rod:
        I_COM,zz = m L^2 / 12

    This is an especially convenient test case for comparing against your
    planar Newton-Euler / Lagrange-Euler implementations.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")

    def expand(value, name):
        if np.isscalar(value):
            return np.full(n, float(value))
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size != n:
            raise ValueError(f"{name} must have length {n}.")
        return value

    lengths = expand(lengths, "lengths")
    masses = expand(masses, "masses")
    damping = expand(damping, "damping")

    if not (0.0 <= com_fraction <= 1.0):
        raise ValueError("com_fraction must be between 0 and 1.")

    links = []

    for i in range(n):
        L = lengths[i]
        m = masses[i]

        B = make_transform(np.eye(3), np.array([L, 0.0, 0.0]))
        X = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        d = np.array([com_fraction * L, 0.0, 0.0])

        I_com = np.diag([
            m * L**2 / 12.0,
            m * L**2 / 12.0,
            m * L**2 / 12.0,
        ])

        links.append(
            Link(
                B=B,
                screw=X,
                mass=m,
                inertia_com=I_com,
                com=d,
                damping=damping[i],
            )
        )

    return Robot(links)


# ---------------------------------------------------------------------------
# Compatibility-style helper resembling your current code
# ---------------------------------------------------------------------------

def link_data(
    n: int,
    length: float = 1.0,
    mass: float = 1.0,
    damping: float = 0.0,
) -> list[Link]:
    """Return link objects in the same spirit as your existing link_data(n)."""
    return make_planar_robot(
        n=n,
        lengths=length,
        masses=mass,
        damping=damping,
    ).links


def lie_group_inverse_dynamics(
    q: Sequence[float],
    qd: Sequence[float],
    qdd: Sequence[float],
    links: Sequence[Link],
    gravity: Sequence[float] = (0.0, -9.81, 0.0),
) -> np.ndarray:
    """Drop-in style function: q, qd, qdd, links -> tau."""
    return Robot(links).inverse_dynamics(q, qd, qdd, gravity=gravity)


def generate_torques(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    links: Sequence[Link],
    gravity: Sequence[float] = (0.0, -9.81, 0.0),
) -> np.ndarray:
    """Compute torque for every row of q/qd/qdd."""
    return Robot(links).inverse_dynamics_trajectory(
        q, qd, qdd, gravity=gravity
    )


# ---------------------------------------------------------------------------
# Example execution
# ---------------------------------------------------------------------------

def example():
    """
    Example corresponding to the execution pattern of your existing codes.
    Generates an arbitrary n-link trajectory, calculates all torques, and
    writes the result to a CSV.
    """
    n = 5
    N = 500

    robot = make_planar_robot(
        n=n,
        lengths=1.0,
        masses=1.0,
        damping=50.0,
    )

    time_step = np.linspace(0.0, 5.0, N)

    # Arbitrary smooth trajectory; replace these with your CSV data.
    out_dir = './ra'

    trj_data = f"{out_dir}/data5s/trajectory_data_gen{n}.csv"

    q, qd, qdd = load_joint_data(trj_data, n, len(time_step), 'csv')


    tau = robot.inverse_dynamics_trajectory(
        q, qd, qdd,
        gravity=(0.0, 9.81, 0.0),
    )

    return robot, time_step, q, qd, qdd, tau


if __name__ == "__main__":
    robot, time_step, q, qd, qdd, tau = example()

    print("Lie-group inverse dynamics")
    print("--------------------------")
    print(f"Number of links: {robot.n}")
    print(f"Number of trajectory points: {len(time_step)}")
    print(f"q shape:    {q.shape}")
    print(f"qd shape:   {qd.shape}")
    print(f"qdd shape:  {qdd.shape}")
    print(f"tau shape:  {tau.shape}")
    print("\nFirst torque sample:")
    print(tau[0])

    print("\nMass-matrix validation:")
    for i, result in robot.validate_mass_matrices().items():
        print(
            f"Link {i}: "
            f"symmetry error={result['symmetry_error']:.3e}, "
            f"min eigenvalue={result['minimum_eigenvalue']:.3e}, "
            f"PD={result['positive_definite']}"
        )

    out_dir = f'./ra/data5s/torquesLie_Group{robot.n}.csv'
    print(os.getcwd())  # Print the current working directory

    # output = f"{out_dir}/data5s/Lie_group{robot.n}.csv"
    robot.save_torques_csv(out_dir, tau)
    print(f"\nSaved torque data to: {out_dir}")
