"""[isaac v6] numpy forward / inverse kinematics of ManiSkill's ``panda_stick`` URDF (robot root frame).

Used for the reset-time random TCP init (``tcp_init_random``): the joint solution is computed here (kinematics only,
no PhysX state involved) and written to the articulation once, like LIBERO's ``teleport_pen_tip`` and ManiSkill's
``_maybe_randomize_tcp_init``. Independent of Isaac, so it can be tested on the CPU.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parent / "assets" / "maniskill" / "panda_stick.urdf"
# Franka Emika Panda datasheet joint velocity limits (rad/s), same table as burger.backends.libero.environment
FRANKA_JOINT_VEL_LIMIT = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])


def _origin(xyz, rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    T = np.eye(4)
    T[:3, :3] = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    T[:3, 3] = xyz
    return T


def _rot_axis(axis, a) -> np.ndarray:
    k = np.asarray(axis, dtype=np.float64); k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    return T


def _axis_angle(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-9:
        return 0.5 * v
    if math.pi - ang < 1e-6:   # near pi: axis from the symmetric part
        k = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0)); i = int(np.argmax(k))
        k = k * np.sign(np.where(np.arange(3) == i, 1.0, (R[i] + R[:, i])[np.arange(3)]))
        return ang * k / np.linalg.norm(k)
    return ang / (2 * math.sin(ang)) * v


class PandaStickKinematics:
    """Serial chain panda_link0 -> panda_hand_tcp parsed from the URDF; joint order panda_joint1..7."""

    def __init__(self, urdf: Path = URDF, tip_link: str = "panda_hand_tcp"):
        root = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], tip_link
        while link in by_child:
            j = by_child[link]; chain.append(j); link = j.find("parent").get("link")
        self.joints = []   # (origin T, axis or None, lower, upper)
        for j in reversed(chain):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
            if j.get("type") == "revolute":
                lim = j.find("limit")
                self.joints.append((_origin(xyz, rpy), [float(v) for v in j.find("axis").get("xyz").split()], float(lim.get("lower")), float(lim.get("upper"))))
            else:
                self.joints.append((_origin(xyz, rpy), None, 0.0, 0.0))
        self.lower = np.array([lo for _, ax, lo, _ in self.joints if ax is not None])
        self.upper = np.array([up for _, ax, _, up in self.joints if ax is not None])
        assert len(self.lower) == 7, "panda_stick chain must have 7 revolute joints"

    def fk(self, q):
        """TCP pose in the root frame -> (pos (3,), R (3,3), joint axes (7,3), joint origins (7,3))."""
        T = np.eye(4); axes, origins = [], []; k = 0
        for To, ax, _, _ in self.joints:
            T = T @ To
            if ax is not None:
                axes.append(T[:3, :3] @ (np.asarray(ax) / np.linalg.norm(ax))); origins.append(T[:3, 3].copy())
                T = T @ _rot_axis(ax, float(q[k])); k += 1
        return T[:3, 3].copy(), T[:3, :3].copy(), np.array(axes), np.array(origins)

    def jacobian(self, q):
        p, R, axes, origins = self.fk(q)
        return np.vstack([np.cross(axes, p - origins).T, axes.T]), p, R

    def ik(self, pos_target, R_target, q0, iters: int = 100, lam: float = 1e-4, margin: float = 0.0, tol: float = 1e-4):
        """Iterated Levenberg-Marquardt (ManiSkill's step, repeated) to an absolute TCP pose. Returns (q, pos_err_m, rot_err_rad)."""
        q = np.array(q0, dtype=np.float64)
        lo, hi = self.lower + margin, self.upper - margin
        for _ in range(int(iters)):
            J, p, R = self.jacobian(q)
            e = np.concatenate([np.asarray(pos_target) - p, _axis_angle(R_target @ R.T)])
            if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 10 * tol:
                break
            q = np.clip(q + np.linalg.solve(J.T @ J + lam * np.eye(7), J.T @ e), lo, hi)
        p, R, _, _ = self.fk(q)
        return q, float(np.linalg.norm(np.asarray(pos_target) - p)), float(np.linalg.norm(_axis_angle(R_target @ R.T)))


def rot_x_pi() -> np.ndarray:
    """pd_ee_pose orientation target of every action (euler xyz (pi, 0, 0)): stick pointing down."""
    return np.diag([1.0, -1.0, -1.0])
