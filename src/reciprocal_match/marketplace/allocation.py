from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class AllocationResult:
    selected_pairs: pd.DataFrame
    user_exposure: pd.DataFrame
    target_pairs: int


def gini(values: np.ndarray | pd.Series) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan")
    if (x < 0).any():
        raise ValueError("Gini requires nonnegative values")
    total = x.sum()
    if total == 0:
        return 0.0
    diff = np.abs(x[:, None] - x[None, :]).sum()
    return float(diff / (2.0 * len(x) * total))


def prepare_marketplace_pairs(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"wave", "iid", "pid", "y_match", "p_match_direct"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing marketplace columns: {missing}")

    frame = predictions.copy()
    frame["u"] = frame[["iid", "pid"]].min(axis=1).astype(int)
    frame["v"] = frame[["iid", "pid"]].max(axis=1).astype(int)
    rows = []
    for (wave, u, v), group in frame.groupby(["wave", "u", "v"], sort=True):
        if len(group) != 2:
            raise ValueError(f"Expected two directed rows for pair {(wave, u, v)}")
        y = group["y_match"].dropna().unique()
        p = group["p_match_direct"].to_numpy(dtype=float)
        if len(y) != 1:
            raise ValueError(f"Reverse rows disagree on y_match for pair {(wave, u, v)}")
        if not np.allclose(p, p[0], atol=1e-12, rtol=0.0):
            raise ValueError(f"Reverse rows disagree on p_match_direct for pair {(wave, u, v)}")
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("p_match_direct must contain finite probabilities in [0, 1]")
        rows.append({
            "wave": int(wave), "u": int(u), "v": int(v),
            "y_match": int(y[0]), "score": float(p[0]),
        })
    return pd.DataFrame(rows).sort_values(["wave", "u", "v"]).reset_index(drop=True)


def target_pair_count(n_users: int, capacity: int, fraction: float, n_edges: int) -> int:
    if capacity < 1:
        raise ValueError("capacity must be >= 1")
    if not (0 < fraction <= 1):
        raise ValueError("target capacity fraction must be in (0, 1]")
    capacity_edges = (n_users * capacity) // 2
    target = int(np.floor(capacity_edges * fraction))
    return max(1, min(target, n_edges))


def expected_random_exposure(pairs: pd.DataFrame, target_pairs: int) -> pd.DataFrame:
    users = sorted(set(pairs["u"]) | set(pairs["v"]))
    degree = pd.Series(0, index=users, dtype=int)
    for r in pairs.itertuples(index=False):
        degree.loc[r.u] += 1
        degree.loc[r.v] += 1
    q = target_pairs * degree.astype(float) / float(len(pairs))
    out = pd.DataFrame({"user": degree.index.astype(int), "eligible_degree": degree.values, "q_random": q.values})
    if not np.isclose(out["q_random"].sum(), 2.0 * target_pairs):
        raise AssertionError("Expected random exposure must sum to twice the pair budget")
    return out


def solve_exposure_controlled_allocation(
    pairs: pd.DataFrame,
    capacity: int,
    target_pairs: int,
    lambda_value: float,
) -> AllocationResult:
    if pairs["wave"].nunique() != 1:
        raise ValueError("Allocator expects exactly one wave")
    if lambda_value < 0:
        raise ValueError("lambda must be nonnegative")

    users = sorted(set(pairs["u"]) | set(pairs["v"]))
    user_idx = {u: i for i, u in enumerate(users)}
    n_edges = len(pairs)
    n_slots = len(users) * capacity
    n_vars = n_edges + n_slots

    opportunity = expected_random_exposure(pairs, target_pairs).set_index("user")
    c = np.zeros(n_vars, dtype=float)
    c[:n_edges] = -pairs["score"].to_numpy(dtype=float)

    # Linearized convex penalty on squared relative exposure:
    # (E_u / Q_u)^2 has marginal increment (2j - 1) / Q_u^2 for slot j.
    for user_pos, user in enumerate(users):
        q = float(opportunity.loc[user, "q_random"])
        if q <= 0:
            raise ValueError("All eligible users must have positive random exposure expectation")
        for j in range(1, capacity + 1):
            slot = n_edges + user_pos * capacity + (j - 1)
            c[slot] = lambda_value * ((2 * j - 1) / (q * q))

    # One equality fixes the total selected pairs. One equality per user links
    # selected incident edges to that user's occupied exposure slots.
    n_constraints = 1 + len(users)
    A = lil_matrix((n_constraints, n_vars), dtype=float)
    lower = np.zeros(n_constraints, dtype=float)
    upper = np.zeros(n_constraints, dtype=float)
    A[0, :n_edges] = 1.0
    lower[0] = upper[0] = target_pairs

    for user_pos, user in enumerate(users):
        row = 1 + user_pos
        incident = np.flatnonzero((pairs["u"].to_numpy() == user) | (pairs["v"].to_numpy() == user))
        A[row, incident] = 1.0
        first_slot = n_edges + user_pos * capacity
        A[row, first_slot:first_slot + capacity] = -1.0
        lower[row] = upper[row] = 0.0

    result = milp(
        c=c,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(A.tocsr(), lower, upper),
        options={"time_limit": 30.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP allocation failed: {result.message}")

    selected_mask = result.x[:n_edges] > 0.5
    selected = pairs.loc[selected_mask].copy().reset_index(drop=True)
    if len(selected) != target_pairs:
        raise AssertionError("MILP returned unexpected number of selected pairs")

    exposure = opportunity.reset_index().copy()
    realized = pd.Series(0, index=users, dtype=int)
    for r in selected.itertuples(index=False):
        realized.loc[r.u] += 1
        realized.loc[r.v] += 1
    exposure["exposure"] = exposure["user"].map(realized).astype(int)
    exposure["relative_exposure"] = exposure["exposure"] / exposure["q_random"]
    exposure["lambda"] = float(lambda_value)
    return AllocationResult(selected, exposure, target_pairs)


def random_greedy_allocation(
    pairs: pd.DataFrame,
    capacity: int,
    target_pairs: int,
    seed: int,
) -> AllocationResult:
    rng = np.random.default_rng(seed)
    users = sorted(set(pairs["u"]) | set(pairs["v"]))
    for _ in range(100):
        order = rng.permutation(len(pairs))
        degree = {u: 0 for u in users}
        chosen = []
        for idx in order:
            r = pairs.iloc[int(idx)]
            if degree[int(r.u)] >= capacity or degree[int(r.v)] >= capacity:
                continue
            chosen.append(int(idx))
            degree[int(r.u)] += 1
            degree[int(r.v)] += 1
            if len(chosen) == target_pairs:
                break
        if len(chosen) == target_pairs:
            selected = pairs.iloc[chosen].copy().reset_index(drop=True)
            opportunity = expected_random_exposure(pairs, target_pairs)
            opportunity["exposure"] = opportunity["user"].map(degree).astype(int)
            opportunity["relative_exposure"] = opportunity["exposure"] / opportunity["q_random"]
            opportunity["lambda"] = np.nan
            return AllocationResult(selected, opportunity, target_pairs)
    raise RuntimeError("Random greedy allocator could not fill the requested pair budget")
