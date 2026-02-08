import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm


# =====================================================================
#  QP Solver
# =====================================================================

def _solve_robust(H_sub: np.ndarray, y_sub: np.ndarray) -> np.ndarray:
    """Robustly solve H_sub @ x = y_sub."""
    try:
        return np.linalg.solve(H_sub, y_sub)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(H_sub, y_sub, rcond=None)
        return sol


def _check_dual_feasibility(
        H: np.ndarray,
        lambdas_new: np.ndarray,
        b: np.ndarray,
        nu_val: float,
        active: np.ndarray,
        tol: float,
) -> bool:
    """
    Check γ_i ≥ 0 for all inactive variables.
    If violated, add most-violating back to active set (in-place).
    Returns True if active set was updated.
    """
    g = H @ lambdas_new - b + nu_val
    inactive = ~active
    if np.any(inactive):
        g_inact = g[inactive]
        if np.min(g_inact) < -tol:
            inact_indices = np.where(inactive)[0]
            viol_local = np.argmin(g_inact)
            active[inact_indices[viol_local]] = True
            return True
    return False


def solve_qp_active_set(
        G: np.ndarray, b: np.ndarray, eps_reg: float = 1e-12, tol: float = 1e-12
) -> np.ndarray:
    """
    YOYO Scale: min 0.5 * λ^T H λ - b^T λ
                s.t. λ >= 0, sum(λ) <= 1
    """
    N = len(b)
    H = G + eps_reg * np.eye(N, dtype=np.float64)
    lambdas = np.zeros(N, dtype=np.float64)
    active = np.ones(N, dtype=bool)
    max_iter = 10 * N + 50

    for _ in range(max_iter):
        idx = np.where(active)[0]
        if idx.size == 0:
            lambdas[:] = 0.0
            break

        Hs = H[np.ix_(idx, idx)]
        bs = b[idx]
        ones_s = np.ones(idx.size, dtype=np.float64)

        v = _solve_robust(Hs, bs)
        sum_v = v.sum()
        nu_val = 0.0

        if sum_v > 1.0 + tol:
            u = _solve_robust(Hs, ones_s)
            denom = ones_s @ u
            if abs(denom) > 1e-30:
                nu_tmp = (sum_v - 1.0) / denom
                if nu_tmp > 0.0:
                    nu_val = nu_tmp
                    v = v - nu_val * u

        lambdas_new = np.zeros_like(lambdas)
        lambdas_new[idx] = v

        neg_mask = lambdas_new[idx] <= tol
        if np.any(neg_mask):
            active[idx[neg_mask]] = False
            continue

        if _check_dual_feasibility(H, lambdas_new, b, nu_val, active, tol):
            continue

        lambdas = lambdas_new
        break

    lambdas = np.clip(lambdas, 0.0, None)
    total = lambdas.sum()
    if total > 1.0:
        lambdas /= total

    return lambdas


# =====================================================================
#  Simulation
# =====================================================================

def simulate_single_run(D=30000, N=5, mu_dist=5.0, avg_noise_radius=12.0, noise_std=6.0):
    """Run one simulation and return distances for the 3 remaining methods."""
    w0 = np.zeros(D, dtype=np.float64)
    mu = np.zeros(D, dtype=np.float64)
    mu[0] = mu_dist

    model_ts = []
    actual_radii = []

    for _ in range(N):
        r_i = max(0.1, np.random.normal(avg_noise_radius, noise_std))
        actual_radii.append(r_i)

        eps = np.random.randn(D).astype(np.float64)
        eps[0] = 0.0
        eps_norm = np.linalg.norm(eps)
        if eps_norm > 0:
            eps = (eps / eps_norm) * r_i
        else:
            eps = np.zeros_like(eps)

        wi = mu + eps
        model_ts.append(wi)

    best_single_dist = min(actual_radii)

    vs = np.stack([m - w0 for m in model_ts])  # (N, D)
    w_avg = np.mean(model_ts, axis=0)
    G = vs @ vs.T  # (N, N)

    # -----------------------------------------------------------------
    # 1) Simple Average
    # -----------------------------------------------------------------
    dist_avg = np.linalg.norm(w_avg - mu)

    # -----------------------------------------------------------------
    # 2) Model Stock
    # -----------------------------------------------------------------
    cos_list = []
    for i in range(N):
        for j in range(i + 1, N):
            ni, nj = np.linalg.norm(vs[i]), np.linalg.norm(vs[j])
            cos_ij = np.dot(vs[i], vs[j]) / (ni * nj) if ni > 0 and nj > 0 else 0.0
            cos_list.append(cos_ij)
    avg_cos = np.mean(cos_list) if cos_list else 0.0
    denom_ms = 1.0 + (N - 1) * avg_cos
    t_ms = (N * avg_cos) / denom_ms if denom_ms != 0 else 0.0
    w_ms = w0 + t_ms * (w_avg - w0)
    dist_ms = np.linalg.norm(w_ms - mu)

    # -----------------------------------------------------------------
    # 3) YOYO Scale
    # -----------------------------------------------------------------
    if N <= 1:
        coeffs_yoyo = [1.0] if N == 1 else []
        w_yoyo = model_ts[0].copy() if N == 1 else w0.copy()
    else:
        b_qp = (G.sum(axis=1) - np.diag(G)) / (N - 1)
        lambdas_qp = solve_qp_active_set(G, b_qp)
        coeffs_yoyo = lambdas_qp.tolist()
        w_yoyo = w0 + vs.T @ lambdas_qp

    dist_yoyo = np.linalg.norm(w_yoyo - mu)

    # -----------------------------------------------------------------
    # Collect results
    # -----------------------------------------------------------------
    distances = {
        "Simple Average": dist_avg,
        "Model Stock": dist_ms,
        "YOYO Scale": dist_yoyo,
    }

    debug = {
        "actual_radii": actual_radii,
        "t_ms": t_ms,
        "coeffs_yoyo": coeffs_yoyo,
    }

    return distances, best_single_dist, debug


def simulate_asymmetric_merging(num_runs=100):
    D = 10000000
    N = 10
    mu_dist = 5.0
    avg_noise_radius = 12.0
    noise_std = 6.0

    method_names = [
        "Simple Average",
        "Model Stock",
        "YOYO Scale",
    ]

    print(f"=== Simulation Configuration ===")
    print(f"Dimensionality: {D},  Number of Models: {N}")
    print(f"True center distance ||μ − w₀||: {mu_dist}")
    print(f"Average noise radius: {avg_noise_radius},  Radius std: {noise_std}")
    print(f"Number of simulation runs: {num_runs}")
    print()

    win_counts: Dict[str, int] = {m: 0 for m in method_names}
    dist_accum: Dict[str, float] = {m: 0.0 for m in method_names}
    best_accum: float = 0.0
    last_run = None

    for _ in tqdm(range(num_runs), desc="Simulating", unit="run", ncols=80):
        distances, best_single_dist, debug = simulate_single_run(
            D, N, mu_dist, avg_noise_radius, noise_std
        )
        last_run = (distances, best_single_dist, debug)

        best_accum += best_single_dist
        for m in method_names:
            dist_accum[m] += distances[m]
            if distances[m] < best_single_dist:
                win_counts[m] += 1

    # ---- Last-run details ----
    distances, best_single_dist, debug = last_run

    print(f"\n{'=' * 120}")
    print(f"  LAST RUN DETAILS")
    print(f"{'=' * 120}")
    print(f"Actual noise radii: {', '.join(f'{r:.2f}' for r in debug['actual_radii'])}")
    print(f"{'-' * 120}")
    print(f"{'Method':<40} | {'Dist to μ':>12} | Details")
    print(f"{'-' * 120}")
    print(f"{'Best Single Model':<40} | {best_single_dist:>12.6f} | (min radius)")
    print(f"{'Simple Average':<40} | {distances['Simple Average']:>12.6f} | uniform 1/N")
    print(f"{'Model Stock':<40} | {distances['Model Stock']:>12.6f} | t = {debug['t_ms']:.4f}")
    print(
        f"{'YOYO Scale':<40} | {distances['YOYO Scale']:>12.6f} | λ = {[round(c, 4) for c in debug['coeffs_yoyo']]}")
    print(f"{'-' * 120}")

    best_method_last = min(distances, key=distances.get)
    print(f"✨ Best method (last run): {best_method_last}  (distance = {distances[best_method_last]:.6f})\n")

    # ---- Aggregate statistics ----
    print(f"{'=' * 120}")
    print(f"  AGGREGATE RESULTS  ({num_runs} runs)")
    print(f"{'=' * 120}")
    print(f"{'Method':<40} | {'Win Prob':>10} | {'Avg Dist':>12} | {'Wins':>6} / {num_runs}")
    print(f"{'-' * 120}")
    print(f"{'Best Single Model (baseline)':<40} | {'—':>10} | {best_accum / num_runs:>12.6f} |")

    for m in method_names:
        prob = win_counts[m] / num_runs
        avg_d = dist_accum[m] / num_runs
        print(f"{m:<40} | {prob:>9.2%} | {avg_d:>12.6f} | {win_counts[m]:>6} / {num_runs}")

    print(f"{'-' * 120}")
    best_overall = max(win_counts, key=win_counts.get)
    print(f"🏆 Overall best performer: {best_overall}  ({win_counts[best_overall] / num_runs:.2%} win rate)")

    lowest_avg = min(method_names, key=lambda m: dist_accum[m])
    print(f"📏 Lowest average distance: {lowest_avg}  ({dist_accum[lowest_avg] / num_runs:.6f})")


if __name__ == "__main__":
    np.random.seed(42)
    simulate_asymmetric_merging(num_runs=1000)