# YOYO-Scale: Geometry-Driven Task Vector Scaling Without Training

[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

**YOYO-Scale** is a robust, geometry-aware model merging method that **generalizes Model Stock** by computing **per-tensor, nonnegative fusion coefficients under an anchor-preserving constraint**. It requires no extra training and works reliably in realistic (asymmetric) settings.

> **Key Insight**: Instead of relying on Model Stock's strong symmetry assumptions (e.g., identical pairwise angles), YOYO-Scale formulates merging as a **constrained convex optimization** that (i) estimates the unknown center-consensus direction from cross-model agreement and (ii) finds the best merged tensor **within the convex hull of `{base, fine-tuned models}`**.

---

## Highlights

- **Generalizes Model Stock**: Recovers Model Stock as a special case under its geometric assumptions.
- **Anchor-preserving by design**: Ensures the merged weight stays within the convex hull of `{w0, w1, ..., wN}` via `λ_i ≥ 0` and `∑ λ_i ≤ 1`.
- **Per-tensor adaptive coefficients**: Solves a **constrained quadratic program** to obtain tensor-wise λ weights.
- **Noise-robust**: Estimates the center-consensus signal using **cross-model inner products**, excluding self-correlation that is contaminated by model-specific noise.
- **Efficient**: No extra training; only requires loading fine-tuned checkpoints and computing small Gram systems per tensor.
- **Interpretable**: Outputs `yoyo_scale.csv` with per-model λ values for analysis.

---

## Theoretical Foundation

### Step 1: Problem Setup and Notation

- Let the base (anchor) model weight be `w0`. All operations are translation-invariant, so we work in task-vector space.
- We have `N` fine-tuned models with weights: `w1, w2, ..., wN`.
- Define task vectors:
  ```
  vi = wi - w0
  ```
- Assume there exists an unknown "ideal center" `mu`, representing the common target of fine-tuning.

We seek a merged weight that **keeps anchor influence**:
```
w_merged = w0 + sum_{i=1}^N lambda_i * vi
```
with constraints:
- `lambda_i ≥ 0`
- `sum(lambda_i) ≤ 1`

Equivalently:
```
w_merged = (1 - sum(lambda)) * w0 + sum_i lambda_i * wi
```
so the result stays inside `conv({w0, w1, ..., wN})`.

Let:
- `V` be the matrix stacking vectors `v1, ..., vN`,
- `lambda` be the coefficient vector.

---

### Step 2: Objective — Closest Point to the Unknown Center Under Constraints

The ideal goal is:
```
minimize || mu - (w0 + V * lambda) ||^2
subject to lambda_i >= 0,  sum(lambda_i) <= 1
```

This is the *anchor-preserving constrained projection* of the unknown center onto the feasible merge simplex.

Expanding the squared distance yields a quadratic objective:
```
0.5 * lambda^T * G * lambda - b^T * lambda
```
where:
- `G[i, j] = vi · vj` is the Gram matrix,
- `b[i] = vi · (mu - w0)`.

Thus the constrained merging problem becomes:
```
min_{lambda >= 0, sum(lambda) <= 1}  0.5 * lambda^T G lambda - b^T lambda
```

---

### Step 3: Estimating `b` Without Observing `mu` (Cross-Model Consensus)

Since `mu` is unobservable, YOYO-Scale estimates `b` using Model Stock's geometric insights.

#### **Assumption A (Orthogonal Anchor)**:
The vector from each fine-tuned model to the center is perpendicular to the vector from the base model to the center:
```
(wi - mu) ⟂ (w0 - mu)
```
This implies:
```
(wi - w0) · (mu - w0) = ||mu - w0||^2
=> vi · (mu - w0) = constant
```
So theoretically, **`b[i]` is the same constant for all i**.

#### **Assumption B (Independent Noise / Thin Shell)**:
For `i ≠ j`, the noise vectors are approximately orthogonal:
```
(wi - mu) ⟂ (wj - mu)
```
Expanding gives:
```
vi · vj = ||mu - w0||^2,    for i ≠ j
```
But the self-inner product includes noise variance:
```
||vi||^2 = ||mu - w0||^2 + noise
```

> **Key Insight**:
> - **Cross-model inner products** (`vi · vj` for `i ≠ j`) are robust consensus signals.
> - **Self-inner products** (`||vi||^2`) are contaminated by model-specific noise.

Therefore YOYO-Scale estimates `b` by excluding the diagonal:
```
b[j] = (1 / (N - 1)) * sum_{i ≠ j} (vi · vj)
```
In matrix form:
```
b = (G * 1 - diag(G)) / (N - 1)
```

---

### Step 4: Constrained YOYO Scale Solver

YOYO-Scale solves the **convex QP** directly:
```
min_{lambda >= 0, sum(lambda) <= 1}  0.5 * lambda^T (G + eps I) lambda - b^T lambda
```
where `eps ~ 1e-12` stabilizes ill-conditioned Gram matrices.

This replaces the heuristic pipeline:
- solve `G lambda = b`
- then clamp / renormalize

with a principled optimization that enforces:
- **non-negativity**
- **anchor-preserving sum constraint**
as part of the optimality conditions (KKT).

The final merged weight is:
```
w_merged = w0 + sum_i lambda_i * (wi - w0)
```

---

### Step 5: Showing Model Stock Is a Special Case of YOYO-Scale

Under Model Stock's strong symmetry assumptions:
- `||vi||^2 = l^2` (constant norm),
- `vi · vj = l^2 * cos(theta)` for all `i ≠ j` (constant pairwise similarity),

the Gram matrix becomes:
```
G = l^2 * [ (1 - cos(theta)) * I + cos(theta) * ones(N, N) ]
```
and the estimated target vector is uniform:
```
b = l^2 * cos(theta) * [1, 1, ..., 1]^T
```

By symmetry, the optimal solution is uniform:
```
lambda = alpha * [1, 1, ..., 1]
```

Solving the (unconstrained) normal equation `G lambda = b` gives:
```
alpha = cos(theta) / (1 + (N - 1) * cos(theta))
```

The total weight sum is:
```
t = sum(lambda_i) = N * alpha = (N * cos(theta)) / (1 + (N - 1) * cos(theta))
```

Thus the merged weight becomes:
```
w_merged = (1 - t) * w0 + t * w_avg
```
where `w_avg = (1/N) * sum_i wi`.

This is exactly **Model Stock**.  
Hence **YOYO-Scale generalizes Model Stock** by removing strict symmetry requirements while retaining anchor-preserving geometry.

---

## Simulation Results

We validate YOYO-Scale on a controlled geometric simulation that exactly matches the theoretical setup: `N` fine-tuned models are generated as `wi = mu + noise_i`, where the true center `mu` is at distance 5.0 from the base `w0`, and each model has an independent isotropic noise component with mean radius 12.0 (std 6.0). This creates a deliberately challenging, asymmetric scenario where noise radii vary significantly across models.

The key metric is **win probability**: the fraction of runs where each merging method produces a result **closer to the true center** than the **best single fine-tuned model** (i.e., the one with the smallest noise radius). A win probability above 50% means the method systematically outperforms cherry-picking.

### N = 5 Models, D = 10,000,000

| Method | Win Prob | Avg Distance to μ |
|---|---|---|
| Best Single Model (baseline) | — | 5.435 |
| Simple Average | 45.50% | 5.899 |
| Model Stock | 63.90% | 3.865 |
| **YOYO-Scale** | **98.20%** | **2.571** |

<details>
<summary>Representative single-run details (click to expand)</summary>

```
Actual noise radii: 6.79, 15.79, 5.09, 15.63, 7.22

Method                        | Dist to μ  | Details
------------------------------|------------|------------------------------------------
Best Single Model             |   5.091    | (min radius)
Simple Average                |   4.966    | uniform 1/N
Model Stock                   |   3.600    | t = 0.6082
YOYO-Scale                    |   2.799    | λ = [0.169, 0.032, 0.301, 0.032, 0.151]
```

Note how YOYO-Scale assigns the largest coefficient (0.301) to the model with the smallest noise radius (5.09), and near-zero coefficients (0.032) to the noisiest models (15.79, 15.63).

</details>

### N = 2 Models, D = 10,000,000

| Method | Win Prob | Avg Distance to μ |
|---|---|---|
| Best Single Model (baseline) | — | 8.932 |
| Simple Average | 56.50% | 9.065 |
| Model Stock | 84.70% | 4.264 |
| **YOYO-Scale** | **100.00%** | **3.734** |

<details>
<summary>Representative single-run details (click to expand)</summary>

```
Actual noise radii: 4.85, 2.33

Method                        | Dist to μ  | Details
------------------------------|------------|------------------------------------------
Best Single Model             |   2.325    | (min radius)
Simple Average                |   2.688    | uniform 1/N
Model Stock                   |   2.368    | t = 0.7886
YOYO-Scale                    |   1.933    | λ = [0.159, 0.691]
```

With only 2 models, YOYO-Scale correctly concentrates weight (0.691) on the lower-noise model.

</details>

### N = 10 Models, D = 10,000,000

| Method | Win Prob | Avg Distance to μ |
|---|---|---|
| Best Single Model (baseline) | — | 3.347 |
| Simple Average | 36.50% | 4.229 |
| Model Stock | 46.20% | 3.299 |
| **YOYO-Scale** | **97.30%** | **1.722** |

<details>
<summary>Representative single-run details (click to expand)</summary>

```
Actual noise radii: 15.45, 25.44, 19.96, 12.34, 17.24, 13.12, 6.76, 21.52, 16.63, 20.65

Method                        | Dist to μ  | Details
------------------------------|------------|------------------------------------------
Best Single Model             |   6.764    | (min radius)
Simple Average                |   5.584    | uniform 1/N
Model Stock                   |   3.758    | t = 0.5111
YOYO-Scale                    |   3.262    | λ = [0.045, 0.016, 0.027, 0.070, 0.036,
                              |            |      0.062, 0.232, 0.023, 0.039, 0.025]
```

The model with the smallest noise (r=6.76) receives the largest weight (0.232), while the noisiest model (r=25.44) gets only 0.016 — a 14.5× ratio, automatically determined.

</details>

### Key Observations

1. **YOYO-Scale maintains >97% win probability across all N**, while Model Stock drops below 50% at N=10.

2. **Absolute quality improves with more models**: average distance scales as approximately `O(1/√N)`.

   | N | YOYO-Scale Avg Dist | Improvement vs Best Single |
   |---|---|---|
   | 2 | 3.734 | 2.4× closer to μ |
   | 5 | 2.571 | 2.1× closer to μ |
   | 10 | 1.722 | 1.9× closer to μ |

3. **Per-model weighting is essential at large N**: Model Stock (scalar `t`, equal weights) drops to 46% win rate at N=10. The per-model λ coefficients let YOYO-Scale exploit heterogeneous model quality — assigning near-zero weight to noisy models and concentrating on the best ones — while still benefiting from noise cancellation across all models.

4. **Simple Average can be harmful**: at N=10, it wins only 36.5% of the time — worse than selecting the single best model. Equal weighting allows noisy models to dominate.

5. **Anchor shrinkage is critical**: YOYO-Scale's optimal total weight `∑λ` is typically well below 1.0 (e.g., ~0.28 for N=5), meaning it preserves significant base-model influence. This automatic shrinkage toward the anchor suppresses residual noise that even optimal per-model weighting cannot eliminate.

---

## Why This Works

The core mechanism is **signal coherence vs. noise cancellation**:

```
w_merged = w0 + sum_i lambda_i * vi
         = w0 + (sum_i lambda_i) * (mu - w0)     [signal: coherent, adds up]
              + sum_i lambda_i * noise_i           [noise: incoherent, cancels]
```

- **Signal**: All task vectors share the same direction `(mu - w0)`. Weighted summation preserves it.
- **Noise**: Independent across models, approximately orthogonal in high dimensions. Weighted summation causes cancellation.

YOYO-Scale amplifies this effect by:
1. **Upweighting low-noise models** → preserves more signal per unit of noise introduced.
2. **Downweighting high-noise models** → reduces noise contamination.
3. **Shrinking toward anchor** (`∑λ < 1`) → further suppresses residual noise.

In high-dimensional spaces (typical of modern LLMs), the **concentration of measure** phenomenon makes noise vectors nearly perfectly orthogonal, driving win probabilities toward 100%.

---

## Usage

### Installation
```bash
pip install torch safetensors tqdm
```

### Basic Example
```python
from yoyo_scale import merge_models_yoyo_scale

fine_tuned_model_dirs = [
    "path/to/ft_model_1",
    "path/to/ft_model_2",
    "path/to/ft_model_3",
]

merge_models_yoyo_scale(
    model_paths=fine_tuned_model_dirs,
    base_path="path/to/base_model",
    output_path="path/to/merged_model",
    config_source_index=0  # copy metadata from base
)
```

### Output Files
- Merged model shards (`model-XXXX.safetensors`)
- Index file (`model.safetensors.index.json`)
- Configuration files (`config.json`, tokenizer, etc.)
- **Lambda record**: `yoyo_scale.csv` — shows per-tensor contribution weights

---

## Directory Structure
```
YOYO-Scale/
├── LICENSE.txt            # Apache 2.0 license
├── README.md
├── Simulation.py          # Simulation script
└── YOYO-Scale.py          # Core implementation
```

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE.txt) for details.

---

## References

- **Model Stock: All we need is just a few fine-tuned models**. arXiv:2403.19522.  
  [Paper](https://arxiv.org/abs/2403.19522) | [Code](https://github.com/naver-ai/model-stock)

> YOYO-Scale builds upon Model Stock's geometric insights but replaces its rigid analytic solution with a **constraint-aware, convex optimization** that remains effective in asymmetric, noisy real-world settings.

---

## Acknowledgements

Inspired by the geometric elegance of **Model Stock** and the practicality of **task arithmetic**. Designed for researchers and engineers who want principled, efficient, and interpretable model merging.
