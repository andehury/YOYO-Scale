# YOYO-Scale: Geometry-Driven Task Vector Scaling Without Training

[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

**YOYO-Scale** is a robust, geometry-aware model merging method that **generalizes Model Stock** by computing **per-tensor, nonnegative fusion coefficients under an anchor-preserving constraint**. It requires no extra training and works reliably in realistic (asymmetric) settings.

> **Key Insight**: Instead of relying on Model Stock’s strong symmetry assumptions (e.g., identical pairwise angles), YOYO-Scale formulates merging as a **constrained convex optimization** that (i) estimates the unknown center-consensus direction from cross-model agreement and (ii) finds the best merged tensor **within the convex hull of `{base, fine-tuned models}`**.

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
- Assume there exists an unknown “ideal center” `mu`, representing the common target of fine-tuning.

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

Since `mu` is unobservable, YOYO-Scale estimates `b` using Model Stock’s geometric insights.

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

Under Model Stock’s strong symmetry assumptions:
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
├── LICENSE.txt            # Apache 2.0
├── README.md
└── YOYO-Scale.py          # Core implementation
```

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE.txt) for details.

---

## References

- **Model Stock: All we need is just a few fine-tuned models**. arXiv:2403.19522.  
  [Paper](https://arxiv.org/abs/2403.19522) | [Code](https://github.com/naver-ai/model-stock)

> YOYO-Scale builds upon Model Stock’s geometric insights but replaces its rigid analytic solution with a **constraint-aware, convex optimization** that remains effective in asymmetric, noisy real-world settings.

---

## Acknowledgements

Inspired by the geometric elegance of **Model Stock** and the practicality of **task arithmetic**. Designed for researchers and engineers who want principled, efficient, and interpretable model merging.
