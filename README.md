# YOYO Scale: Generalized Task Vector Scaling for Model Merging

[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

**YOYO Scale** is a robust, geometry-aware model merging method that **generalizes Model Stock** by adaptively computing per-tensor fusion coefficients through consensus-based orthogonal projection. It requires no strong symmetry assumptions and works reliably across diverse architectures and tasks.

> **Key Insight**: Instead of assuming all fine-tuned models are symmetrically distributed around an ideal center (as in Model Stock), YOYO Scale *estimates* the optimal direction within the span of task vectors by maximizing agreement among cross-model weight updates.

---

## Highlights

- **Generalizes Model Stock**: Recovers Model Stock as a special case under its geometric assumptions.
- **Per-tensor adaptive coefficients**: Learns layer-wise (or even tensor-wise) λ weights via a closed-form linear system.
- **Noise-robust**: Excludes self-correlation to avoid overfitting to model-specific noise.
- **Efficient**: No extra training; only requires loading fine-tuned checkpoints.
- **Interpretable**: Outputs `yoyo_scale.csv` with per-model λ values for analysis.

---

## Theoretical Foundation

### Step 1: Problem Setup and Notation

- Without loss of generality, set the base model weight as the origin: `w0 = 0` (since all operations are translation-invariant).
- We have `N` fine-tuned models with weights: `w1, w2, ..., wN`.
- Define task vectors: `vi = wi - w0 = wi`.
- Assume there exists an unknown "ideal center" `mu`, representing the common target of all fine-tuning tasks.
- Our goal is to find a merged weight `w_H` that lies in the span of `{v1, ..., vN}` and is as close as possible to `mu`.

Thus, we write:
```
w_H = sum_{i=1}^N lambda_i * vi = V * lambda
```
where `V` is an `N x d` matrix whose rows are `v1, ..., vN`, and `lambda` is an `N`-dimensional coefficient vector.

---

### Step 2: Optimality Condition — Orthogonal Projection

In Euclidean space, `w_H` is the best approximation of `mu` within the subspace if and only if the residual `(mu - w_H)` is orthogonal to every basis vector `vj`:
```
(mu - w_H) · vj = 0,    for all j = 1, ..., N
```

Substituting `w_H = sum_i lambda_i * vi`, we get:
```
mu · vj = sum_{i=1}^N lambda_i * (vi · vj)
```

In matrix form, this becomes:
```
G * lambda = b
```
where:
- `G[i, j] = vi · vj` (the Gram matrix),
- `b[j] = mu · vj`.

This is the **universal starting point** for all projection-based merging methods.

---

### Step 3: Estimating `b` Using Model Stock’s Geometric Assumptions

Since `mu` is unobservable, we use insights from Model Stock:

#### **Assumption A (Orthogonal Anchor)**:
The vector from each fine-tuned model to the center is perpendicular to the vector from the base model to the center:
```
(wi - mu) ⟂ (w0 - mu)
```
Since `w0 = 0`, this implies:
```
(vi - mu) · (-mu) = 0  →  vi · mu = ||mu||^2
```
So theoretically, **`b[j] = ||mu||^2` for all `j`** — a constant.

#### **Assumption B (Independent Noise / Thin Shell)**:
For `i ≠ j`, the noise vectors are approximately orthogonal:
```
(wi - mu) ⟂ (wj - mu)
```
Expanding gives:
```
vi · vj = ||mu||^2,    for i ≠ j
```
But the self-inner product includes noise variance:
```
||vi||^2 = ||mu||^2 + ||wi - mu||^2 = ||mu||^2 + sigma^2
```
where `sigma^2` is the noise variance.

> **Key Insight**:  
> - **Cross-model inner products** (`vi · vj` for `i ≠ j`) provide an **unbiased estimate** of `||mu||^2`.  
> - **Self-inner products** (`||vi||^2`) are **contaminated by noise variance**.

Therefore, to robustly estimate `b[j] = ||mu||^2`, we **exclude the self-term** and average over other models:
```
b[j] = (1 / (N - 1)) * sum_{i ≠ j} (vi · vj)
```
This is the **core innovation of YOYO Scale**.

---

### Step 4: Solving the System with Stability Constraints

We solve:
```
G * lambda = b
```

Because `G` may be ill-conditioned, we add a small regularization term:
```
(G + epsilon * I) * lambda = b,    where epsilon = 1e-12
```

To ensure physical meaning and numerical stability, we apply two constraints:
1. **Non-negativity**: `lambda_i >= 0` → clamp negative values to zero.
2. **Normalization**: If `sum(lambda_i) > 1`, scale all coefficients by `1 / sum(lambda_i)` to prevent overshooting.

The final merged weight is:
```
w_merged = w0 + sum_{i=1}^N lambda_i * (wi - w0)
```

---

### Step 5: Showing Model Stock Is a Special Case of YOYO Scale

Under Model Stock’s strong symmetry assumptions:
- `||vi||^2 = l^2` (constant norm),
- `vi · vj = l^2 * cos(theta)` for all `i ≠ j` (constant pairwise similarity).

Then the Gram matrix becomes:
```
G = l^2 * [ (1 - cos(theta)) * I + cos(theta) * ones(N, N) ]
```
and the target vector is:
```
b = l^2 * cos(theta) * [1, 1, ..., 1]^T
```

By symmetry, the solution must be uniform: `lambda = alpha * [1, 1, ..., 1]`.

Substituting into `G * lambda = b`:
```
l^2 * [ (1 - cos(theta)) * alpha + N * cos(theta) * alpha ] = l^2 * cos(theta)
→ alpha = cos(theta) / (1 + (N - 1) * cos(theta))
```

The total weight sum is:
```
t = sum(lambda_i) = N * alpha = (N * cos(theta)) / (1 + (N - 1) * cos(theta))
```

Thus, the merged weight becomes:
```
w_merged = t * w_avg + (1 - t) * w0
```
where `w_avg = (1/N) * sum_i wi`.

This is exactly **Model Stock**.

Thus, **YOYO Scale generalizes Model Stock** by removing the need for strict symmetry assumptions.

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
yoyo_scale/
├── yoyo_scale.py          # Core implementation
├── README.md              
└── LICENSE                # Apache 2.0
```

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

---

## References

- **Model Stock: All we need is just a few fine-tuned models**. arXiv:2403.19522.  
  [Paper](https://arxiv.org/abs/2403.19522) | [Code](https://github.com/naver-ai/model-stock)

> YOYO Scale builds upon Model Stock’s geometric insights but replaces its rigid analytic solution with a flexible, data-driven solver that works in realistic, asymmetric settings.

---

## Acknowledgements

Inspired by the geometric elegance of **Model Stock** and the practicality of **task arithmetic**. Designed for researchers and engineers who want principled, efficient, and interpretable model merging.
