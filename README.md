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

### 1. Problem Setup

Given:
- A base model weight: `w0`
- `N` fine-tuned models: `w1, ..., wN`
- Task vectors: `vi = wi - w0`

We seek a merged weight:  
`w_merged = w0 + sum_{i=1}^N lambda_i * vi`

---

### 2. Optimal Condition: Consensus Maximization

YOYO Scale posits that the best update direction `Delta = sum_i lambda_i * vi` should be maximally aligned with the common signal shared across all fine-tuned models.

Under the thin-shell hypothesis (weights lie on a high-dimensional shell), the inner product `vi · vj` for `i != j` estimates the squared norm of the true center `||mu||^2`, while `||vi||^2` includes noise variance.

Thus, we define the target vector:  
`b_j = (1 / (N - 1)) * sum_{i != j} (vi · vj)`

---

### 3. Linear System

Let `G` be the Gram matrix of size `N x N`, where:  
`G[i, j] = vi · vj`

Solve the linear system:  
`G * lambda = b`

---

### 4. Constraints for Stability

- **Non-negativity**: `lambda_i >= 0` → clamp negative values to zero.
- **Normalization**: If `sum(lambda_i) > 1`, scale all `lambda_i` by `1 / sum(lambda_i)` to prevent overshooting.

---

### 5. Connection to Model Stock

When fine-tuned models satisfy:
- `||vi|| = l` (constant norm)
- `vi · vj = l^2 * cos(theta)` for all `i != j` (constant pairwise similarity)

Then the solution is uniform: `lambda = alpha * [1, 1, ..., 1]`, and YOYO Scale reduces to:

`w_merged = t * w_avg + (1 - t) * w0`  
where  
`t = (N * cos(theta)) / (1 + (N - 1) * cos(theta))`

This is exactly **Model Stock** (Equation 3 in the paper).

Thus, **YOYO Scale generalizes Model Stock** by removing the need for strict symmetry assumptions.

---

## 🛠️ Usage

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
├── LICENSE                # Apache 2.0
└── examples/
    └── merge_qwen.py      # Example script
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
