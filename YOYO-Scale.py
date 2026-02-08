import os
import json
import torch
import shutil
import csv
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file
from typing import List, Dict, Tuple


def load_tensor_index(model_path: str) -> Dict[str, str]:
    """Load mapping from tensor names to shard filenames."""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        return index_data['weight_map']
    else:
        single_file = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single_file):
            raise FileNotFoundError(f"No model file found in {model_path}")
        with safe_open(single_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
        return {k: "model.safetensors" for k in keys}


def calculate_yoyo_scale_weight(
        base_tensor: torch.Tensor,
        model_tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[float]]:
    """
    Compute merged tensor using YOYO Scale method.

    Optimization (per tensor):
        min_{lambda}  0.5 * lambda^T (G + eps I) lambda - b^T lambda
        s.t.          lambda >= 0,  sum(lambda) <= 1

    where:
        v_i = (w_i - w0),  vs = stack(v_i) in R^{N x D}
        G = vs vs^T
        b_j = mean_{i != j} <v_i, v_j>  (YOYO estimate; excludes diagonal/noise term)
    """
    N = len(model_tensors)
    if N == 0:
        return base_tensor.clone(), []

    # Keep original behavior for N==1 (cannot estimate b from cross terms).
    if N == 1:
        return model_tensors[0].clone(), [1.0]

    # Flatten and convert to float64 for numerical stability
    w0 = base_tensor.flatten().to(torch.float64)
    # vs: [N, D], where each row is (w_i - w0)
    vs = torch.stack([(w.flatten().to(torch.float64) - w0) for w in model_tensors], dim=0)

    # 1) Gram matrix: G[i, j] = v_i · v_j
    G = torch.mm(vs, vs.t())

    # 2) YOYO estimate for b: b_j = mean_{i != j} (v_i · v_j)
    b = (G.sum(dim=1) - G.diagonal()) / (N - 1)

    # 3) Solve the constrained QP:
    #       min 0.5*lambda^T H lambda - b^T lambda
    #       s.t. lambda>=0, 1^T lambda <= 1
    #
    #    KKT-based active-set (primal-dual) in N-dim; N is typically small.
    eps = 1e-12
    H = G + eps * torch.eye(N, device=G.device, dtype=torch.float64)

    tol = 1e-12
    max_iter = 10 * N + 50

    lambdas = torch.zeros(N, device=G.device, dtype=torch.float64)

    # Active set mask for lambda_i > 0
    active = torch.ones(N, device=G.device, dtype=torch.bool)

    # Helper to solve restricted system robustly
    # Helper to solve restricted system robustly
    def _solve(H_sub: torch.Tensor, y_sub: torch.Tensor) -> torch.Tensor:
        try:
            return torch.linalg.solve(H_sub, y_sub)
        except torch.linalg.LinAlgError:
            # Fall back to the least squares; ensure correct shape
            result = torch.linalg.lstsq(H_sub, y_sub.unsqueeze(1))
            sol = result.solution.squeeze(1)
            # Ensure same dtype and device
            return sol.to(dtype=y_sub.dtype, device=y_sub.device)

    for _ in range(max_iter):
        idx = torch.nonzero(active, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            lambdas.zero_()
            break

        Hs = H.index_select(0, idx).index_select(1, idx)
        bs = b.index_select(0, idx)
        ones_s = torch.ones(idx.numel(), device=G.device, dtype=torch.float64)

        # Candidate with nu = 0 (sum constraint inactive)
        v = _solve(Hs, bs)  # v = Hs^{-1} b_s
        sum_v = float(v.sum().item())

        nu_val = 0.0
        if sum_v > 1.0 + tol:
            # Enforce sum constraint as active: sum(lambda)=1, nu>=0
            u = _solve(Hs, ones_s)  # u = Hs^{-1} 1
            denom = float((ones_s @ u).item())
            if abs(denom) > 1e-30:
                nu_tmp = (sum_v - 1.0) / denom
                # For inequality constraint, require nu>=0; else keep nu=0
                if nu_tmp > 0.0:
                    nu_val = float(nu_tmp)
                    v = v - nu_val * u  # lambda_s = Hs^{-1}(b_s - nu*1)

        lambdas_new = torch.zeros_like(lambdas)
        lambdas_new[idx] = v

        # Primal feasibility on non-negativity: if violated, drop those vars from active set
        neg_mask = lambdas_new[idx] <= tol
        if bool(neg_mask.any().item()):
            active[idx[neg_mask]] = False
            continue

        # Dual feasibility for inactive constraints:
        #   gamma = H lambda - b + nu*1  must satisfy gamma_i >= 0 for inactive i
        g = torch.mv(H, lambdas_new) - b + nu_val * torch.ones_like(b)

        inactive = ~active
        if bool(inactive.any().item()):
            g_inact = g[inactive]
            min_g = float(g_inact.min().item())
            if min_g < -tol:
                # Add the most violating inactive variable back to active set
                inact_idx = torch.nonzero(inactive, as_tuple=False).squeeze(1)
                viol_local = int(torch.argmin(g_inact).item())
                viol_idx = int(inact_idx[viol_local].item())
                active[viol_idx] = True
                continue

        # Also ensure inequality sum constraint consistency:
        # if nu==0, allow sum<=1; if nu>0, sum should be ~1
        t_sum = float(lambdas_new.sum().item())
        if nu_val == 0.0 and t_sum > 1.0 + 1e-8:
            # Numerical guard: if slightly above, activate sum constraint next round by forcing active remain
            # (will be handled naturally next iter), but also avoid infinite loop:
            pass

        lambdas = lambdas_new
        break

    # Numerical cleanup: enforce constraints softly
    lambdas = torch.clamp(lambdas, min=0.0)
    t_sum = lambdas.sum()
    if t_sum > 1.0:
        lambdas = lambdas / t_sum  # should rarely trigger if QP solved well

    final_lambdas_list = lambdas.tolist()

    # 4) Compute final update: w = w0 + sum(lambda_i * v_i)
    update = torch.mv(vs.t(), lambdas)
    merged_flat = w0 + update
    merged = merged_flat.view_as(base_tensor).to(base_tensor.dtype)

    return merged, final_lambdas_list


def merge_models_yoyo_scale(
        model_paths: List[str],
        base_path: str,
        output_path: str,
        config_source_index: int = 0
):
    """
    Merge multiple fine-tuned models into a base model using the YOYO Scale method.
    Saves per-model lambda coefficients for each tensor.

    Args:
        model_paths (List[str]): Paths to fine-tuned models.
        base_path (str): Path to base model.
        output_path (str): Output directory for merged model.
        config_source_index (int): Index of model to copy config/index from (0 = base).
    """
    print(f"🚀 Starting YOYO Scale Model Merge")
    print(f"🔹 Base Model: {base_path}")
    print(f"🔹 Models to merge ({len(model_paths)}):")
    for idx, model_path in enumerate(model_paths, 1):
        print(f"  - [{idx}] {model_path}")
    print(
        f"🔹 Config source (shard & metadata): {'Base' if config_source_index == 0 else f'Model {config_source_index}'}")
    print(f"🔹 Output Path: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    all_model_paths = [base_path] + model_paths
    if config_source_index < 0 or config_source_index >= len(all_model_paths):
        raise ValueError(f"config_source_index must be in range [0, {len(all_model_paths) - 1}]")

    config_source_path = all_model_paths[config_source_index]

    print("🔍 Building tensor indices for all models...")
    base_weight_map = load_tensor_index(base_path)
    model_weight_maps = [load_tensor_index(p) for p in model_paths]
    config_weight_map = load_tensor_index(config_source_path)

    config_shards: Dict[str, List[str]] = {}
    for tensor_name, shard_file in config_weight_map.items():
        config_shards.setdefault(shard_file, []).append(tensor_name)

    tensor_lambda_records = []  # Will store per-model lambdas

    for shard_file, tensor_names in config_shards.items():
        print(f"\n📦 Processing Shard: {shard_file}")
        merged_state_dict = {}

        progress_bar = tqdm(tensor_names, unit="tensor", desc="Merging")
        for tensor_name in progress_bar:
            try:
                # Load base tensor
                base_shard = base_weight_map.get(tensor_name)
                if base_shard is None:
                    raise KeyError(f"Tensor '{tensor_name}' not found in base model.")
                with safe_open(os.path.join(base_path, base_shard), framework="pt", device="cpu") as f:
                    base_t = f.get_tensor(tensor_name).float()

                # Load fine-tuned tensors
                model_ts = []
                for ft_model_path, weight_map in zip(model_paths, model_weight_maps):
                    ft_shard = weight_map.get(tensor_name)
                    if ft_shard is None:
                        raise KeyError(f"Tensor '{tensor_name}' not found in model {ft_model_path}.")
                    with safe_open(os.path.join(ft_model_path, ft_shard), framework="pt", device="cpu") as f:
                        model_ts.append(f.get_tensor(tensor_name).float())

                # Skip non-float tensors (e.g., int for embedding positions)
                if base_t.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                    merged_t = base_t
                    final_lambdas = [1.0] * len(model_ts)  # dummy
                else:
                    merged_t, final_lambdas = calculate_yoyo_scale_weight(base_t, model_ts)

                merged_state_dict[tensor_name] = merged_t.to(torch.bfloat16)

                # Record lambdas (as strings for CSV)
                lambda_strs = [f"{lam:.6f}" for lam in final_lambdas]
                tensor_lambda_records.append({
                    "tensor_name": tensor_name,
                    **{f"model_{i}_lambda": lambda_strs[i] for i in range(len(final_lambdas))}
                })

                del base_t, model_ts, merged_t

            except Exception as e:
                print(f"\n❌ Error merging tensor '{tensor_name}': {e}")
                # Fallback: copy tensor from config source
                with safe_open(os.path.join(config_source_path, shard_file), framework="pt", device="cpu") as f:
                    merged_state_dict[tensor_name] = f.get_tensor(tensor_name).to(torch.bfloat16)
                # Use NaN for all lambdas
                nan_lambdas = ["nan"] * len(model_paths)
                tensor_lambda_records.append({
                    "tensor_name": tensor_name,
                    **{f"model_{i}_lambda": nan_lambdas[i] for i in range(len(nan_lambdas))}
                })

        save_file(merged_state_dict, os.path.join(output_path, shard_file))
        print(f"💾 Saved shard: {shard_file}")
        del merged_state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save per-model lambdas to CSV
    if model_paths:
        fieldnames = ["tensor_name"] + [f"model_{i}_lambda" for i in range(len(model_paths))]
        t_csv_path = os.path.join(output_path, "yoyo_scale.csv")
        with open(t_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in tensor_lambda_records:
                writer.writerow(record)
        print(f"📊 Saved per-model YOYO Scale lambdas to: {t_csv_path}")

    # Copy index file if exists
    index_src = os.path.join(config_source_path, "model.safetensors.index.json")
    if os.path.exists(index_src):
        shutil.copy(index_src, os.path.join(output_path, "model.safetensors.index.json"))
        print("✅ Copied index.json")

    # Copy configuration files
    print("\n📑 Copying configuration files...")
    config_files = [
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json"
    ]
    for cf in config_files:
        src = os.path.join(config_source_path, cf)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_path, cf))
            print(f"  - Copied {cf}")

    print(f"\n✨ All done! YOYO Scale merged model saved to: {output_path}")


# --- Main Execution ---
if __name__ == "__main__":
    fine_tuned_model_dirs = [
        r"path/to/ft_model_1",
        r"path/to/ft_model_2",
        r"path/to/ft_model_3",
    ]

    path_base_model = r"path/to/base_model"
    output_dir = r"path/to/merged_model"
    config_index = 0

    errors = []
    if not os.path.exists(path_base_model):
        errors.append(f"Base model not found: {path_base_model}")
    for model_idx, model_dir in enumerate(fine_tuned_model_dirs, 1):
        if not os.path.exists(model_dir):
            errors.append(f"Fine-tuned model [{model_idx}] not found: {model_dir}")

    if errors:
        print("❌ Errors:")
        for err in errors:
            print(f"  - {err}")
        exit(1)

    merge_models_yoyo_scale(
        model_paths=fine_tuned_model_dirs,
        base_path=path_base_model,
        output_path=output_dir,
        config_source_index=config_index

    )


