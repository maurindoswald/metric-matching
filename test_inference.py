#%%
import torch
from metric_matching.callbacks.sphere_metric_eval import SphereMetricEvalCallback
from metric_matching.data.sphere.performance import (
    mean_frobenius_basis_distance,
    tangent_space_basis,
)
from metric_matching.systems.mm_system import load_metric_model_from_checkpoint
from metric_matching.utils.spectra import batched_eig_of_UtU


d = 8
D = 64
B = 1000



def evaluate_point(model, points, h_value, d, D):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x = torch.as_tensor(points, device=device, dtype=dtype)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[1] != D or x.shape[0] == 0:
        raise ValueError(
            f"Expected points with shape ({D},) or (B, {D}) where B > 0, "
            f"got {tuple(x.shape)}."
        )

    h = torch.full((x.shape[0],), h_value, device=device, dtype=dtype)
    evaluator = SphereMetricEvalCallback(bandwidths=[h_value], d=d, D=D)

    with torch.no_grad():
        factor = model(h, x)
        true_basis = tangent_space_basis(x, d=d, D=D)
        predicted_basis = evaluator._predict_tangent_basis(model, factor)
        # distance = mean_frobenius_basis_distance(true_basis, predicted_basis)
        P_gt = true_basis @ true_basis.transpose(-1, -2)
        P_pred = predicted_basis @ predicted_basis.transpose(-1, -2)

        distances = torch.linalg.norm(
            P_gt - P_pred,
            dim=(-2, -1)
        )
        distance = distances.mean()

    return {
        "mean_distance": distance,
        "distances": distances,
        "ground_truth_basis": true_basis,
        "predicted_basis": predicted_basis,
        "factor": factor,
    }


# model = load_metric_model_from_checkpoint(preferred_root="outputs/metric/sphere/2026-11-17_12-45-50/checkpoints", epoch=2799).eval()
model = load_metric_model_from_checkpoint(preferred_root="outputs/metric/sphere/2026-11-25_11-45-44/checkpoints", epoch=2999).eval()

torch.manual_seed(0)
base = torch.randn((B, d+1), dtype=torch.float32)
base = base/torch.norm(base, dim=1, keepdim=True)
x_test_point = torch.zeros((B, D), dtype=torch.float32)

x_test_point[:, :d+1] = base

result = evaluate_point(model, x_test_point, h_value=0.0625, d=d, D=D)
U = result["factor"]
evals, evecs = batched_eig_of_UtU(U)
distances = result["distances"]


print(f"U shape: {U.shape}")
print(f"Gamma shape: {(U.transpose(-2,-1)@U).shape}")
print(f"eigenvalues: {evals.shape}")
print(f"eigenvectors: {evecs.shape}")
print(f"mean tangent-space distance: {result['mean_distance'].item():.6f}")


print(f"Mean: {distances.mean().item():.6f}")
print(f"Std:  {distances.std().item():.6f}")
print(f"Min:  {distances.min().item():.6f}")
print(f"Max:  {distances.max().item():.6f}")
