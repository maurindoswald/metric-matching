import torch

from metric_matching.systems.mm_system import load_metric_model_from_checkpoint
from optimization.riemannian_quadratic import DiffOptQuadratic


if __name__ == "__main__":
    D = 64
    d = 8
    h = 0.125

    model = load_metric_model_from_checkpoint(
        preferred_root="outputs/metric/sphere/2026-11-17_12-45-50/checkpoints",
        epoch=699,
    ).eval()

    torch.manual_seed(0)
    M = torch.randn(D, D)
    A = M.T @ M + 1e-2 * torch.eye(D)

    base = torch.randn(1, d + 1)
    base = base / torch.linalg.vector_norm(base, dim=1, keepdim=True)
    x0 = torch.zeros(1, D)
    x0[:, : d + 1] = base

    def retract_to_sphere(x):
        x = x.clone()
        x[d + 1 :] = 0
        x[: d + 1] /= torch.linalg.vector_norm(x[: d + 1])
        return x

    optimizer = DiffOptQuadratic(model, A, x0, retraction=retract_to_sphere)
    approx = optimizer.optimize(h=h)

    # The sphere is embedded in the first d+1 coordinates, so its
    # constrained quadratic minimum is the smallest eigenvalue of this block.
    eigenvalues, eigenvectors = torch.linalg.eigh(A[: d + 1, : d + 1])
    gt_minimum = eigenvalues[0]
    gt_minimizer = torch.zeros(D)
    gt_minimizer[: d + 1] = eigenvectors[:, 0]

    approx_minimizer = retract_to_sphere(approx.x)
    minimizer_difference = torch.minimum(
        torch.linalg.vector_norm(approx_minimizer - gt_minimizer),
        torch.linalg.vector_norm(approx_minimizer + gt_minimizer),
    )
    approx_sphere_minimum = approx_minimizer @ A @ approx_minimizer

    print(f"status: {approx.status} (converged={approx.converged})")
    print(f"iterations: {approx.iterations}")
    print(f"approximated minimizer norm: {torch.linalg.vector_norm(approx.x).item():.6f}")
    print(f"approximated minimizer direction: {approx_minimizer[: d + 1]}")
    print(f"ground-truth minimizer direction: {gt_minimizer[: d + 1]}")
    print(f"approximated minimum: {approx.fun.item():.6f}")
    print(f"approximated minimum after normalization: {approx_sphere_minimum.item():.6f}")
    print(f"ground-truth minimum on sphere: {gt_minimum.item():.6f}")
    print(f"minimizer difference (sign-invariant): {minimizer_difference.item():.6f}")
