import torch
from optimization.DiffOpt_Quadratic import DiffOpt_Quadratic
from metric_matching.systems.mm_system import load_metric_model_from_checkpoint
from metric_matching.data.sphere.performance import (
    mean_frobenius_basis_distance,
    tangent_space_basis,
)



if __name__ == "__main__":
    
    D = 64
    d = 8
    B = 50

    model = load_metric_model_from_checkpoint(preferred_root="outputs/metric/sphere/2026-11-25_11-45-44/checkpoints", epoch=2899).eval()
    torch.manual_seed(0)
    
    M = torch.randn(D, D)
    A = (M.T @ M) + 1e-2*torch.eye(D)
    
    base = torch.randn((B, d+1), dtype=torch.float32)
    base = base/torch.norm(base, dim=1, keepdim=True)
    x_test_point = torch.zeros((B, D), dtype=torch.float32)
    x_test_point[:, :d+1] = base
    x_test_point.squeeze()

    x_test_point = x_test_point[49,:].unsqueeze(0)
    model.eval()


    optimizer = DiffOpt_Quadratic(model, A, x_test_point)
    
    appro_op = optimizer.optimize(h_value=0.0625)
    gt_op = optimizer.ground_truth(d)

    evaluation = optimizer.evaluate_solution(appro_op, gt_op, A, d)
    dev_diff = torch.minimum(torch.abs(appro_op["minimizer"]-gt_op["gt_minimizer"]), torch.abs(appro_op["minimizer"]+gt_op["gt_minimizer"]))
    dev = torch.mean(dev_diff)

    print(f"initial cost function value: {optimizer.loss_function_val(x_test_point)}")
    print(f"Approximate minimum: {appro_op["minimum"].item():.6f}")
    print(f"Ground truth minimum: {gt_op["gt_minimum"].item():.6f}")
    print(f"Iteration: {appro_op["iterations"]}")

    print(f"Objective error: {evaluation["objective_error"].item():.6f}")
    print(f"Minimizer error: {evaluation["minimizer_error"].item():.6f}")
    print(f"Manifold error: {evaluation["manifold_error"].item():.6f}")
    print(f"average deviation: {dev:.6f}")
    
    
    
    # gt_op = optimizer.groud_truth(d)
    # print(f"approximated minimum: {appro_op["minimum"]}")
    # print(f"gt minimum: {gt_op["gt_minimum"]}")

    # diff1 = torch.norm(appro_op["minimizer"]-gt_op["gt_minimizer"])
    # diff2 = torch.norm(appro_op["minimizer"]+gt_op["gt_minimizer"])
    # diff = min(diff1,diff2)

    # dev = torch.mean(torch.abs(appro_op["minimizer"]-gt_op["gt_minimizer"]))
    # # diff = (appro_op["minimizer"]-gt_op["gt_minimum"])
    # print(f"difference of minimizer: {diff}")
    # print(f"average deviation: {dev}")
