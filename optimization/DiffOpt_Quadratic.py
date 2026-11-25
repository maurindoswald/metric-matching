import torch
import numpy as np


# class DiffusionOptimization():

import torch


class DiffOpt_Quadratic:

    def __init__(
        self,
        model,
        A,
        x0,
        max_iterations=500,
        beta=0.5,
        c1=1e-4,
        iter_ls_max=30,
        convergence_eps=1e-6,
        initial_step_size=0.001,
    ):

        self.model = model
        self.A = A

        self.d = 8

        self.current_x = x0.clone()

        self.max_iterations = max_iterations
        self.beta = beta
        self.c1 = c1
        self.iter_ls_max = iter_ls_max
        self.convergence_eps = convergence_eps

        self.step_size = initial_step_size

    def loss_function_val(self, x):
        return torch.sum((x @ self.A) * x, dim=-1)

    # def calc_projector(self, h):

    #     pred = self.model(h, self.current_x)
    #     # Orthonormalize the tangent basis
    #     Q, _ = torch.linalg.qr(
    #         pred.transpose(-2, -1),
    #         mode="reduced"
    #     )
    #     return Q @ Q.transpose(-2, -1)

    def calc_projector(self, h):
        pred = self.model(h, self.current_x)    
        U = pred.transpose(1,2)
        C = U.transpose(-1,-2) @ U
        evals, V = torch.linalg.eigh(C)        
        U_pred = (U @ V[..., -self.d:]) / evals[..., -self.d:].clamp_min(
                1e-12
            ).sqrt().unsqueeze(-2)
        proj = U_pred @ U_pred.transpose(-1,-2)
        return proj

        # _, S, Vh = torch.linalg.svd(pred, full_matrices=False)
        # V = Vh[..., :self.d, :]                                
        # return V.transpose(-2, -1) @ V   
    
    def exact_projector(self, d):
        B, D = self.current_x.shape
        xt = self.current_x[:, :d+1]
        n = xt / torch.norm(xt, dim=-1, keepdim=True)
        P = torch.zeros(B, D, D, dtype=self.current_x.dtype)
        P[:, :d+1, :d+1] = torch.eye(d+1) - n.unsqueeze(-1) @ n.unsqueeze(-2)
        return P

    def calc_gradient(self, projector):

        # Euclidean gradient
        grad = 2.0 * (self.current_x @ self.A)
        # Projected gradient
        return (
            grad.unsqueeze(-2) @ projector
        ).squeeze(-2)


    def line_search(self, grad):

        alpha = self.step_size
        f_current = self.loss_function_val(self.current_x)
        grad_norm_sq = torch.sum(grad**2, dim=-1)

        for _ in range(self.iter_ls_max):

            candidate = self.current_x - alpha * grad
            # candidate = candidate / torch.norm(candidate, dim=-1, keepdim=True)
            f_candidate = self.loss_function_val(candidate)
            
            armijo_rhs = (f_current - self.c1 * alpha * grad_norm_sq)

            if torch.all(f_candidate <= armijo_rhs):
                self.step_size = alpha
                return candidate

            alpha *= self.beta

        return self.current_x


    def optimize(self, h_value=0.125):

        h = torch.full(
            (self.current_x.shape[0],),
            h_value,
            device=self.current_x.device,
            dtype=self.current_x.dtype
        )

        history = []

        for iteration in range(self.max_iterations):

            projector = self.calc_projector(h)
            # projector = self.exact_projector(self.d)
            grad = self.calc_gradient(projector)
            grad_norm = torch.linalg.norm(grad).item()
            loss = self.loss_function_val(self.current_x).mean().item()

            history.append({
                "iteration": iteration,
                "loss": loss,
                "gradient_norm": grad_norm,
                "step_size": self.step_size,
            })

            if grad_norm < self.convergence_eps:
                break

            old_x = self.current_x.clone()
            self.current_x = self.line_search(grad)
            step_norm = torch.linalg.norm(
                self.current_x - old_x
            ).item()

            if step_norm < self.convergence_eps:
                break
            

        return {
            "minimizer": self.current_x,
            "minimum": self.loss_function_val(self.current_x),
            "iterations": iteration + 1,
            "history": history,
        }

    def ground_truth(self, d):
        A_local = self.A[:d+1, :d+1]

        eigval, eigvec = torch.linalg.eigh(A_local)

        v_min = eigvec[:, 0]
        minimizer = torch.zeros_like(self.current_x)

        minimizer[:, :d+1] = v_min

        return {
            "gt_minimizer": minimizer,
            "gt_minimum": eigval[0],
        }
    
    def evaluate_solution(self, approx, gt, A, d):

        x = approx["minimizer"]
        x_gt = gt["gt_minimizer"]

        f_approx = torch.sum((x @ A) * x, dim=-1)
        f_gt = gt["gt_minimum"]

        objective_error = torch.abs(f_approx - f_gt)

        diff1 = torch.linalg.norm(x - x_gt)
        diff2 = torch.linalg.norm(x + x_gt)
        minimizer_error = torch.minimum(diff1, diff2)

        sphere_error = torch.abs(torch.linalg.norm(x[:, :d+1], dim=1) - 1)

        outside_error = torch.linalg.norm(x[:, d+1:],dim=1)

        manifold_error = sphere_error + outside_error

        return {
            "objective_error": objective_error,
            "minimizer_error": minimizer_error,
            "manifold_error": manifold_error,
        }    


# class DiffOpt_Quadratic:
#     """Optimize f(x)=x.T@A@x on a Manifold expressed by the learned CDC operator"""
    
#     def __init__(
#         self, 
#         model, 
#         A, 
#         x0, 
#         max_iterations = 500,
#         beta=0.95, 
#         iter_ls_max = 100, 
#         convergence_eps = 1e-3
#     ) -> None:
    
#         self.model = model
#         self.A = A

#         self.current_x = x0
#         self.last_x = 0
#         self.convergence_eps = convergence_eps
#         self.current_grad = 0
#         self.proj = None

#         self.max_iterations = max_iterations
        
#         # line search parameter
#         self.beta = beta
#         self.iter_ls_max = iter_ls_max
#         self.current_step_sz = 0.1
        

#     def calc_step_size(self):
#         c1 = 1e-4
#         c2 = 0.9
#         alpha = 0.1
#         alpha_ls = 0.1
#         for _ in range(self.iter_ls_max):
#             # alpha = self.current_step_sz
#             x_k = self.current_x
#             # proj = self.model(h, x_k)
#             # self.calc_intrinsic_grad()
#             # gamma = self.current_x - alpha*self.proj*self.current_grad
#             gamma = self.current_x - alpha*self.current_grad
#             f = self.loss_function_val(x_k)
#             armijo = self.loss_function_val(gamma) < f - c1*alpha*(self.current_grad*self.current_grad).sum()
#             # wolff = torch.dot()
#             # if armijo and wolff:
#                 # alpha_ls = alpha
#             if armijo:
#                 alpha_ls = alpha
#                 break
#             alpha = self.beta*alpha

#         self.current_step_sz = alpha_ls


#     def loss_function_val(self,x):
#         return torch.sum(x@self.A * x, dim=1)
 
#     def calc_intrinsic_grad(self):
#         # proj = self.model(h, self.current_x)
#         eukl_grad = self.current_x @ (self.A + torch.transpose(self.A, -2, -1)) 
#         self.current_grad = eukl_grad @ self.proj 

#     def line_search(self):    
#         self.calc_step_size()
#         next_x = self.current_x - self.current_step_sz*(self.current_grad) #@ self.proj)
#         self.last_x = self.current_x
#         self.current_x = next_x

#     def optimize(self,h_value=7.0):
#         # h_tensor = torch.as_tensor(h, device=self.current_x.device, dtype=self.current_x.dtype)
#         h = torch.full((self.current_x.shape[0],), h_value, device=self.current_x.device, dtype=self.current_x.dtype)
#         iteration=0
#         for i in range(self.max_iterations):
#             pred = self.model(h,self.current_x)
#             self.proj = pred.transpose(-2,-1) @ pred
#             self.proj = self.proj.squeeze(0)
#             self.calc_intrinsic_grad()
#             self.line_search()
#             if torch.norm(self.current_x-self.last_x) < self.convergence_eps:
#                 # print(f"Minimizer: {self.current_x}")
#                 # print(f"Minimum: {self.loss_function_val()}")
#                 iteration = i
#                 break
#         return {"minimizer": self.current_x, "minimum": self.loss_function_val(self.current_x), "iterations": iteration+1}

#     # def ground_truth(self, d):
#     #     A_local = self.A[:d+1, :d+1]
#     #     eigval, eigvec = torch.linalg.eigh(A_local)

#     #     v_min = eigvec[:, 0]
#     #     minimizer = torch.zeros_like(self.current_x)

#     #     minimizer[:, :d+1] = v_min

#     #     return {
#     #         "gt_minimizer": minimizer,
#     #         "gt_minimum": eigval[0],
#     #     }
    


    