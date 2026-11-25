from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass
class OptimizationResult:
    """Result of a pointwise Riemannian gradient descent run."""

    x: Tensor
    fun: Tensor
    iterations: int
    converged: bool
    status: str


class DiffOptQuadratic:
    """Minimize x.T @ A @ x using a learned carré du champ operator.

    The model is expected to be an ``MMSystem`` in evaluation mode. For
    ``smart_training=True``, it returns a factor U and this class forms the
    learned operator as U.T @ U. Otherwise, the model is expected to return
    the full operator matrix.

    This is a pointwise optimizer: ``x0`` may have shape ``[D]`` or ``[1, D]``.
    The model output is used as the inverse-Riemannian-metric operator in the
    gradient flow, so each update is ``x <- x - alpha * Gamma(x) @ grad f(x)``.
    """

    def __init__(
        self,
        model: nn.Module,
        A: Tensor,
        x0: Tensor,
        *,
        beta: float = 0.5,
        alpha_init: float = 1.0,
        max_iterations: int = 500,
        max_line_search: int = 30,
        armijo_c1: float = 1e-4,
        retraction: Callable[[Tensor], Tensor] | None = None,
        grad_tol: float = 1e-6,
        step_tol: float = 1e-8,
        loss_tol: float = 1e-10,
        loss_patience: int = 3,
    ) -> None:
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be between 0 and 1, got {beta}.")
        if alpha_init <= 0:
            raise ValueError(f"alpha_init must be positive, got {alpha_init}.")
        if max_iterations < 1 or max_line_search < 1 or loss_patience < 1:
            raise ValueError("Iteration limits and loss_patience must be positive.")
        if not 0.0 < armijo_c1 < 1.0:
            raise ValueError(f"armijo_c1 must be between 0 and 1, got {armijo_c1}.")
        if min(grad_tol, step_tol, loss_tol) < 0:
            raise ValueError("Convergence tolerances must be non-negative.")

        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else x0.device
        dtype = parameter.dtype if parameter is not None else x0.dtype
        x = torch.as_tensor(x0, device=device, dtype=dtype)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 1:
            raise ValueError(f"x0 must have shape [D] or [1, D], got {tuple(x.shape)}.")

        matrix = torch.as_tensor(A, device=device, dtype=dtype)
        if matrix.shape != (x.numel(), x.numel()):
            raise ValueError(
                f"A must have shape ({x.numel()}, {x.numel()}), "
                f"got {tuple(matrix.shape)}."
            )

        self.model = model
        self.A = matrix
        self.x0 = x.detach().clone()
        self.beta = beta
        self.alpha_init = alpha_init
        self.max_iterations = max_iterations
        self.max_line_search = max_line_search
        self.armijo_c1 = armijo_c1
        self.retraction = retraction
        self.grad_tol = grad_tol
        self.step_tol = step_tol
        self.loss_tol = loss_tol
        self.loss_patience = loss_patience

    def _objective(self, x: Tensor) -> Tensor:
        return torch.dot(x, self.A @ x)

    def _euclidean_gradient(self, x: Tensor) -> Tensor:
        return (self.A + self.A.transpose(-2, -1)) @ x

    def _operator(self, x: Tensor, h: float | Tensor) -> Tensor:
        h_tensor = torch.as_tensor(h, device=x.device, dtype=x.dtype)
        if h_tensor.numel() != 1:
            raise ValueError("h must be a scalar for pointwise optimization.")

        output = self.model(h_tensor.reshape(1), x.unsqueeze(0))
        dimension = x.numel()

        if getattr(self.model, "smart_training", True):
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[-1] != dimension:
                raise ValueError(
                    "Expected the smart-training model output to have shape "
                    f"[1, rank, {dimension}], got {tuple(output.shape)}."
                )
            operator = output.transpose(-2, -1) @ output
        else:
            if output.shape != (1, dimension, dimension):
                raise ValueError(
                    "Expected the model to return a full operator with shape "
                    f"[1, {dimension}, {dimension}], got {tuple(output.shape)}."
                )
            operator = output

        operator = operator[0]
        return 0.5 * (operator + operator.transpose(-2, -1))

    @torch.no_grad()
    def optimize(self, h: float | Tensor) -> OptimizationResult:
        """Run backtracking Riemannian gradient descent from the initial point."""
        x = self.x0.clone()
        previous_loss = self._objective(x)
        small_loss_changes = 0

        for iteration in range(1, self.max_iterations + 1):
            gradient = self._euclidean_gradient(x)
            operator = self._operator(x, h)
            direction = operator @ gradient
            direction_norm = torch.linalg.vector_norm(direction)

            if not torch.isfinite(direction_norm):
                return OptimizationResult(
                    x=x,
                    fun=self._objective(x),
                    iterations=iteration - 1,
                    converged=False,
                    status="non_finite_direction",
                )
            if direction_norm <= self.grad_tol:
                return OptimizationResult(
                    x=x,
                    fun=self._objective(x),
                    iterations=iteration - 1,
                    converged=True,
                    status="gradient_tolerance",
                )

            directional_derivative = torch.dot(gradient, direction)
            if (
                self.retraction is None
                and (
                    not torch.isfinite(directional_derivative)
                    or directional_derivative <= 0
                )
            ):
                return OptimizationResult(
                    x=x,
                    fun=self._objective(x),
                    iterations=iteration - 1,
                    converged=False,
                    status="not_a_descent_direction",
                )

            current_loss = self._objective(x)
            alpha = self.alpha_init
            accepted = False
            for _ in range(self.max_line_search):
                candidate = x - alpha * direction
                if self.retraction is not None:
                    candidate = self.retraction(candidate)
                candidate_loss = self._objective(candidate)
                predicted_decrease = torch.dot(gradient, x - candidate)
                armijo_bound = current_loss - self.armijo_c1 * predicted_decrease
                if (
                    torch.isfinite(candidate_loss)
                    and torch.isfinite(predicted_decrease)
                    and predicted_decrease > 0
                    and candidate_loss <= armijo_bound
                ):
                    accepted = True
                    break
                alpha *= self.beta

            if not accepted:
                return OptimizationResult(
                    x=x,
                    fun=current_loss,
                    iterations=iteration - 1,
                    converged=False,
                    status="line_search_failed",
                )

            step_norm = torch.linalg.vector_norm(candidate - x)
            relative_loss_change = (
                (current_loss - candidate_loss).abs()
                / current_loss.abs().clamp_min(torch.finfo(x.dtype).eps)
            )
            x = candidate

            if relative_loss_change <= self.loss_tol:
                small_loss_changes += 1
            else:
                small_loss_changes = 0

            if step_norm <= self.step_tol:
                return OptimizationResult(
                    x=x,
                    fun=candidate_loss,
                    iterations=iteration,
                    converged=True,
                    status="step_tolerance",
                )
            if small_loss_changes >= self.loss_patience:
                return OptimizationResult(
                    x=x,
                    fun=candidate_loss,
                    iterations=iteration,
                    converged=True,
                    status="loss_tolerance",
                )

            previous_loss = candidate_loss

        return OptimizationResult(
            x=x,
            fun=previous_loss,
            iterations=self.max_iterations,
            converged=False,
            status="max_iterations",
        )
