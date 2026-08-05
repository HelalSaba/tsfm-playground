"""Synthetic autoregressive data generators for TSFM sanity experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ARBatch:
    """One batch of AR(p) one-step forecasting data.

    Coefficients are stored in lag order: coeffs[:, 0] multiplies lag-1,
    coeffs[:, 1] multiplies lag-2, and so on.

    For mixed-order batches, coeffs is padded to the maximum order and orders[b]
    gives the true order of sample b. Coefficients beyond orders[b] are zero.
    """

    context: torch.Tensor
    target: torch.Tensor
    conditional_mean: torch.Tensor
    coeffs: torch.Tensor
    pacf: torch.Tensor
    orders: torch.Tensor | None = None

    @property
    def ar_order(self) -> int:
        return int(self.coeffs.shape[1])

    @property
    def phi(self) -> torch.Tensor:
        """Backward-compatible AR(1) name.

        For AR(1), returns shape (batch,). For AR(p), returns coeffs with shape
        (batch, p).
        """
        if self.ar_order == 1:
            return self.coeffs[:, 0]
        return self.coeffs


# Backward-compatible alias used by older scripts.
AR1Batch = ARBatch


def pacf_to_ar_coeffs(pacf: torch.Tensor) -> torch.Tensor:
    """Convert PACF / reflection coefficients to stationary AR coefficients.

    Parameters
    ----------
    pacf:
        Tensor of shape (batch, p), with values strictly inside (-1, 1).

    Returns
    -------
    Tensor of shape (batch, p), in lag order: lag-1, lag-2, ..., lag-p.
    """
    if pacf.ndim != 2:
        raise ValueError("pacf must have shape (batch, p)")
    if torch.any(pacf <= -1.0) or torch.any(pacf >= 1.0):
        raise ValueError("All PACF values must lie strictly inside (-1, 1)")

    batch_size, p = pacf.shape
    phi = pacf.new_zeros(batch_size, p, p)
    phi[:, 0, 0] = pacf[:, 0]

    for m in range(1, p):
        prev = phi[:, m - 1, :m]
        kappa = pacf[:, m : m + 1]
        phi[:, m, :m] = prev - kappa * torch.flip(prev, dims=[1])
        phi[:, m, m] = pacf[:, m]

    return phi[:, p - 1, :]


def _generate_from_pacf(
    *,
    pacf: torch.Tensor,
    orders: torch.Tensor,
    context_length: int,
    noise_std: float,
    burn_in: int,
) -> ARBatch:
    batch_size, max_order = pacf.shape
    device = pacf.device
    dtype = pacf.dtype
    coeffs = pacf_to_ar_coeffs(pacf)

    total_length = burn_in + context_length + 1
    eps = torch.randn(batch_size, total_length, device=device, dtype=dtype) * noise_std
    x = torch.zeros(batch_size, total_length, device=device, dtype=dtype)

    # Vectorized over the batch, looped over time because AR recursion is sequential.
    for t in range(max_order, total_length):
        recent = torch.flip(x[:, t - max_order : t], dims=[1])  # lag-1, ..., lag-max_order
        x[:, t] = (coeffs * recent).sum(dim=1) + eps[:, t]

    series = x[:, burn_in:]
    context = series[:, :context_length]
    target = series[:, context_length]
    recent_context = torch.flip(context[:, -max_order:], dims=[1])
    conditional_mean = (coeffs * recent_context).sum(dim=1)

    return ARBatch(
        context=context,
        target=target,
        conditional_mean=conditional_mean,
        coeffs=coeffs,
        pacf=pacf,
        orders=orders,
    )


def generate_arp_batch(
    batch_size: int,
    context_length: int,
    *,
    ar_order: int = 1,
    pacf_low: float = -0.9,
    pacf_high: float = 0.9,
    noise_std: float = 1.0,
    burn_in: int = 64,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> ARBatch:
    """Generate an AR(p) batch with one stationary coefficient vector per series."""
    if ar_order < 1:
        raise ValueError("ar_order must be at least 1")
    if context_length <= ar_order:
        raise ValueError("context_length must be greater than ar_order")
    if not (-1.0 < pacf_low < pacf_high < 1.0):
        raise ValueError("Require -1 < pacf_low < pacf_high < 1")
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    if burn_in < ar_order:
        raise ValueError("burn_in must be at least ar_order")

    device = torch.device("cpu") if device is None else torch.device(device)
    pacf = torch.empty(batch_size, ar_order, device=device, dtype=dtype).uniform_(pacf_low, pacf_high)
    orders = torch.full((batch_size,), ar_order, device=device, dtype=torch.long)
    return _generate_from_pacf(
        pacf=pacf,
        orders=orders,
        context_length=context_length,
        noise_std=noise_std,
        burn_in=burn_in,
    )


def generate_mixed_arp_batch(
    batch_size: int,
    context_length: int,
    *,
    min_ar_order: int = 1,
    max_ar_order: int = 5,
    pacf_low: float = -0.9,
    pacf_high: float = 0.9,
    noise_std: float = 1.0,
    burn_in: int = 64,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> ARBatch:
    """Generate a batch mixing AR orders uniformly from min_ar_order..max_ar_order.

    Each sample has its own true order. PACF entries after that order are set to
    zero, so the resulting AR coefficients beyond that order are exactly zero.
    """
    if min_ar_order < 1:
        raise ValueError("min_ar_order must be at least 1")
    if max_ar_order < min_ar_order:
        raise ValueError("max_ar_order must be >= min_ar_order")
    if context_length <= max_ar_order:
        raise ValueError("context_length must be greater than max_ar_order")
    if not (-1.0 < pacf_low < pacf_high < 1.0):
        raise ValueError("Require -1 < pacf_low < pacf_high < 1")
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    if burn_in < max_ar_order:
        raise ValueError("burn_in must be at least max_ar_order")

    device = torch.device("cpu") if device is None else torch.device(device)
    orders = torch.randint(
        low=min_ar_order,
        high=max_ar_order + 1,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )
    pacf = torch.zeros(batch_size, max_ar_order, device=device, dtype=dtype)
    active = torch.arange(max_ar_order, device=device).unsqueeze(0) < orders.unsqueeze(1)
    pacf[active] = torch.empty(int(active.sum().item()), device=device, dtype=dtype).uniform_(pacf_low, pacf_high)

    return _generate_from_pacf(
        pacf=pacf,
        orders=orders,
        context_length=context_length,
        noise_std=noise_std,
        burn_in=burn_in,
    )


def generate_ar1_batch(
    batch_size: int,
    context_length: int,
    *,
    phi_low: float = -0.9,
    phi_high: float = 0.9,
    noise_std: float = 1.0,
    burn_in: int = 64,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> ARBatch:
    """Backward-compatible AR(1) wrapper around ``generate_arp_batch``."""
    return generate_arp_batch(
        batch_size=batch_size,
        context_length=context_length,
        ar_order=1,
        pacf_low=phi_low,
        pacf_high=phi_high,
        noise_std=noise_std,
        burn_in=burn_in,
        device=device,
        dtype=dtype,
    )


def arp_ols_forecast(context: torch.Tensor, ar_order: int, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate AR(p) coefficients per series by ridge-stabilized least squares."""
    if context.ndim != 2:
        raise ValueError("context must have shape (batch, context_length)")
    if ar_order < 1:
        raise ValueError("ar_order must be at least 1")
    if context.shape[1] <= ar_order:
        raise ValueError("context length must be greater than ar_order")

    windows = context.unfold(dimension=1, size=ar_order + 1, step=1)
    x_lagged = torch.flip(windows[:, :, :ar_order], dims=[2])  # lag-1, ..., lag-p
    y = windows[:, :, ar_order]

    xt = x_lagged.transpose(1, 2)
    eye = torch.eye(ar_order, device=context.device, dtype=context.dtype).unsqueeze(0)
    xtx = torch.matmul(xt, x_lagged) + eps * eye
    xty = torch.matmul(xt, y.unsqueeze(-1))
    coeff_hat = torch.linalg.solve(xtx, xty).squeeze(-1)

    recent = torch.flip(context[:, -ar_order:], dims=[1])
    forecast = (coeff_hat * recent).sum(dim=1)
    return forecast, coeff_hat


def arp_ols_forecast_by_order(
    context: torch.Tensor,
    orders: torch.Tensor,
    *,
    max_ar_order: int | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OLS baseline for a batch whose samples may have different AR orders."""
    if orders.ndim != 1:
        raise ValueError("orders must have shape (batch,)")
    if len(orders) != context.shape[0]:
        raise ValueError("orders length must match batch size")

    max_order = int(max_ar_order or orders.max().item())
    forecast = context.new_empty(context.shape[0])
    coeff_hat_padded = context.new_zeros(context.shape[0], max_order)

    for order in sorted(int(o) for o in orders.unique().detach().cpu().tolist()):
        idx = orders == order
        f, c = arp_ols_forecast(context[idx], ar_order=order, eps=eps)
        forecast[idx] = f
        coeff_hat_padded[idx, :order] = c

    return forecast, coeff_hat_padded


def ar1_ols_forecast(context: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible AR(1) OLS baseline."""
    forecast, coeff_hat = arp_ols_forecast(context, ar_order=1, eps=eps)
    return forecast, coeff_hat[:, 0]


def ar_batch_to_uvmodel(batch: ARBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ARBatch to UVModel's expected univariate shapes."""
    context = batch.context.unsqueeze(-1)  # (B, T, 1)
    true_horizon = batch.target[:, None, None]  # (B, 1, 1)
    conditional_mean = batch.conditional_mean[:, None, None]  # (B, 1, 1)
    return context, true_horizon, conditional_mean


def ar1_batch_to_uvmodel(batch: ARBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible alias for AR(1) scripts."""
    return ar_batch_to_uvmodel(batch)
