import torch


def compute_quantile_loss(
    quantile_preds: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the quantile loss (pinball loss) for a batch of predictions and targets.

    Parameters
    ----------
    quantile_preds: torch.Tensor
        Predicted quantiles of shape (batch_size, n_quantiles, pred_length).

    target: torch.Tensor
        Ground truth values of shape (batch_size, pred_length).

    Returns
    -------
    torch.Tensor
        The computed quantile loss.
    """
    # Ensure target has the same shape as quantile_preds for broadcasting
    target = target.unsqueeze(1)  # (batch_size, 1, pred_length)

    # Calculate the error
    errors = target - quantile_preds  # (batch_size, n_quantiles, pred_length)

    # Standard 0.1, 0.2, ..., 0.9 quantiles
    quantiles = (
        torch.arange(0.1, 1.0, step=0.1, device=quantile_preds.device).unsqueeze(0).unsqueeze(-1)
    )  # (1, n_quantiles, 1)

    # Compute the pinball loss
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)  # (batch_size, n_quantiles, pred_length)

    return loss.mean()
