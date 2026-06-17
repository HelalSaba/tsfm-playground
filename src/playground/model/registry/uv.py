import torch
import torch.nn as nn

from einops import rearrange

from ..layers.base import Residual
from ..layers.normalization import InstanceNorm
from ..layers.patching import Patch
from ..layers.transformer import TransformerEncoderLayer

from ...training.loss import compute_quantile_loss


class UVModel(nn.Module):
    """
    Very simple univariate adaption of Chronos-2 model.
    https://github.com/amazon-science/chronos-forecasting/tree/main/src/chronos/chronos2
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_kv: int,
        n_heads: int,
        dropout: float,
        activation_fn: str,
        n_quantiles: int,
        n_encoder_layers: int,
        pred_length: int,
        use_arcsinh: bool,
        use_rope: bool,
        context_length: int,
        patch_size: int,
        patch_stride: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.d_kv = d_kv
        self.n_heads = n_heads
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.n_encoder_layers = n_encoder_layers

        self.n_quantiles = n_quantiles
        self.context_length = context_length
        self.pred_length = pred_length
        self.use_arcsinh = use_arcsinh
        self.use_rope = use_rope

        self.instance_norm = InstanceNorm(self.use_arcsinh)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch = Patch(patch_size=patch_size, patch_stride=patch_stride)

        self.input_patch_encoder = Residual(
            in_dim=self.patch_size,
            h_dim=self.d_ff,
            out_dim=self.d_model,
            dropout_p=self.dropout,
        )

        self.encoder = nn.ModuleList(
            TransformerEncoderLayer(
                n_heads=self.n_heads,
                d_model=self.d_model,
                d_kv=self.d_kv,
                mlp_hidden_dim=self.d_ff,
                dropout=self.dropout,
                activation=self.activation_fn,
                use_rope=self.use_rope,
                attn_dropout=self.dropout,
            )
            for _ in range(self.n_encoder_layers)
        )

        self.decoder = Residual(
            in_dim=self.d_model,
            h_dim=self.d_ff,
            out_dim=self.n_quantiles * self.patch_size,
            dropout_p=self.dropout,
        )

    def forward(
        self,
        context: torch.Tensor,
        n_horizon: int,
        true_horizon: torch.Tensor | None = None,
    ):
        """
        Take as input
        - context: (batch_size, series_length, num_series)
        - true_horizon: (batch_size, horizon_length, num_series) or None
        """

        batch_size, series_length, num_series = context.shape
        dtype = context.dtype

        if series_length > self.context_length:
            context = context[:, -self.context_length :, :]

        # NOTE: right now we are actually not doing ICL
        context = rearrange(context, "b l c -> (b c) l")

        # for now we init horizion as NaNs, which we then transform to 0 so essentially we init the forecast horizon as 0s.
        # TODO: we might want to try something different here examples are:
        # 1. Init with some learnable parameters
        # 2. Init with (seasonal) naive forecast - repeat last value in context
        # 3. Mean (or exponential smoothing)
        # 4. ...

        horizon = torch.full(
            (batch_size, n_horizon, num_series), float("nan"), device=context.device, dtype=context.dtype
        )
        horizon = rearrange(horizon, "b l c -> (b c) l")
        true_horizon = rearrange(true_horizon, "b l c -> (b c) l") if true_horizon is not None else None

        # normalization always over sequence dim
        context, loc_scale = self.instance_norm(context)

        # normalization is performed at fp32 precision, cast back to original dtype
        # in case original dtype is lower precision (e.g. bf16 or fp16)
        context = context.to(dtype)

        # patching
        patched_ctx = self.patch(context)  # b np ps
        patched_ctx = torch.nan_to_num(patched_ctx, nan=0.0)
        ctx_emb = self.input_patch_encoder(patched_ctx)  # b np d

        patched_horizon = self.patch(horizon)  # b np ps
        patched_horizon = torch.nan_to_num(patched_horizon, nan=0.0)
        horizon_emb = self.input_patch_encoder(patched_horizon)  # b np d

        n_output_patches = horizon_emb.shape[1]

        input_emb = torch.cat([ctx_emb, horizon_emb], dim=1)

        # allow full self attn also into the future
        attn_mask = torch.ones(
            (input_emb.shape[0], input_emb.shape[1], input_emb.shape[1]), device=context.device
        )  # b s s

        # we need position ids for RoPE
        position_ids = torch.arange(input_emb.shape[1], device=context.device).unsqueeze(0)  # 1 s

        for layer in self.encoder:
            input_emb = layer(input_emb, attn_mask, position_ids)

        forecast_emb = input_emb[:, -n_output_patches:, :]
        quantile_forecast = self.decoder(forecast_emb)
        quantile_forecast = rearrange(
            quantile_forecast,
            "b n (q p) -> b q (n p)",
            q=self.n_quantiles,
            n=n_output_patches,
            p=self.patch_size,
        )

        true_horizon_steps = true_horizon.shape[1]
        loss = compute_quantile_loss(quantile_forecast[:, :, :true_horizon_steps], true_horizon)

        # unscale
        quantile_forecast = rearrange(
            quantile_forecast, "b q h -> b (q h)", q=self.n_quantiles, h=n_output_patches * self.patch_size
        )

        quantile_forecast = self.instance_norm.inverse(quantile_forecast, loc_scale)
        quantile_forecast = rearrange(
            quantile_forecast, "b (q h) -> b q h", q=self.n_quantiles, h=n_output_patches * self.patch_size
        )

        return loss, quantile_forecast[:, :, :n_horizon]
