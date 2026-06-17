import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import MLP
from .positional import RoPE, validate_rope


class MHA(nn.Module):
    """
    Simple and flexbile MHA implementation (v2, uses RoPE with n_prefix logic).
    Implementation adapted from:
    https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/layers.py
    """

    def __init__(self, n_heads, d_model, d_kv, dropout, use_rope, attn_dropout=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.d_mha = d_kv * n_heads

        self.q = nn.Linear(self.d_model, self.d_mha, bias=False)
        self.k = nn.Linear(self.d_model, self.d_mha, bias=False)
        self.v = nn.Linear(self.d_model, self.d_mha, bias=False)
        self.o = nn.Linear(self.d_mha, d_model, bias=False)

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE(dim=d_kv)
        self.dropout = attn_dropout if attn_dropout is not None else dropout

    def _attention(self, q, k, v, mask=None):
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=1.0,  # no scaling
        )
        return attn_output

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_mha] -> [batch_size, n_heads, seq_len, d_kv]
        batch_size, _, _ = x.shape
        return rearrange(x, "b s (h d) -> b h s d", b=batch_size, h=self.n_heads, d=self.d_kv)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, n_heads, seq_len, d_kv] -> [batch_size, seq_len, d_mha]
        batch_size, _, _, _ = x.shape
        return rearrange(x, "b h s d -> b s (h d)", b=batch_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run MHA. Allow for attention mask as well as selective rope mask.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model).

        attention_mask: torch.Tensor
            Attention mask of shape (batch_size, sequence_length, sequence_length).
        """
        if self.use_rope:
            validate_rope(position_ids)

        Q = self._split_heads(self.q(x))  # b n_heads s d_kv
        K = self._split_heads(self.k(x))
        V = self._split_heads(self.v(x))

        # broadcast mask over heads
        attention_mask = attention_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # b n_heads s s
        if self.use_rope:
            cos, sin = self.rope(x, position_ids)

            # position_ids covers patch tokens only; infer number of prefix (CLS) tokens
            # from the difference between sequence length and position_ids length.
            # Apply RoPE only to patch tokens; leave prefix tokens unrotated.
            n_prefix = Q.shape[2] - position_ids.shape[1]
            Q_prefix, Q_patches = Q[:, :, :n_prefix], Q[:, :, n_prefix:]
            K_prefix, K_patches = K[:, :, :n_prefix], K[:, :, n_prefix:]
            Q_patches, K_patches = RoPE.apply_rotary_pos_emb(Q_patches, K_patches, cos, sin, unsqueeze_dim=1)
            Q = torch.cat([Q_prefix, Q_patches], dim=2)
            K = torch.cat([K_prefix, K_patches], dim=2)

        attn_output = self._attention(Q, K, V, mask=attention_mask)  # b n_heads s d_kv
        attn_output = self._combine_heads(attn_output)  # b s d_mha
        output = self.o(attn_output)  # b s d_model
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_kv: int,
        mlp_hidden_dim: int,
        dropout: float,
        activation: str,
        use_rope: bool,
        attn_dropout: float = None,
    ):
        super().__init__()
        self.mha = MHA(n_heads, d_model, d_kv, dropout, use_rope, attn_dropout=attn_dropout)
        self.mha_layernorm = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_hidden_dim, activation, dropout)
        self.mlp_layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Multi-head attention
        mha_output = self.mha(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        x = x + self.dropout(mha_output)
        x = self.mha_layernorm(x)

        # MLP
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.mlp_layernorm(x)

        return x
