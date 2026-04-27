"""Compress PLATO tiles into MLA latent KV representations.

A PLATO tile is a structured unit carrying tokens, metadata, and room tags.
Rather than treating tiles as raw token sequences, we:
1. Embed tile tokens.
2. Project into a low-dimensional latent KV space (Multi-Head Latent Attention).
3. Cache the compressed representation for recurrent depth loops.

This mirrors the OpenMythos "tile-as-memory" philosophy while remaining
compatible with standard transformer KV caches.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from plato_mythos.config import PlatoMythosConfig


class TileCompressor(nn.Module):
    """Compress a PLATO tile into a fixed-size latent vector."""

    def __init__(self, config: PlatoMythosConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_kv_latent = config.d_kv_latent

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.proj_down = nn.Linear(config.d_model, config.d_kv_latent, bias=False)
        self.proj_up = nn.Linear(config.d_kv_latent, config.d_model, bias=False)

        # Tile-level metadata encoder (domain ID + confidence as scalar).
        self.meta_encoder = nn.Linear(2, config.d_model, bias=False)

    def embed_tile(
        self,
        token_ids: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed tile tokens and optionally blend metadata.

        Args:
            token_ids: (batch, seq) integer token indices.
            domain_id: (batch,) optional domain indices.
            confidence: (batch,) optional confidence scalars.

        Returns:
            h: (batch, seq, d_model)
        """
        h = self.token_embed(token_ids)  # (B, S, D)

        if domain_id is not None and confidence is not None:
            meta = torch.stack([domain_id.float(), confidence], dim=-1)  # (B, 2)
            meta_bias = self.meta_encoder(meta).unsqueeze(1)  # (B, 1, D)
            h = h + meta_bias

        return h

    def compress(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Project hidden states down to latent KV space.

        Args:
            hidden: (batch, seq, d_model)

        Returns:
            latent: (batch, seq, d_kv_latent)
        """
        return self.proj_down(hidden)

    def decompress(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Project latent KV back up to model dimension.

        Args:
            latent: (batch, seq, d_kv_latent)

        Returns:
            hidden: (batch, seq, d_model)
        """
        return self.proj_up(latent)


class MLAKVCache(nn.Module):
    """Multi-Head Latent Attention KV cache using tile compression.

    Instead of storing full key/value tensors per head, we store a single
    latent tensor per token and derive K/V via independent up-projections.
    """

    def __init__(self, config: PlatoMythosConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.d_kv_latent = config.d_kv_latent

        # Up-projection to keys and values (shared across heads via reshaping).
        self.k_up = nn.Linear(config.d_kv_latent, config.num_kv_heads * config.head_dim, bias=False)
        self.v_up = nn.Linear(config.d_kv_latent, config.num_kv_heads * config.head_dim, bias=False)

    def from_latent(
        self,
        latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand cached latent into multi-head K and V.

        Args:
            latent: (batch, seq, d_kv_latent)

        Returns:
            k: (batch, num_kv_heads, seq, head_dim)
            v: (batch, num_kv_heads, seq, head_dim)
        """
        batch, seq, _ = latent.shape
        k = self.k_up(latent)
        v = self.v_up(latent)
        k = k.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return k, v

    def precompute_rope(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sin/cos tensors for RoPE (simplified)."""
        dim = self.head_dim
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
        )
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary positional embeddings to x.

        Args:
            x: (batch, heads, seq, head_dim)
            cos: (seq, head_dim)
            sin: (seq, head_dim)

        Returns:
            rotated: same shape as x
        """
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + rotated * sin


class TilesAsKV(nn.Module):
    """High-level module: tile in -> latent KV -> standard attention interface out."""

    def __init__(self, config: PlatoMythosConfig):
        super().__init__()
        self.compressor = TileCompressor(config)
        self.mla_cache = MLAKVCache(config)

    def forward(
        self,
        token_ids: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress tiles and prepare K/V for attention.

        Returns:
            hidden: (batch, seq, d_model) token embeddings
            k: (batch, num_kv_heads, seq, head_dim)
            v: (batch, num_kv_heads, seq, head_dim)
        """
        hidden = self.compressor.embed_tile(token_ids, domain_id, confidence)
        latent = self.compressor.compress(hidden)
        k, v = self.mla_cache.from_latent(latent)
        return hidden, k, v
