"""Maps PLATO curriculum stages to Recurrent-Depth Transformer loop depth.

In OpenMythos, a curriculum is a sequence of pedagogical stages that
progressively increase complexity.  In the RDT, we repurpose this as a
variable-depth unroll: easy tokens get fewer loop iterations, hard tokens
get more.  The curriculum stage (an integer or float progress score)
determines the per-token loop budget.
"""

from typing import Optional

import torch
import torch.nn as nn

from plato_mythos.config import PlatoMythosConfig


class CurriculumScheduler(nn.Module):
    """Learns to map a curriculum progress score to an adaptive loop depth.

    The mapping is differentiable via a soft depth predictor, but at inference
    we also support hard discrete depths for deterministic behaviour.
    """

    def __init__(self, config: PlatoMythosConfig):
        super().__init__()
        self.config = config
        self.max_depth = config.max_loop_depth
        self.prelude_dims = config.d_model

        # Small MLP predicting a continuous depth factor in [0, 1].
        self.depth_mlp = nn.Sequential(
            nn.Linear(self.prelude_dims, self.prelude_dims // 4),
            nn.SiLU(),
            nn.Linear(self.prelude_dims // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prelude_state: torch.Tensor,
        curriculum_stage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict a soft loop-depth budget per token.

        Args:
            prelude_state: (batch, seq, d_model) from the Prelude block.
            curriculum_stage: (batch, seq) optional stage in [0, 1].
                              0 = easiest, 1 = hardest.

        Returns:
            depth_factor: (batch, seq) in [0, 1].  Multiply by max_depth
                          to get expected loop iterations.
        """
        factor = self.depth_mlp(prelude_state).squeeze(-1)  # (B, S)

        if curriculum_stage is not None:
            # Blend learned factor with explicit curriculum guidance.
            # Curriculum stage acts as a prior; model can deviate slightly.
            blend = 0.7 * curriculum_stage + 0.3 * factor
            factor = blend

        return factor

    def to_hard_depth(
        self,
        depth_factor: torch.Tensor,
        min_steps: int = 1,
    ) -> torch.Tensor:
        """Convert soft depth factors to integer loop counts.

        Args:
            depth_factor: (batch, seq) in [0, 1]
            min_steps: minimum loop iterations

        Returns:
            depths: (batch, seq) LongTensor
        """
        depths = (depth_factor * self.max_depth).long()
        depths = depths.clamp(min=min_steps, max=self.max_depth)
        return depths


class RecurrentBlock(nn.Module):
    """Single recurrent block executed multiple times per token position.

    The block is intentionally weight-shared across loop iterations so that
depth acts as computation time, not capacity expansion.
    """

    def __init__(self, config: PlatoMythosConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.norm1 = nn.RMSNorm(d) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d)
        self.norm2 = nn.RMSNorm(d) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d)

        # Simplified grouped-query self-attention.
        self.q_proj = nn.Linear(d, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(d, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(d, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, d, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One recurrent step.

        Args:
            x: (batch, seq, d_model)
            k_cache: optional cached keys to attend to
            v_cache: optional cached values to attend to

        Returns:
            out: (batch, seq, d_model)
        """
        batch, seq, d = x.shape

        # Pre-norm attention.
        h = self.norm1(x)
        q = self.q_proj(h).view(batch, seq, self.config.num_heads, self.config.head_dim)
        k = self.k_proj(h).view(batch, seq, self.config.num_kv_heads, self.config.head_dim)
        v = self.v_proj(h).view(batch, seq, self.config.num_kv_heads, self.config.head_dim)

        if k_cache is not None and v_cache is not None:
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        # GQA: repeat KV heads to match Q heads.
        num_rep = self.config.num_heads // self.config.num_kv_heads
        k = k.repeat_interleave(num_rep, dim=2)
        v = v.repeat_interleave(num_rep, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        out = self.o_proj(out)
        x = x + out

        # FFN.
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
