"""Depth-wise LoRA adapters ("shells") for recurrent-loop specialization.

Each loop iteration in the recurrent block can be viewed as a distinct
"shell" or agent.  Instead of sharing the same weights across all depths,
we attach a small LoRA pair per shell so that early shells and late shells
can evolve different inductive biases while the base weight stays frozen.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from plato_mythos.config import PlatoMythosConfig


class ShellLoRA(nn.Module):
    """Per-shell (per-iteration) LoRA adapter.

    Multiple shells share a common base projection, but each shell
    maintains unique low-rank A and B matrices.

    forward(x, shell_id) selects the appropriate A/B pair and returns
    the LoRA residual: alpha/r * B @ A @ x.
    """

    def __init__(self, config: PlatoMythosConfig, max_shells: Optional[int] = None):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.r = config.lora_r
        self.alpha = config.lora_alpha
        self.dropout = nn.Dropout(config.lora_dropout) if config.lora_dropout > 0 else None

        # If max_shells is not provided, use the curriculum max loop depth.
        self.max_shells = max_shells if max_shells is not None else config.max_loop_depth

        # One down-projection (d_model -> r) and one up-projection (r -> d_model) per shell.
        # We store them as stacked parameter tensors for vectorized indexing.
        self.lora_A = nn.Parameter(torch.zeros(self.max_shells, self.r, self.d_model))
        self.lora_B = nn.Parameter(torch.zeros(self.max_shells, self.d_model, self.r))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize A with Kaiming uniform and B with zeros (standard LoRA)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, shell_id: int) -> torch.Tensor:
        """Apply the shell-specific LoRA residual.

        Args:
            x: (batch, seq, d_model)
            shell_id: integer index in [0, max_shells).

        Returns:
            out: (batch, seq, d_model) LoRA residual.
        """
        if shell_id < 0 or shell_id >= self.max_shells:
            raise IndexError(
                f"shell_id {shell_id} out of range [0, {self.max_shells})"
            )

        # Matmul path: x -> A -> B -> scaled residual
        # x: (B, S, d)
        h = F.linear(x, self.lora_A[shell_id])  # (B, S, r)
        if self.dropout is not None:
            h = self.dropout(h)
        out = F.linear(h, self.lora_B[shell_id])  # (B, S, d)
        out = out * (self.alpha / self.r)
        return out

    def merge_into_base(self, base_weight: nn.Linear) -> nn.Linear:
        """For inference, merge all shell LoRAs into a new base weight.

        This returns a *new* Linear whose weight equals the base weight
        plus the averaged LoRA delta across all shells.  The original
        modules are left untouched.

        Args:
            base_weight: nn.Linear(d_model, d_model)

        Returns:
            merged: nn.Linear with combined weights.
        """
        device = base_weight.weight.device
        dtype = base_weight.weight.dtype

        # Average delta across shells: (d, d)
        delta = torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype)
        for i in range(self.max_shells):
            delta += (self.lora_B[i] @ self.lora_A[i]) * (self.alpha / self.r)
        delta /= self.max_shells

        merged = nn.Linear(self.d_model, self.d_model, bias=base_weight.bias is not None)
        merged.weight.data = base_weight.weight.data + delta
        if base_weight.bias is not None:
            merged.bias.data = base_weight.bias.data.clone()
        return merged


class ShellLoRALayer(nn.Module):
    """Wrapper that applies ShellLoRA around an existing nn.Linear.

    This is useful when you want the base projection to remain a
    standard PyTorch layer while the LoRA path is injected externally.
    """

    def __init__(self, base: nn.Linear, shell_lora: ShellLoRA):
        super().__init__()
        self.base = base
        self.shell_lora = shell_lora

    def forward(self, x: torch.Tensor, shell_id: int) -> torch.Tensor:
        """Base projection + shell-specific LoRA residual."""
        return self.base(x) + self.shell_lora(x, shell_id)
