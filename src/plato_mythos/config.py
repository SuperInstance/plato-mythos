"""Configuration for PlatoMythos.

Maps MythosConfig hyperparameters onto PLATO-specific concepts:
rooms become expert groups, curriculum stages become loop depths,
tiles become compressed latent KV pairs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PlatoMythosConfig:
    """Full configuration for the PlatoMythos RDT model.

    Attributes:
        vocab_size: Token vocabulary size.
        d_model: Hidden dimension for the transformer backbone.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (GQA). Defaults to num_heads.
        num_layers: Number of stacked recurrent blocks.
        max_loop_depth: Maximum curriculum/loop rounds per token position.
        d_kv_latent: Dimension of the MLA latent KV compression.
        num_rooms: Number of PLATO rooms (expert groups).
        experts_per_room: Experts inside each room.
        top_k_experts: How many experts to activate per token.
        deadband_threshold: ACT halts when state delta L2 norm falls below this.
        deadband_max_steps: Hard cap on ACT loop iterations.
        lora_r: LoRA rank for depth-wise shell adapters.
        lora_alpha: LoRA scaling alpha.
        lora_dropout: Dropout on LoRA path.
        room_tags: Optional mapping from room index to (domain, confidence_range).
        prelude_layers: Static layers before the recurrent loop.
        coda_layers: Static layers after the recurrent loop.
        tie_weights: Whether to tie input/output embeddings.
        dropout: Global dropout rate.
        dtype: Torch dtype string ("float32", "bfloat16", etc.).
        device: Target device string.
    """

    vocab_size: int = 32000
    d_model: int = 1024
    num_heads: int = 16
    num_kv_heads: Optional[int] = None
    num_layers: int = 4
    max_loop_depth: int = 8
    d_kv_latent: int = 256
    num_rooms: int = 8
    experts_per_room: int = 2
    top_k_experts: int = 2
    deadband_threshold: float = 1e-3
    deadband_max_steps: int = 12
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    room_tags: Optional[Dict[int, Tuple[str, Tuple[float, float]]]] = None
    prelude_layers: int = 2
    coda_layers: int = 2
    tie_weights: bool = False
    dropout: float = 0.0
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.room_tags is None:
            self.room_tags = {
                i: (f"room_{i}", (0.0, 1.0))
                for i in range(self.num_rooms)
            }
        if self.top_k_experts > self.num_rooms * self.experts_per_room:
            raise ValueError(
                "top_k_experts cannot exceed total number of experts"
            )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.num_heads

    @property
    def total_experts(self) -> int:
        """Total MoE expert count across all rooms."""
        return self.num_rooms * self.experts_per_room

    def room_for_domain(self, domain: str) -> int:
        """Return the room index assigned to a given domain string."""
        for idx, (dom, _) in self.room_tags.items():
            if dom == domain:
                return idx
        # Hash-based fallback for unknown domains.
        return hash(domain) % self.num_rooms

    def confidence_gate(self, confidence: float) -> List[int]:
        """Return candidate room indices sorted by confidence range proximity."""
        scored = []
        for idx, (_, (low, high)) in self.room_tags.items():
            center = (low + high) / 2.0
            width = (high - low) / 2.0 + 1e-6
            dist = abs(confidence - center) / width
            scored.append((dist, idx))
        scored.sort()
        return [idx for _, idx in scored]
