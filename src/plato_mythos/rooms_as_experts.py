"""MoE routing where PLATO room tags replace learned gating.

Each PLATO tile carries a domain and confidence.  Instead of a standard
learned top-k gate, we:
1. Restrict the candidate expert set to the room matching the domain.
2. Use confidence to bias the soft assignment within that room.
3. Fall back to hash-based routing for unknown domains.

This makes every routing decision human-interpretable: you can read the
room tag and know exactly which expert group handled the token.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from plato_mythos.config import PlatoMythosConfig


class RoomExpert(nn.Module):
    """Single Feed-Forward expert inside a room."""

    def __init__(self, config: PlatoMythosConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.w2 = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, 4 * config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RoomRouter(nn.Module):
    """Interpretable router: domain -> room, confidence -> intra-room weights."""

    def __init__(self, config: PlatoMythosConfig):
        super().__init__()
        self.config = config
        self.num_rooms = config.num_rooms
        self.experts_per_room = config.experts_per_room
        self.top_k = config.top_k_experts

        # Learnable intra-room preference vector per room.
        # Shape: (num_rooms, experts_per_room)
        self.intra_room_bias = nn.Parameter(
            torch.zeros(self.num_rooms, self.experts_per_room)
        )

        # Confidence scaling: learned temperature per room.
        self.confidence_temp = nn.Parameter(torch.ones(self.num_rooms))

    def _domain_to_room(self, domain: str) -> int:
        return self.config.room_for_domain(domain)

    def _confidence_weights(
        self,
        room_idx: int,
        confidence: float,
    ) -> torch.Tensor:
        """Return softmax weights over experts in the chosen room."""
        bias = self.intra_room_bias[room_idx]
        temp = F.softplus(self.confidence_temp[room_idx]) + 1e-3
        # Confidence is injected as a location shift.
        shifted = bias + (confidence - 0.5) / temp
        return F.softmax(shifted, dim=-1)

    def route(
        self,
        domain: str,
        confidence: float,
    ) -> Tuple[List[int], torch.Tensor]:
        """Return (expert_indices, expert_weights) for a single token's tags.

        The returned indices are global expert IDs:
            global_id = room_idx * experts_per_room + local_id
        """
        room_idx = self._domain_to_room(domain)
        base = room_idx * self.experts_per_room
        local_probs = self._confidence_weights(room_idx, confidence)

        # Top-k within the room (never spills to other rooms).
        k = min(self.top_k, self.experts_per_room)
        top_local_vals, top_local_idx = torch.topk(local_probs, k)
        top_local_vals = top_local_vals / (top_local_vals.sum() + 1e-9)

        global_indices = [(base + int(i)) for i in top_local_idx.tolist()]
        return global_indices, top_local_vals

    def forward(
        self,
        x: torch.Tensor,
        domains: List[str],
        confidences: List[float],
    ) -> torch.Tensor:
        """Route each token in x through its domain-determined experts.

        Args:
            x: (batch, seq, d_model)
            domains: length seq list of domain strings
            confidences: length seq list of confidence floats

        Returns:
            out: (batch, seq, d_model)
        """
        batch, seq, d_model = x.shape
        out = torch.zeros_like(x)

        for b in range(batch):
            for s in range(seq):
                token = x[b, s]
                expert_ids, weights = self.route(domains[s], confidences[s])
                for eid, w in zip(expert_ids, weights):
                    expert = self.experts[eid]
                    out[b, s] += w * expert(token)
        return out


class RoomsAsExperts(nn.Module):
    """Full MoE layer backed by PLATO room semantics."""

    def __init__(self, config: PlatoMythosConfig):
        super().__init__()
        self.config = config
        self.router = RoomRouter(config)
        self.experts = nn.ModuleList(
            [RoomExpert(config) for _ in range(config.total_experts)]
        )
        # Share experts list with router so router.forward can access them.
        self.router.experts = self.experts

    def forward(
        self,
        x: torch.Tensor,
        domains: List[str],
        confidences: List[float],
    ) -> torch.Tensor:
        return self.router(x, domains, confidences)
