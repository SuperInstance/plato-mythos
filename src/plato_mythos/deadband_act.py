"""Adaptive Compute Time with PLATO deadband thresholds.

In PLATO, deadbands define priority tiers for control loops:
    P0 = 0.99  (critical — halt only when virtually certain)
    P1 = 0.8   (standard — normal confidence required)
    P2 = 0.5   (low — eager halting acceptable)

We repurpose these tiers as adaptive-compute halting thresholds.
The higher the priority, the longer the model thinks before emitting
a token.  This maps directly onto recurrent-depth loop unrolling.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class DeadbandACT(nn.Module):
    """Adaptive halting controller using PLATO-style deadband thresholds.

    At each recurrent step we compute a halting probability from the
    L2-norm delta between consecutive hidden states.  The cumulative
    probability is compared against a priority-dependent threshold to
    decide whether to continue thinking.
    """

    # PLATO deadband thresholds (priority tiers)
    P0_CRITICAL: float = 0.99
    P1_STANDARD: float = 0.8
    P2_LOW: float = 0.5

    def __init__(self, threshold: float = 1e-3, max_steps: int = 12):
        super().__init__()
        self.threshold = threshold
        self.max_steps = max_steps
        self._prev_state: Optional[torch.Tensor] = None
        self._cum_prob: float = 0.0

    def reset(self, state: Optional[torch.Tensor] = None):
        """Reset cumulative probability and stored previous state."""
        self._prev_state = state
        self._cum_prob = 0.0

    def halt_probability(self, state: torch.Tensor) -> float:
        """Compute per-step halt probability from state delta.

        Uses a sigmoid of the negative mean L2 norm delta so that
        small deltas (convergence) produce high halt probabilities.

        Args:
            state: (batch, seq, d_model) current hidden state.

        Returns:
            prob: scalar halt probability in [0, 1].
        """
        if self._prev_state is None:
            self._prev_state = state.detach().clone()
            return 0.0

        delta = (state - self._prev_state).detach()
        l2_norm = delta.pow(2).sum(dim=-1).sqrt().mean().item()
        # Sigmoid so that tiny deltas -> prob ~ 1, large deltas -> prob ~ 0
        prob = torch.sigmoid(torch.tensor(-l2_norm / (self.threshold + 1e-9))).item()
        self._prev_state = state.detach().clone()
        return prob

    def should_continue(self, state: torch.Tensor, step: int) -> bool:
        """Return True if the recurrent loop should keep running.

        Args:
            state: current hidden state.
            step: zero-based loop index.

        Returns:
            bool: True when cumulative halt prob is below threshold
                  and step is within max_steps.
        """
        if step >= self.max_steps - 1:
            return False

        prob = self.halt_probability(state)
        self._cum_prob += prob

        # Default to standard tier if no explicit priority is set.
        tier = getattr(self, "_current_priority", "P1")
        tier_threshold = {
            "P0": self.P0_CRITICAL,
            "P1": self.P1_STANDARD,
            "P2": self.P2_LOW,
        }.get(tier, self.P1_STANDARD)

        return self._cum_prob < tier_threshold

    def get_priority(self, token_metadata: Optional[Dict]) -> str:
        """Classify token priority as P0, P1, or P2.

        Args:
            token_metadata: dict with optional keys:
                "confidence" (float), "critical" (bool).

        Returns:
            priority string: "P0", "P1", or "P2".
        """
        if token_metadata is None:
            return "P1"

        if token_metadata.get("critical", False):
            return "P0"

        confidence = token_metadata.get("confidence", 0.5)
        if confidence >= self.P0_CRITICAL:
            return "P0"
        if confidence >= self.P1_STANDARD:
            return "P1"
        return "P2"

    def set_priority(self, priority: str):
        """Store the active priority tier for the next should_continue call."""
        self._current_priority = priority
