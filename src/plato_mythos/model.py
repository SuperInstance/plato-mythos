"""PlatoMythos: PLATO-native Recurrent-Depth Transformer.

This model ties together the four PLATO concepts:
    * Tiles  -> compressed latent KV pairs (TilesAsKV)
    * Rooms  -> interpretable MoE routing (RoomsAsExperts)
    * Curriculum -> adaptive loop depth (CurriculumScheduler + DeadbandACT)
    * Shells -> depth-wise LoRA adapters (ShellLoRA)

The forward path is:
    embed -> prelude -> recurrent loop (ACT + shell LoRA) -> coda -> output
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from plato_mythos.config import PlatoMythosConfig
from plato_mythos.curriculum_loop import RecurrentBlock
from plato_mythos.deadband_act import DeadbandACT
from plato_mythos.rooms_as_experts import RoomsAsExperts
from plato_mythos.shell_lora import ShellLoRA
from plato_mythos.tiles_as_kv import TilesAsKV


class PlatoMythos(nn.Module):
    """Full PLATO-native recurrent-depth transformer model."""

    def __init__(self, config: PlatoMythosConfig):
        super().__init__()
        self.config = config

        # 1. Tile compression / embedding.
        self.tiles_as_kv = TilesAsKV(config)

        # 2. Prelude: static transformer layers before the recurrent loop.
        self.prelude = nn.ModuleList(
            [RecurrentBlock(config) for _ in range(config.prelude_layers)]
        )

        # 3. Recurrent block (weight-shared across loop iterations).
        self.recurrent_block = RecurrentBlock(config)

        # 4. Shell LoRA: one low-rank adapter per loop depth.
        self.shell_lora = ShellLoRA(config)

        # 5. Deadband ACT: priority-aware adaptive halting.
        self.deadband = DeadbandACT(
            threshold=config.deadband_threshold,
            max_steps=config.deadband_max_steps,
        )

        # 6. Coda: static transformer layers after the recurrent loop.
        self.coda = nn.ModuleList(
            [RecurrentBlock(config) for _ in range(config.coda_layers)]
        )

        # 7. Output head (maps d_model -> vocab logits).
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Optional weight tying between input embedding and output head.
        if config.tie_weights:
            self.output_head.weight = self.tiles_as_kv.compressor.token_embed.weight

        # 8. Room-based MoE (applied after coda for interpretable mixing).
        self.rooms_as_experts = RoomsAsExperts(config)

    def forward(
        self,
        tiles: Union[torch.Tensor, Dict[str, Any]],
        rooms: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a full forward pass.

        Args:
            tiles: Either a (batch, seq) LongTensor of token IDs, or a dict
                   with keys "token_ids", optionally "domain_id" and
                   "confidence".
            rooms: Optional dict with "domains" and "confidences" lists,
                   or raw metadata for the deadband priority gate.

        Returns:
            If tiles is a Tensor, returns (batch, seq, vocab_size) logits.
            If tiles is a dict, returns a dict with the same keys where
            "token_ids" is replaced by logits.
        """
        # Unpack tiles.
        if isinstance(tiles, dict):
            token_ids = tiles["token_ids"]
            domain_id = tiles.get("domain_id")
            confidence = tiles.get("confidence")
        else:
            token_ids = tiles
            domain_id = None
            confidence = None

        # Embed and compress tiles.
        x, _k, _v = self.tiles_as_kv(token_ids, domain_id, confidence)

        # Prelude.
        for layer in self.prelude:
            x = layer(x)

        # Recurrent loop with deadband ACT + shell LoRA.
        self.deadband.reset(state=x)
        if rooms is not None:
            priority = self.deadband.get_priority(rooms)
            self.deadband.set_priority(priority)

        for step in range(self.config.max_loop_depth):
            x = self.recurrent_block(x)
            x = x + self.shell_lora(x, step)
            if not self.deadband.should_continue(x, step):
                break

        # Coda.
        for layer in self.coda:
            x = layer(x)

        # Optional room-based expert mixing.
        if rooms is not None and "domains" in rooms and "confidences" in rooms:
            x = self.rooms_as_experts(x, rooms["domains"], rooms["confidences"])

        # Output projection.
        logits = self.output_head(x)

        # Return in same format as input.
        if isinstance(tiles, dict):
            out = dict(tiles)
            out["token_ids"] = logits
            return out
        return logits

    @torch.no_grad()
    def generate(
        self,
        tiles: Union[torch.Tensor, Dict[str, Any]],
        rooms: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Auto-regressive generation with PLATO-native routing.

        Args:
            tiles: Initial token tensor or dict.
            rooms: Optional room routing metadata.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: If set, restricts sampling to the k most likely tokens.

        Returns:
            generated: (batch, seq + max_new_tokens) token IDs.
        """
        if isinstance(tiles, dict):
            token_ids = tiles["token_ids"].clone()
        else:
            token_ids = tiles.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(token_ids, rooms)
            # Take the last position.
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)

        return token_ids
