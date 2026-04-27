# Plato-Mythos

PLATO-native Recurrent-Depth Transformer mapping PLATO control-theory concepts to OpenMythos generative modelling.

## PLATO ↔ Mythos Mapping

| PLATO Concept | Mythos Translation | Implementation |
|---------------|-------------------|----------------|
| **Tiles** | Compressed latent memory units | `TilesAsKV` — embeds tokens and compresses them into MLA latent KV pairs |
| **Rooms** | Interpretable expert groups | `RoomsAsExperts` — routes tokens to domain-tagged MoE experts using confidence-biased gating |
| **Curriculum** | Adaptive compute budget | `CurriculumScheduler` + `DeadbandACT` — easy tokens get fewer loop iterations, hard tokens get more |
| **Deadbands (P0/P1/P2)** | Priority-aware halting thresholds | `DeadbandACT` — P0=0.99 (critical), P1=0.8 (standard), P2=0.5 (low) controls when the recurrent loop stops thinking |
| **Shells** | Depth-wise LoRA adapters | `ShellLoRA` — each loop iteration owns unique low-rank A/B matrices while sharing a base weight |

## Architecture

```
Tiles (token_ids + metadata)
    │
    ▼
┌─────────────────┐
│  TilesAsKV      │  embed + compress → latent KV
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Prelude layers │  static transformer blocks
└─────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Recurrent loop (max_loop_depth iterations)  │
│    x = RecurrentBlock(x)                     │
│    x = x + ShellLoRA(x, step)                │
│    if DeadbandACT.should_continue(x, step):  │
│        break                                 │
└──────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│  Coda layers    │  static transformer blocks
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  RoomsAsExperts │  domain-routed MoE mixing
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Output head    │  d_model → vocab logits
└─────────────────┘
    │
    ▼
Generated Tiles (same format as input)
```

## Quick Start

```python
from plato_mythos import PlatoMythos, PlatoMythosConfig

config = PlatoMythosConfig(vocab_size=32000, d_model=1024, max_loop_depth=8)
model = PlatoMythos(config)

# Forward pass with raw token tiles
tiles = torch.randint(0, config.vocab_size, (2, 16))
logits = model(tiles)  # (2, 16, vocab_size)

# Forward pass with PLATO-style metadata
tiles = {
    "token_ids": torch.randint(0, config.vocab_size, (1, 8)),
    "domain_id": torch.zeros(1, dtype=torch.long),
    "confidence": torch.ones(1) * 0.9,
}
rooms = {"domains": ["room_0"] * 8, "confidences": [0.9] * 8}
out = model(tiles, rooms)
```

## Installation

```bash
pip install -e .
```

## Testing

```bash
pytest tests/
```
