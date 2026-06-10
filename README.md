# plato-mythos

> PLATO is the model. Rooms are experts. Tiles are knowledge. Curriculum is depth.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![SuperInstance](https://img.shields.io/badge/SuperInstance-Fleet-green.svg)](https://github.com/SuperInstance)

## What It Is

plato-mythos is a Recurrent-Depth Transformer (RDT) fine-tuned on PLATO tile data, where the SuperInstance knowledge system *is* the neural architecture. Built on [OpenMythos](https://github.com/kyegomez/OpenMythos) — the open-source reconstruction of the Claude Mythos architecture.

**The core insight**: You don't fine-tune a generic model and hope it learns your domain. You structure the knowledge system so that the architecture *emerges* from it.

```
PLATO Room    →  MoE Expert Group     (routing, not random gates)
PLATO Tile    →  MLA Compressed KV    (knowledge = latent representation)
PLATO Curriculum → RDT Loop Depth    (more rounds = deeper reasoning)
PLATO Deadband → ACT Halting         (stop when confident)
PLATO Shell   →  Depth-wise LoRA     (same base, different adapter per loop)
```

## Architecture

```
Input Token
    ↓
┌─────────────────────────────────────┐
│  PRELUDE (orientation)              │
│  Standard transformer layers × N    │
│  Agent learns room context          │
└──────────────┬──────────────────────┘
               │  e = encoded input
               ▼
┌─────────────────────────────────────┐
│  RECURRENT BLOCK (reasoning loop)   │
│  h_{t+1} = A·h_t + B·e             │
│         + Transformer(h_t, e)       │
│                                     │
│  MoE: rooms → expert routing        │
│  LoRA: shells → depth adapters      │
│  ACT: deadband → halting signal     │
│                                     │
│  Looped T times (variable depth)    │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  CODA (synthesis)                   │
│  Standard transformer layers × M    │
│  Produce final tile output          │
└──────────────┬──────────────────────┘
               ▼
         Output Token
```

**Update rule** (per loop iteration):
```python
h_{t+1} = A @ h_t + B @ e + Transformer(h_t, e)
# A must have spectral radius ρ(A) < 1 (LTI stability)
# e is the Prelude output, re-injected every loop (prevents drift)
```

## Why This Is Different from Fine-Tuning

| Traditional Fine-Tuning | plato-mythos |
|------------------------|--------------|
| Generic model + your data | Knowledge system *is* the architecture |
| Random MoE routing (learned gates) | Room membership determines routing |
| Fixed depth (number of layers) | Variable depth (loop count at inference) |
| Separate training data | PLATO tiles (8,800+ already exist) |
| No safety guarantees | Conservation law (γ + η = C) enforced |
| Black-box decisions | Interpretable routing (room → expert) |

## Dataset

### Current State (1,009 tiles extracted)

| Metric | Value |
|--------|-------|
| Unique tiles | 1,009 |
| Domains | 37 |
| Tiles with confidence | 396 (39.3%) |
| Mean confidence | 0.52 |
| Source repos scanned | 13 |

**Top domains**: research (490), Knowledge (136), Constraint (64), fleet-operations (56), dojo (45)

**Gap**: Design docs reference 8,800+ tiles across 913 rooms. Only 1,009 found on disk — the rest exist in live PLATO server memory or need regeneration from source repos.

### Dataset Format

```jsonl
{"messages": [{"role": "user", "content": "What is the conservation law?"}, {"role": "assistant", "content": "γ + η = C. Free potential plus actualized work equals capacity..."}], "domain": "Constraint", "confidence": 0.95}
```

### Data Location

```
data/
├── tiles.jsonl          # 1,009 unique Q&A tiles in JSONL
├── dataset-summary.md   # Full statistics & gap analysis
├── _stats.json          # Machine-readable stats
scripts/
└── convert_tiles.py     # Reproducible conversion from source repos
```

## Variants

| Variant | Params | Dim | Experts | Loops | Context | Target Hardware |
|---------|--------|-----|---------|-------|---------|-----------------|
| edge-tiny | 1B | 512 | 8 | 4 | 4k | Jetson Orin (2GB VRAM) |
| fleet-standard | 3B | 3072 | 64 | 16 | 4k | Oracle Cloud A100 |
| research-heavy | 10B | 4096 | 128 | 24 | 8k | RTX 4090 / H100 |

## Quick Start

### Install Dependencies

```bash
pip install open-mythos torch
```

### Validate Architecture (CPU)

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

# Edge-tiny config for PLATO
cfg = MythosConfig(
    vocab_size=32000,
    dim=512,
    n_heads=8,
    max_seq_len=4096,
    max_loop_iters=4,
    prelude_layers=2,
    coda_layers=1,
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=2048,
    lora_rank=8,
    attn_type="mla",
    n_kv_heads=8,
    kv_lora_rank=64,
    q_lora_rank=128,
    qk_rope_head_dim=32,
    qk_nope_head_dim=32,
    v_head_dim=32,
)

model = OpenMythos(cfg)
ids = torch.randint(0, cfg.vocab_size, (1, 64))
logits = model(ids, n_loops=4)
print(f"Logits: {logits.shape}")  # [1, 64, 32000]

# Verify LTI stability
A = model.recurrent.injection.get_A()
rho = torch.linalg.eigvals(A).abs().max().item()
print(f"Spectral radius ρ(A) = {rho:.4f} (must be < 1)")
```

### Train on PLATO Tiles

```python
# Fine-tune on our tile dataset
from torch.utils.data import Dataset, DataLoader
import json

class PlatoTileDataset(Dataset):
    def __init__(self, path="data/tiles.jsonl"):
        self.tiles = []
        with open(path) as f:
            for line in f:
                tile = json.loads(line)
                # Convert messages to token sequences
                self.tiles.append(tile)
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        return self.tiles[idx]

dataset = PlatoTileDataset()
print(f"Loaded {len(dataset)} PLATO tiles across {len(set(t['domain'] for t in dataset.tiles))} domains")
```

## The PLATO → Mythos Mapping

Every PLATO concept maps to a neural network component:

```
┌──────────────────┐     ┌──────────────────┐
│   PLATO SYSTEM   │     │  RDT ARCHITECTURE│
├──────────────────┤     ├──────────────────┤
│ Room             │────▶│ MoE Expert Group │  Domain routing
│ Tile             │────▶│ MLA KV Pair      │  Compressed knowledge
│ Curriculum Round │────▶│ Loop Iteration   │  Depth = reasoning
│ Deadband (P0)    │────▶│ ACT Threshold    │  Confidence halt
│ Shell            │────▶│ Depth LoRA       │  Per-loop adapter
│ Instinct Pipeline│────▶│ MLA Compression  │  KV → latent
│ Fleet            │────▶│ Shared Experts   │  Cross-domain
│ Conservation     │────▶│ Budget Reserve   │  γ + η = C
└──────────────────┘     └──────────────────┘
```

## Rooms-as-Experts Routing

Instead of learned MoE gates, plato-mythos routes tokens based on PLATO room membership:

```python
# Traditional MoE: black-box learned routing
scores = softmax(x @ W_gate)

# PLATO MoE: deterministic room-based routing
room_scores = tile_domain_affinity(token, room_experts)
```

A "fleet coordination" token routes to the `fleet_orchestration` expert group. A "sheaf cohomology" token routes to `construct`. No training needed for the router — the knowledge structure IS the routing table.

## Training Compute Options

| Option | Hardware | Cost | Trainable Variant |
|--------|----------|------|-------------------|
| Google Colab Free | T4 16GB | Free | 1B edge-tiny |
| Google Colab Pro | A100 40GB | $10/mo | 3B fleet-standard |
| Lambda Labs | H100 80GB | $2.50/hr | 10B research-heavy |
| Vast.ai | RTX 4090 | $0.30/hr | 3B fleet-standard |
| Jetson Orin (JC1) | 2GB VRAM | Free (existing) | 1B inference only |

## Integration with SuperInstance Fleet

| Fleet Worker | plato-mythos Role |
|-------------|-------------------|
| forge-pi | Routes queries to model via rooms-as-experts |
| fleet-edge | Dispatches model inference requests |
| fleet-vector-api | Provides semantic search for capability routing |
| fleet-budget | Enforces conservation law on inference budget |
| fleet-event-router | Pub/sub for model inference events |

## References

- [OpenMythos](https://github.com/kyegomez/OpenMythos) — Base RDT architecture
- [Recurrent Transformer paper](https://arxiv.org/abs/2604.21215) — Layerwise recurrent memory
- [Parcae](https://arxiv.org/abs/2604.12946) — Stable recurrent-depth scaling
- [PLATO Mythos Design](https://github.com/SuperInstance/oracle1-workspace) — Original architecture mapping
- [Conservation Law](https://github.com/SuperInstance/superinstance-architecture) — γ + η = C

## Related

- [plato-agent-python](https://github.com/SuperInstance/plato-agent-python) — PLATO agent runtime
- [plato-engine-block](https://github.com/SuperInstance/plato-engine-block) — Rust DCS engine
- [ternary-protocol-python](https://github.com/SuperInstance/ternary-protocol-python) — Ternary {-1,0,+1} messaging
- [superinstance-architecture](https://github.com/SuperInstance/superinstance-architecture) — Full stack architecture
- [forge-pi](https://github.com/SuperInstance/forge-pi) — Edge agent runtime

## License

Apache-2.0 — See [LICENSE](LICENSE)
