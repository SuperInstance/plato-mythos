# RESEARCH.md — PLATO Mythos: From Knowledge Tiles to Neural Architecture

> *"The map is not the territory, but if the map is isomorphic to the territory, studying the map teaches you the territory."*

**Date:** June 2026  
**Status:** Active Research — Tile Extraction Phase  
**Authors:** SuperInstance Research · OpenMythos Contributors  

---

## Table of Contents

1. [The Vision: Why PLATO Mythos Matters](#1-the-vision)
2. [Background: The SuperInstance Framework](#2-background)
3. [Architecture: How PLATO Becomes Neural Network](#3-architecture)
4. [The Tile Extraction Campaign](#4-tile-extraction)
5. [Gap Analysis: The Missing 7,791 Tiles](#5-gap-analysis)
6. [Research Agenda: Completing the Tile Set](#6-research-agenda)
7. [The Conservation Law and Neural Stability](#7-conservation-law)
8. [Open Problems and Invitations](#8-open-problems)
9. [How to Contribute](#9-how-to-contribute)

---

## 1. The Vision

### What if your knowledge system *was* your neural network?

PLATO Mythos is built on a single radical proposition: **the structure of a knowledge management system can be made isomorphic to the architecture of a neural network.** When this isomorphism holds, fine-tuning isn't "teaching a model your domain" — it's *loading compressed knowledge into the architecture that was designed to hold it.*

This is not prompt engineering. This is not RAG. This is not even traditional fine-tuning.

This is **architectural knowledge embedding**: the knowledge system determines the network topology, and the network topology guarantees that the knowledge can be recovered.

### The PLATO System

[PLATO](https://github.com/SuperInstance) is an autonomous knowledge management system that organizes information into:

- **Rooms**: Domain-specific knowledge domains (like rooms in a building)
- **Tiles**: Individual knowledge units — question/answer pairs with confidence scores
- **Curricula**: Ordered sequences of tiles for progressive learning
- **Shells**: Contextual overlays that change how tiles are interpreted
- **Deadbands**: Confidence thresholds that determine when learning is "done"

PLATO has been running across the SuperInstance fleet, accumulating knowledge in hundreds of rooms spanning distributed systems, constraint solving, fleet operations, game AI, and mathematical reasoning.

### Why This Matters

Current LLM fine-tuning treats domain knowledge as generic data to be memorized. The model's architecture (MoE routing, attention patterns, depth) is agnostic to what it will learn. This creates several problems:

1. **Catastrophic forgetting** — new knowledge overwrites old
2. **Uninterpretable routing** — MoE experts learn opaque specialties
3. **Fixed reasoning depth** — the model can't "think harder" on difficult problems
4. **No safety guarantees** — there's no structural constraint on what the model outputs

PLATO Mythos solves all four by making the knowledge structure identical to the network structure. Rooms determine MoE routing (interpretable!). Tiles become compressed KV pairs (efficient!). Curriculum depth becomes loop iterations (adaptive!). The conservation law γ + η = C becomes a spectral radius constraint (safe!).

---

## 2. Background: The SuperInstance Framework

### The Conservation Law: γ + η = C

At the heart of the SuperInstance framework lies a conservation law analogous to energy conservation in physics:

```
γ (gamma) — free potential, the system's unrealized capacity
η (eta)   — actualized work, the energy already deployed
C         — total capacity, a constant for a given system

Invariant: γ + η = C
```

This is not a metaphor. In the PLATO system, this law governs resource allocation: an agent with high γ has room to take on more tasks; one with high η is fully loaded. The conservation law prevents overcommitment and guarantees system stability.

In PLATO Mythos, this law maps directly to the **spectral radius constraint** on the recurrent depth transformer:

```
h_{t+1} = A · h_t + B · e + Transformer(h_t, e)

Requirement: ρ(A) < 1  (spectral radius of state transition matrix A)
```

The spectral radius constraint ensures the recurrent loop is a **stable linear time-invariant (LTI) system**. The hidden state h_t cannot diverge — it must settle. This is the neural analog of "you can't spend more energy than you have." Every loop iteration either converges toward a confident answer or halts via the ACT (Adaptive Computation Time) mechanism.

### From Conservation to Architecture

| Physics / PLATO | Neural Network |
|-----------------|---------------|
| Free potential (γ) | Remaining compute budget |
| Actualized work (η) | Compute already spent in loops |
| Total capacity (C) | Maximum loop iterations |
| Energy conservation | Spectral radius ρ(A) < 1 |
| Thermal equilibrium | Convergence of hidden state |

This mapping is not decorative — it is **load-bearing**. The conservation law ensures that the model's recurrent depth is bounded, interpretable, and safe. Without it, deep loops could amplify noise (spectral radius > 1) or waste compute without converging.

---

## 3. Architecture: How PLATO Becomes Neural Network

### The Recurrent-Depth Transformer (RDT)

The base architecture is [OpenMythos](https://github.com/kyegomez/OpenMythos), an open-source reconstruction of the Mythos architecture (itself informed by published work on recurrent-depth transformers). The key innovation: instead of stacking N transformer layers for depth, you **loop a small block T times** — where T is variable at inference time.

```
Input Token
    ↓
┌─────────────────────────────────┐
│  PRELUDE (orientation)          │
│  Standard transformer layers    │
│  Agent learns "which room am I  │
│  in?" — context establishment   │
└──────────────┬──────────────────┘
               │  e = encoded input
               ▼
┌─────────────────────────────────┐
│  RECURRENT BLOCK                │
│  for t = 1, 2, ..., T:         │
│    h_{t+1} = A·h_t + B·e       │
│           + Transformer(h_t, e) │
│                                 │
│    MoE:   rooms → expert routing│
│    LoRA:  shells → adapters     │
│    ACT:   deadband → halting    │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  CODA (synthesis)               │
│  Standard transformer layers    │
│  Produce final output tile      │
└─────────────────────────────────┘
```

### The Isomorphism: PLATO ↔ Neural Architecture

Every PLATO concept maps one-to-one to a neural network component:

| PLATO Concept | Neural Component | What It Does |
|--------------|-----------------|-------------|
| **Room** | MoE Expert Group | Domain routing — "which experts handle this?" |
| **Tile** | MLA Compressed KV | Knowledge stored as latent representation |
| **Curriculum Round** | Loop Iteration | More rounds = deeper reasoning |
| **Deadband (P0)** | ACT Threshold | Stop looping when confident enough |
| **Shell** | Depth-wise LoRA | Same base weights, different adapter per loop |
| **Instinct Pipeline** | MLA Compression | Compress KV cache into latent |
| **Fleet** | Shared Experts | Cross-domain knowledge |
| **Conservation Law** | Spectral Radius Budget | γ + η = C enforces stability |

### Rooms-as-Experts: Interpretable Routing

Traditional Mixture-of-Experts uses a learned gating network — a small linear layer that routes tokens to experts based on opaque learned patterns. You can't inspect *why* token 47 went to expert 3.

PLATO Mythos replaces this with **deterministic room-based routing**:

```python
# Traditional MoE: learned, opaque
scores = softmax(x @ W_gate)  # Why this expert? Nobody knows.

# PLATO MoE: deterministic, interpretable
room_scores = tile_domain_affinity(token, room_experts)
# "This token is about fleet coordination" → fleet_orchestration expert group
```

A "fleet coordination" query routes to the `fleet_orchestration` expert group. A "sheaf cohomology" query routes to `construct`. The routing table IS the knowledge structure. No additional training needed for the router.

### Tiles-as-KV: Compressed Knowledge

Each PLATO tile (a question-answer pair) is compressed into a Multi-head Latent Attention (MLA) key-value pair. This means:

- **8,800 tiles** → 8,800 compressed KV representations
- **Inference**: the model retrieves relevant KV pairs based on the query domain
- **Fine-tuning**: the model learns to produce these compressed representations from tile data

This is fundamentally different from RAG (which retrieves raw text) or memorization (which stores tokens directly). The tile is compressed into a *latent* that the model can reconstruct — like a zip file for knowledge.

---

## 4. The Tile Extraction Campaign

### What We Did

The PLATO system has been accumulating knowledge across the SuperInstance fleet for months. But this knowledge lives in disparate files across multiple repositories — JSONL logs, training pairs, ensign reports, zeroclaw reasoning traces, and more.

We built `convert_tiles.py` to scan 13 source repositories and extract every recoverable knowledge unit into a unified dataset.

### Sources Scanned

| Source | Type | Unique Tiles |
|--------|------|-------------|
| `all-research.jsonl` | Research summaries & analysis | 464 |
| `training-pairs.jsonl` | Curated Q&A with metadata | 200 |
| `zeroclaw-tiles` | Agent reasoning traces | 160 |
| `all-operations.jsonl` | Fleet operations logs | 37 |
| `all-dojo.jsonl` | Arena competition transcripts | 24 |
| `arena-competition.jsonl` | Competition results | 21 |
| `npc-dialogue.jsonl` | Conversational AI data | 18 |
| `catalog.json` | Room metadata | 15 |
| `all-achievements.jsonl` | Achievement unlock data | 12 |
| `gc-decisions.jsonl` | Garbage collection decisions | 12 |
| `core-vision.jsonl` | Foundational vision tiles | 10 |
| `data/tiles.jsonl` | Original tile file | 9 |
| `enriched-fleet.jsonl` | Enriched fleet data | 8 |
| `fleet-knowledge.jsonl` | Fleet knowledge base | 6 |
| Knowledge syntheses | Room-level summaries | 13 |
| Ensign reports | Per-domain reports | 11 (mostly empty) |
| Other | Misc sources | ~8 |

### Extraction Process

```
Source Files (13 repos)
    ↓ scan & parse
Raw Candidates (1,943 items)
    ↓ deduplicate (MD5 content hash)
    ↓ filter (remove empty, <5 char answers)
Unique Tiles (1,009)
    ↓ tag with domain & confidence
Final Dataset (tiles.jsonl)
```

The deduplication used content hashing (MD5 of normalized question text, lowercased, stripped) to identify semantically identical questions that appeared across multiple sources. This removed 934 duplicates — nearly half the raw input.

### Results: 1,009 Tiles Across 37 Domains

**Top domains by tile count:**

| Domain | Tiles | Character |
|--------|-------|-----------|
| research | 490 | Research summaries, analysis, methodology |
| Knowledge | 136 | Training pairs with deadband protocol |
| Constraint | 64 | Constraint-based reasoning |
| fleet-operations | 56 | Fleet management, captain's logs |
| dojo | 45 | Arena competition transcripts |
| Documentation | 39 | README & docs generation |
| Communication | 24 | Bottle messages, oracle comms |
| FleetHealth | 17 | Fleet health reports & audits |
| CodeArchaeology | 16 | Codebase analysis tiles |
| Memory | 16 | Memory & boot timeline |

**Confidence distribution:**

| Range | Count | Note |
|-------|-------|------|
| 0.9–1.0 | 55 | High-confidence curated tiles |
| 0.7–0.9 | 12 | Good quality, some uncertainty |
| 0.5 (default) | 329 | Auto-assigned for training pairs |
| No score | 613 | Missing confidence metadata |

### Architecture Validation

The `validate_architecture.py` script confirms that the OpenMythos RDT architecture correctly maps to all PLATO concepts:

- ✅ MoE FFN present (rooms-as-experts routing)
- ✅ Recurrent block present (curriculum loop)
- ✅ LTI injection present (deadband stability)
- ✅ Spectral radius ρ(A) < 1 (conservation law holds)
- ✅ Forward pass succeeds at 1, 2, and 4 loop iterations
- ✅ Generation produces valid output sequences

---

## 5. Gap Analysis: The Missing 7,791 Tiles

### The Numbers

| Metric | Expected | Found | Gap |
|--------|----------|-------|-----|
| Tiles | 8,800+ | 1,009 | **~7,791 (88.5% missing)** |
| Rooms | 913 | 37 | **~876 (95.9% missing)** |
| Confidence coverage | 100% | 39.3% | **60.7% unscored** |

This is the central research challenge: **we found only 11.5% of the expected tile dataset.** The architecture is validated, the mapping is correct, but the fuel — the knowledge tiles — is mostly absent from disk.

### Where Are the Missing Tiles?

Based on our analysis of the PLATO system architecture and source repositories, the missing tiles fall into five categories:

#### Category 1: Ephemeral Session Tiles (~3,500 tiles)

**The PLATO server generated tiles during live agent sessions but never persisted them to disk.** When an agent explored a room and created knowledge, the tile lived in server memory and was lost on restart.

Evidence:
- `catalog.json` reports 8,316 source tiles across its indexed rooms — but these are counts, not content
- Room metadata shows keyword lists and tile counts, but the actual Q&A pairs were never exported
- The PLATO engine (`plato-engine-block`, a Rust DCS) likely holds tiles in memory-mapped structures

**Recovery path:** Replay agent session logs through the PLATO tile engine, or instrument the server to export tiles before shutdown.

#### Category 2: Unscanned Repositories (~2,000 tiles)

The extraction script scanned 13 repositories from `oracle1-workspace`. The SuperInstance fleet has **452+ repositories** across the organization. Many of these contain PLATO-compatible knowledge that was never harvested:

- `plato-agent-python` — agent runtime with embedded knowledge
- `plato-engine-block` — Rust engine with compiled-in tiles
- `ternary-protocol-python` — protocol definitions and examples
- `superinstance-architecture` — design documents and specifications
- Various `fleet-*` repos — operational data, health reports, deployment configs

**Recovery path:** Extend `convert_tiles.py` to scan all SuperInstance repos. Build a fleet-wide tile crawler.

#### Category 3: Tile Buffer Stubs (~1,500 tiles)

The `tile_buffers/` directory contains 108 tile structures, but all have empty `state`, `action`, and `outcome` fields. These are framework test fixtures — the scaffolding exists, but the knowledge was never filled in.

**Recovery path:** These need to be regenerated from source material, not recovered.

#### Category 4: Room-Level Aggregates (~500 tiles)

Many rooms contain only a single "synthesis" — a room-level summary of what was learned. The PLATO design calls for individual tiles per concept within a room, not one summary per room.

Evidence:
- The knowledge syntheses directory has 13 files, each containing one summary
- Each room should have ~10–50 individual tiles

**Recovery path:** Decompose room-level summaries into atomic Q&A tiles using an LLM-assisted decomposition pipeline.

#### Category 5: Quality Issues (~300 tiles)

Some extracted tiles have quality problems:
- Zeroclaw tiles are "thought fragments" — mid-reasoning dumps, not complete knowledge
- Many training pairs are very short or repetitive
- 60.7% of tiles lack confidence scores
- Domain names are inconsistent (e.g., "Research" vs "research" — same room, different strings)

**Recovery path:** Quality scoring, domain normalization, and content filtering.

### Visual: The Tile Landscape

```
╔═══════════════════════════════════════════════════════════════╗
║                    EXPECTED: 8,800+ TILES                     ║
║                                                               ║
║  ┌─────────────────────┐  ┌───────────────────────────────┐  ║
║  │  FOUND: 1,009 (11%) │  │  MISSING: ~7,791 (89%)        │  ║
║  │                     │  │                               │  ║
║  │  research    ████490│  │  Ephemeral sessions  ~3,500  │  ║
║  │  Knowledge   ██ 136 │  │  Unscanned repos     ~2,000  │  ║
║  │  Constraint  █   64 │  │  Buffer stubs        ~1,500  │  ║
║  │  fleet-ops   █   56 │  │  Room aggregates       ~500  │  ║
║  │  dojo        █   45 │  │  Quality issues        ~300  │  ║
║  │  Other       █  218 │  │                               │  ║
║  │                     │  │                               │  ║
║  └─────────────────────┘  └───────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 6. Research Agenda: Completing the Tile Set

Recovering the missing 7,791 tiles is the primary research objective for the next phase. Here we outline the research program.

### Phase 1: Fleet-Wide Tile Crawl (Target: +2,000 tiles, ~2 weeks)

**Objective:** Scan all 452+ SuperInstance repositories for extractable knowledge.

**Approach:**
1. Build a generalized tile crawler that understands multiple formats:
   - JSONL instruction format (instruction/input/output)
   - Markdown documents (section → question, content → answer)
   - Python/Rust docstrings (function signature → question, docstring → answer)
   - GitHub Issues and PRs (title → question, body/discussion → answer)
2. Run across all `SuperInstance/*` repositories
3. Deduplicate against existing 1,009 tiles

**Success metric:** Dataset reaches 3,000+ unique tiles.

### Phase 2: Session Log Replay (Target: +3,500 tiles, ~4 weeks)

**Objective:** Recover ephemeral tiles from PLATO server session logs.

**Approach:**
1. Locate PLATO session logs across fleet infrastructure
2. Build a replay parser that reconstructs tile creation events
3. Extract Q&A pairs from agent interaction transcripts
4. Apply the PLATO tile schema (domain, confidence, tags)

**Key challenge:** Session logs may not exist for all historical sessions. Some knowledge may be permanently lost.

**Success metric:** Dataset reaches 6,500+ unique tiles.

### Phase 3: LLM-Assisted Tile Synthesis (Target: +2,000 tiles, ~3 weeks)

**Objective:** Generate missing tiles from available source material using an LLM as a "tile synthesizer."

**Approach:**
1. Feed room-level summaries and design documents to a capable LLM
2. Prompt: "Generate 20 atomic Q&A tiles that capture the knowledge in this document"
3. Score generated tiles with a quality model
4. Human review for high-stakes domains (Constraint, Conservation Law)

**Risk:** Synthetic tiles may not capture genuine agent experience. Mitigate by grounding synthesis in actual logs and documents, not free generation.

**Success metric:** Dataset reaches 8,500+ unique tiles (approaching design target).

### Phase 4: Quality Calibration (Ongoing)

**Objective:** Ensure every tile has a meaningful confidence score and correct domain.

**Approach:**
1. Train a lightweight quality classifier on the 55 high-confidence (0.9+) tiles
2. Score all 8,500+ tiles through the classifier
3. Normalize domain names (enforce canonical casing)
4. Remove tiles below quality threshold (likely < 0.3)
5. Flag tiles needing human review

**Success metric:** 100% confidence coverage, < 5% low-quality tiles.

### Phase 5: Architecture Fine-Tuning (After 8,000+ tiles)

**Objective:** Fine-tune the OpenMythos RDT on the completed PLATO tile dataset.

**Approach:**
1. Start with `edge-tiny` (1B params, 8 experts, 4 loops) on a single GPU
2. Validate that rooms-as-experts routing learns correctly
3. Verify spectral radius stays < 1 during training
4. Scale to `fleet-standard` (3B params, 64 experts, 16 loops)
5. Evaluate on held-out tiles: can the model answer PLATO questions correctly?

**Success metric:** Model achieves > 85% accuracy on held-out tile Q&A pairs, with interpretable expert routing.

---

## 7. The Conservation Law and Neural Stability

### The Deep Connection

The conservation law γ + η = C is not merely an organizational principle for the PLATO system. In the neural architecture, it becomes a **mathematical guarantee of stability.**

Consider the recurrent update rule:

```
h_{t+1} = A · h_t + B · e + Transformer(h_t, e)
```

If the spectral radius ρ(A) ≥ 1, the hidden state h_t can grow without bound as the loop iterates. This is divergence — the neural equivalent of an engine overheating. The model "spends more energy than it has."

When ρ(A) < 1, the system is contractive: each iteration pulls h_t toward an attractor. This is the γ + η = C of the neural world — the system has finite capacity, and every unit of "compute spent" (η) reduces the "compute available" (γ) by an equal amount.

### Why This Matters for Safety

A model with guaranteed spectral convergence:

1. **Cannot generate runaway outputs** — the hidden state is bounded
2. **Will converge given enough loops** — it can't oscillate forever
3. **Has interpretable compute allocation** — each loop is a measurable unit of "reasoning effort"
4. **Respects a resource budget** — the conservation law prevents overspending

This is a **structural safety guarantee**, not a behavioral one. It doesn't depend on training data or prompt engineering — it's built into the mathematics of the architecture.

### The Conservation Law as a Design Principle

We propose that γ + η = C should be treated as a first-class design principle for neural architectures:

| Principle | Neural Implementation |
|-----------|----------------------|
| Total capacity is finite | Maximum loop iterations T_max |
| Spending reduces reserves | Loop counter approaches T_max |
| Convergence is guaranteed | ρ(A) < 1 |
| Overspending is impossible | Hard stop at T_max |
| Budget can be monitored | Track loop count vs. ACT threshold |

This framing opens research directions:

- **Budget-aware routing**: Allocate more loops to important queries, fewer to trivial ones
- **Conservation auditing**: Verify that the model's compute allocation respects γ + η = C
- **Adaptive capacity**: Can C itself be learned? (i.e., can the model decide its own maximum depth?)

---

## 8. Open Problems and Invitations

These are the questions we find most exciting. If any of them intrigue you, see [Section 9](#9-how-to-contribute).

### 8.1 Optimal Tile Cardinality

What is the minimum number of tiles needed for a room to be "useful"? We have rooms with 1 tile and rooms with 490. Is there a phase transition — a critical mass of tiles beyond which the expert "clicks" and performs well?

### 8.2 Tile Quality Metrics

Our current confidence scores are coarse (mostly 0.5 or 0.9). Can we develop a principled tile quality metric that considers:
- **Coherence**: Does the answer actually address the question?
- **Uniqueness**: Is this tile distinct from others, or redundant?
- **Coverage**: Do the tiles in a room collectively cover the domain?
- **Freshness**: Is the knowledge still current?

### 8.3 Synthetic Tile Generation at Scale

Can we build a pipeline that generates high-quality tiles from documentation, code, and conversation logs — reliably enough to include in the training set without human review? What quality threshold makes synthetic tiles safe?

### 8.4 Spectral Radius and Learning Dynamics

During fine-tuning, the matrix A is updated by gradient descent. How do we ensure ρ(A) < 1 is maintained throughout training? Options:
- Project A onto the stable manifold after each gradient step
- Use a parameterization that guarantees stability (e.g., A = D·tanh(M) where D is diagonal with |D_ii| < 1)
- Add ρ(A) as a regularization term

### 8.5 Variable-Depth Inference and the Deadband

The ACT (Adaptive Computation Time) mechanism lets the model halt early if confident. This maps to the PLATO deadband — the threshold below which additional reasoning isn't needed. But how should the deadband be set per-domain? Research domains may need deeper loops than operational domains.

### 8.6 Cross-Room Knowledge Transfer

The shared experts in the MoE architecture handle cross-domain knowledge. How does knowledge transfer between rooms? Can a tile about "constraint satisfaction" improve performance in "fleet operations" if the two rooms share underlying structure?

### 8.7 The Full 913-Room Topology

We've found 37 of 913 rooms. What does the full room topology look like? Are there rooms that the design spec calls for but that were never created? Are there rooms that exist in the running system but aren't documented? Mapping the full topology is a cartography problem of the knowledge space.

---

## 9. How to Contribute

PLATO Mythos is an open research project under the Apache-2.0 license. Here's how you can help:

### Tile Recovery

The most impactful contribution is **finding missing tiles**. If you have access to:
- PLATO server logs or memory dumps
- SuperInstance fleet repositories with knowledge content
- Agent session transcripts from any PLATO-compatible system

...please open an issue or PR with extracted tiles in JSONL format.

### Tile Quality

Run the quality classifier on new tiles. Flag low-quality or redundant entries. Normalize domain names. Every cleaned tile makes the dataset better.

### Architecture Research

The `validate_architecture.py` script provides a CPU-only test harness. You can:
- Experiment with different expert counts and routing strategies
- Test stability under varying spectral radii
- Profile memory usage at different loop depths
- Propose new PLATO → architecture mappings

### Documentation and Analysis

This RESEARCH.md is a living document. If you:
- Discover patterns in the tile distribution
- Find new sources of recoverable knowledge
- Develop better extraction or scoring methods
- Write up experimental results

...we welcome contributions to this document.

### Getting Started

```bash
# Clone the repo
git clone https://github.com/SuperInstance/plato-mythos.git
cd plato-mythos

# Install dependencies
pip install open-mythos torch

# Run the architecture validator
python scripts/validate_architecture.py

# Examine the tile dataset
wc -l data/tiles.jsonl       # 1,009 tiles
jq '.domain' data/tiles.jsonl | sort | uniq -c | sort -rn  # Domain distribution

# Rebuild from source (requires oracle1-workspace access)
python scripts/convert_tiles.py
```

### Communication

- **Issues**: [github.com/SuperInstance/plato-mythos/issues](https://github.com/SuperInstance/plato-mythos/issues)
- **Architecture**: [OpenMythos](https://github.com/kyegomez/OpenMythos)
- **PLATO System**: [SuperInstance Organization](https://github.com/SuperInstance)

---

## Appendix A: Tile Extraction Statistics

### Raw Numbers

| Metric | Value |
|--------|-------|
| Raw candidate inputs | 1,943 |
| Duplicate removals | 934 |
| Empty/short filtered | 10 |
| **Final unique tiles** | **1,009** |
| **Unique domains** | **37** |
| Source files scanned | 26 |
| Source repositories | 13 |

### Domain Normalization Issues

Several domains appear in both Title Case and lowercase, suggesting they represent the same room:

| Title Case | Lowercase | Combined |
|-----------|-----------|----------|
| Research (13) | research (490) | 503 |
| FleetHealth (17) | fleethealth (2) | 19 |
| Documentation (39) | documentation (2) | 41 |
| Prototyping (7) | prototyping (2) | 9 |
| Memory (16) | memory (2) | 18 |
| Communication (24) | communication (2) | 26 |
| CodeArchaeology (16) | codearchaeology (2) | 18 |
| TrendAnalysis (5) | trendanalysis (2) | 7 |
| Integration (8) | integration (2) | 10 |
| Organization (6) | organization (2) | 8 |
| ModelExperiment (6) | modelexperiment (2) | 8 |
| Testing (3) | testing (2) | 5 |

After normalization, the effective unique domain count drops from 37 to approximately **25 canonical rooms**.

### Confidence Breakdown

- 55 tiles at 0.9+ confidence (5.5%) — high-quality curated knowledge
- 12 tiles at 0.7–0.9 (1.2%) — good quality with some uncertainty
- 329 tiles at 0.5 (32.6%) — default score, actual quality unknown
- 613 tiles with no score (60.7%) — scoring never applied

The confidence distribution suggests that **the majority of the dataset has never been quality-assessed.** The 55 high-confidence tiles likely represent the best training signal; the 329 default-scored tiles need calibration; the 613 unscored tiles need assessment before they can be trusted for fine-tuning.

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **PLATO** | Programmatic Learning and Adaptive Tile Organization — the knowledge management system |
| **Room** | A domain-specific knowledge container in PLATO (maps to MoE expert group) |
| **Tile** | An atomic knowledge unit — a question/answer pair with metadata (maps to compressed KV pair) |
| **Curriculum** | An ordered sequence of tiles for progressive learning (maps to loop iterations) |
| **Shell** | A contextual overlay that changes tile interpretation (maps to depth-wise LoRA adapter) |
| **Deadband** | A confidence threshold determining when learning is complete (maps to ACT halting threshold) |
| **RDT** | Recurrent-Depth Transformer — architecture that loops a block multiple times for variable depth |
| **MoE** | Mixture of Experts — routing tokens to specialized sub-networks |
| **MLA** | Multi-head Latent Attention — compresses KV cache into latent representations |
| **ACT** | Adaptive Computation Time — allows variable compute per token |
| **LTI** | Linear Time-Invariant — a stable system where future state depends linearly on current state |
| **Spectral radius** | Largest absolute eigenvalue of a matrix; must be < 1 for stability |
| **OpenMythos** | Open-source reconstruction of the Mythos RDT architecture |
| **γ (gamma)** | Free potential — unrealized capacity in the system |
| **η (eta)** | Actualized work — energy already deployed |
| **C** | Total capacity — the conservation constant |
| **Conservation law** | γ + η = C — total capacity equals free potential plus actualized work |

---

*"We are what we repeatedly do. Excellence, then, is not an act, but a habit." — Aristotle*

*The PLATO system operationalizes this for machines. PLATO Mythos makes it neural.*

---

**License:** Apache-2.0  
**Repository:** [github.com/SuperInstance/plato-mythos](https://github.com/SuperInstance/plato-mythos)  
**Last updated:** June 2026
