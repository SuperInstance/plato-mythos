# PLATO Tile Gap Analysis

**Date:** 2026-06-11  
**Status:** Active — Tile Extraction Phase  
**Target:** 8,800+ tiles across 913 rooms  
**Current:** 1,009 tiles across ~25 canonical rooms (37 raw domains)  

---

## Executive Summary

The PLATO Mythos project has extracted **1,009 unique knowledge tiles** from the `oracle1-workspace` repository. The design target is **8,800+ tiles across 913 rooms**, leaving a gap of **~7,791 tiles (88.5%)** and **~876 rooms (96%)**.

This analysis surveys **17 `plato-*` repositories**, **361 `ternary-*` repositories**, **17 `fleet-*` repositories**, and **10 `superinstance-*` repositories** on disk to identify unscanned sources and estimate recoverable tile content.

---

## Current Dataset State

| Metric | Value |
|--------|-------|
| Unique tiles | 1,009 |
| Raw candidates (pre-dedup) | 1,943 |
| Duplicates removed | 934 |
| Raw domains | 37 |
| Canonical domains (after case normalization) | ~25 |
| Tiles with confidence scores | 396 (39.3%) |
| Mean confidence | 0.52 |
| Source repos scanned | 1 (`oracle1-workspace`) |

### Top Domains by Tile Count

| Domain | Tiles | Notes |
|--------|-------|-------|
| research | 503 | After merging Research + research |
| Knowledge | 136 | Training pairs with deadband protocol |
| Constraint | 64 | Constraint-based reasoning |
| fleet-operations | 56 | Fleet management, captain's logs |
| dojo | 45 | Arena competition transcripts |
| Documentation | 41 | After merge |
| Communication | 26 | After merge |
| FleetHealth | 19 | After merge |
| CodeArchaeology | 18 | After merge |
| Memory | 18 | After merge |
| Other (15 domains) | 83 | Various small domains |

---

## PLATO Repos on Disk (17 total)

### Already Scanned

| Repo | Status | Notes |
|------|--------|-------|
| `oracle1-workspace` | ✅ Scanned | Primary source for all 1,009 tiles |
| `plato-mythos` | ✅ Home | Destination repo, contains tiles.jsonl |

### Not Yet Scanned — High Priority

| Repo | Language | Estimated Tiles | Rationale |
|------|----------|----------------|-----------|
| `plato-agent-python` | Python | 30–50 | Agent runtime with room.py, protocol.py, escalation.py — Q&A about agent behavior, room management, alarm handling, tick processing |
| `plato-engine-block` | Rust | 40–60 | Core DCS engine with sensor.rs, actuator.rs, alarm.rs, history.rs, tick.rs — Q&A about engine internals, protocol commands |
| `plato-engine-block-c` | C | 20–30 | C implementation with protocol headers — Q&A about C API, server setup |
| `plato-runtime-kernel` | Rust | 40–60 | Spatial spreadsheet engine with RoomDepth enum (Floor/Board/Panel/Code/Metal), Baton, TutorLoop, GridBridge, AssertionTrap — rich conceptual domain |
| `plato-room-configs` | JSON/Rust | 50–80 | 18 room configs across 5 environments (fishing-boat, smart-home, server-rack, game-world, factory) — each room has sensors, actuators, alarms, metadata |
| `plato-ternary-bridge` | Rust | 20–30 | Ternary {-1,0,+1} threshold/alarm bridging — Q&A about ternary logic, thresholds |
| `plato-fleet-manager` | Rust | 20–30 | Fleet monitoring with HealthStatus enum — Q&A about fleet health, agent lifecycle |
| `plato-flux-compiler` | Rust | 30–40 | DSL compiler with parser, AST, codegen, optimizer — Q&A about flux rules, conditions, compilation |

### Not Yet Scanned — Medium Priority

| Repo | Language | Estimated Tiles | Rationale |
|------|----------|----------------|-----------|
| `plato-dashboard` | Rust | 15–25 | Terminal dashboard with Color/AlarmSeverity/FleetHealth enums — Q&A about visualization, alarm display |
| `plato-music-sync` | Rust | 30–50 | Polyrhythm, counterpoint, cadence, tempo, groove — rich music theory domain with Direction/MotionType/CadenceType/RoomState/TempoMarking enums |
| `plato-demo` | Rust | 10–20 | Minimal engine + music demos with Motion/CadenceType/Severity/ScenarioPhase enums |
| `plato-fleet-chapel` | Chapel | 15–25 | Fleet management in Chapel with FleetManager, PlatoEngine, GrooveTracker, Ternary |
| `plato-quickstart` | Rust | 5–10 | Minimal quickstart example |

### Not Yet Scanned — Low Priority (Language Ports)

| Repo | Language | Estimated Tiles | Rationale |
|------|----------|----------------|-----------|
| `plato-engine-block-elixir` | Elixir | 15–25 | Erlang/OTP port with RoomSupervisor, FleetSupervisor, Ternary modules |
| `plato-engine-block-gleam` | Gleam | 10–15 | BEAM port — smaller surface area |
| `plato-engine-block-zig` | Zig | 10–15 | Zig port with dashboard, protocol, engine, ternary modules |

### PLATO Repo Scan Summary

| Priority | Repos | Est. New Tiles |
|----------|-------|---------------|
| High | 8 | 255–400 |
| Medium | 5 | 100–160 |
| Low | 3 | 35–55 |
| **Total** | **16** | **390–615** |

---

## Ternary Repos on Disk (361 total)

The `ternary-*` namespace contains **361 repositories** spanning an enormous range of topics. These are a **massive untapped source** of tile-like knowledge.

### Categories with High Tile Potential

| Category | Example Repos | Count | Est. Tiles/Repo |
|----------|--------------|-------|----------------|
| Core systems | ternary-core, ternary-engine, ternary-runtime | ~5 | 20–40 |
| Protocol | ternary-protocol, ternary-protocol-python | ~3 | 30–50 |
| Mathematics | ternary-arithmetic, ternary-matrix, ternary-tensor, ternary-algebra | ~15 | 10–20 |
| AI/ML | ternary-transformer, ternary-language-model, ternary-inference | ~10 | 20–40 |
| Music/Audio | ternary-music, ternary-rhythm, ternary-polyrhythm, ternary-tempo | ~8 | 15–30 |
| Data structures | ternary-btree, ternary-ring, ternary-heap, ternary-graph | ~12 | 10–20 |
| Distributed | ternary-consensus, ternary-paxos, ternary-distributed | ~8 | 15–25 |
| Physics | ternary-thermodynamics, ternary-energy, ternary-hamiltonian | ~6 | 10–20 |
| Biology | ternary-genome, ternary-genetic, ternary-evolution-advanced | ~5 | 10–15 |
| Games | ternary-games, ternary-game-theory, ternary-game-of-life | ~5 | 10–20 |
| Other | ternary-chaos, ternary-crystal, ternary-fire, ternary-sandpile... | ~284 | 5–15 |

### Ternary Tile Estimate

| Approach | Repos | Est. Tiles |
|----------|-------|-----------|
| Top 50 repos (deep scan) | 50 | 750–1,500 |
| All repos (shallow README scan) | 361 | 1,800–5,400 |
| Docstrings + code comments only | 361 | 500–1,000 |

**Realistic estimate from ternary repos: 2,000–4,000 new tiles** (with LLM-assisted extraction from READMEs and docstrings)

---

## Fleet & SuperInstance Repos

| Namespace | Count | Est. Tiles | Notes |
|-----------|-------|-----------|-------|
| `fleet-*` | 17 | 200–500 | Fleet operations, auth, health, events, budgets — operational knowledge |
| `superinstance-*` | 10 | 150–300 | Architecture, protocol, knowledge, ecosystem — design-level knowledge |
| `forge-pi` | 1 | 20–50 | Edge agent runtime |
| **Total** | **28** | **370–850** |

---

## Tile Buffer Stubs in oracle1-workspace

The `tile_buffers/` directory contains **108 tile JSON files** across 12 rooms:

```
test-continual, test-curriculum, test-distill, test-evolve,
test-fewshot, test-imitate, test-inverse_rl, test-meta_learn,
test-multitask, test-neurosymbolic, test-qlora, test-wiki
```

**Problem:** All have empty `state`, `action`, and `outcome` fields. These are scaffolding — the room names suggest learning paradigms (continual learning, meta-learning, few-shot, QLoRA, etc.) but contain no actual knowledge.

**Recovery:** These cannot be recovered — they need to be regenerated from source material or synthesized fresh.

---

## Experiment Data in oracle1-workspace

The `experiments/` directory contains:
- **11+ experiment results** (`experiments/results/experiment-*.json`)
- **9+ experiment questions** (`experiments/questions/experiment-*.json`)

These contain structured Q&A about PLATO vs RAG comparisons, room density, latency, etc. — potentially **30–50 tiles** from experiment questions alone.

---

## Mythological References in Code

Scanning all `plato-*` source code for Greek/Roman mythology references:

| Repo | References | Details |
|------|-----------|---------|
| `plato-runtime-kernel` | `hydrate` (Tantalus) | Grid hydration methods |
| `plato-room-configs` | Fishing vessel "F/V Northern Star" | Nautical naming |
| `plato-music-sync` | Orphic/polyrhythm concepts | Music-theory domain |

**Assessment:** The PLATO repos use minimal direct mythological naming. The "mythology" in plato-mythos is architectural metaphor (PLATO → Plato the philosopher), not literal Greek references. The ternary namespace has a few (`ternary-muse`, `ternary-oracle`, `ternary-prophet`, `ternary-lighthouse`, `ternary-tempest`) that could yield themed tiles.

---

## Gap Summary: Where Are the Missing 7,791 Tiles?

| Source | Status | Est. Recoverable Tiles | Effort |
|--------|--------|----------------------|--------|
| **plato-* repos (16 unscanned)** | Not scanned | 390–615 | Medium (2–3 days) |
| **ternary-* repos (361)** | Not scanned | 2,000–4,000 | High (1–2 weeks) |
| **fleet-* + superinstance-* (28)** | Not scanned | 370–850 | Medium (3–5 days) |
| **oracle1-workspace experiments** | Not scanned | 30–50 | Low (1 hour) |
| **oracle1-workspace tile_buffers** | Empty stubs | 0 (need regeneration) | N/A |
| **Ephemeral session tiles** | Lost (server memory) | 0 recoverable, ~3,500 synthetic | High (needs server instrumentation) |
| **LLM-synthesized from docs** | Not done | 1,000–2,000 | Medium (3–5 days) |
| **Total recoverable** | — | **3,790–7,515** | — |

### Scenario Analysis

| Scenario | Tiles After | Gap Remaining |
|----------|-------------|---------------|
| Scan plato-* repos only | ~1,400–1,600 | ~7,200 |
| + Scan fleet/superinstance repos | ~1,800–2,400 | ~6,400 |
| + Scan top 50 ternary repos | ~2,500–3,900 | ~4,900 |
| + Scan all ternary repos | ~3,800–7,900 | ~900 |
| + LLM synthesis from docs | ~4,800–9,900 | **Target met** ✅ |

**Conclusion:** Scanning all ternary repos + LLM synthesis is the path to reaching 8,800 tiles. The plato-* repos alone add ~400–600 tiles.

---

## Recommended Extraction Order

### Phase 1: Quick Wins (1 day)
1. Extract tiles from `oracle1-workspace/experiments/questions/` → ~30–50 tiles
2. Scan `plato-room-configs/` configs → ~50–80 tiles (18 rooms × 3–4 Q&A each)
3. Scan `plato-runtime-kernel/` concepts → ~40–60 tiles (RoomDepth, Baton, TutorLoop, GridBridge)

### Phase 2: PLATO Repo Deep Scan (2–3 days)
4. Scan remaining 14 `plato-*` repos via README → tile conversion
5. Extract Q&A from Python/Rust docstrings across all plato repos
6. Target: +390–615 tiles, total ~1,400–1,600

### Phase 3: Fleet & SuperInstance Scan (3–5 days)
7. Scan `fleet-*` repos for operational knowledge
8. Scan `superinstance-*` repos for architecture/design knowledge
9. Target: +370–850 tiles, total ~1,800–2,400

### Phase 4: Ternary Deep Scan (1–2 weeks)
10. Build generalized crawler for ternary-* repos
11. Priority scan top 50 by README word count
12. Shallow scan remaining 311 repos (README only)
13. Target: +2,000–4,000 tiles, total ~3,800–6,400

### Phase 5: LLM Synthesis (3–5 days)
14. Feed room-level summaries and design docs to LLM
15. Generate 20–50 atomic Q&A tiles per document
16. Quality score and human review for high-stakes domains
17. Target: +1,000–2,000 tiles, total ~4,800–8,400

### Phase 6: Quality & Domain Normalization (ongoing)
18. Normalize domain names (canonical lowercase)
19. Score all tiles for confidence
20. Remove tiles below quality threshold
21. Decompose room-level aggregates into atomic tiles

---

## Architecture Validation Status

The `validate_architecture.py` script confirms:

- ✅ MoE FFN present (rooms-as-experts routing)
- ✅ Recurrent block present (curriculum loop)
- ✅ LTI injection present (deadband stability)
- ✅ Spectral radius ρ(A) < 1 (conservation law holds)
- ✅ Forward pass succeeds at 1, 2, and 4 loop iterations
- ✅ Generation produces valid output sequences

**Architecture is validated. The bottleneck is entirely data — not model design.**

---

## Key Data Structures Found

### PLATO Room Architecture (from plato-runtime-kernel)

```
RoomDepth: Floor → Board → Panel → Code → Metal
           (agents)  (tools)   (settings) (functions) (bits)
```

### PLATO Room Config Schema (from plato-room-configs)

Each room has: `room_id`, `name`, `tick_hz`, `history_capacity`, `sensors{}`, `actuators{}`, `alarms[]`, `metadata{}`

### 18 Physical/Virtual Rooms Defined

| Environment | Rooms |
|-------------|-------|
| Fishing Boat | engine_room, backdeck, wheelhouse, crows_nest, galley, bilge |
| Smart Home | living_room, garage, kitchen |
| Server Rack | rack_unit, network_switch, pdu |
| Game World | tavern, dungeon, forest |
| Factory | warehouse, production_line, hvac |

### Key Enums Across Repos

| Enum | Values | Repo |
|------|--------|------|
| `RoomDepth` | Floor, Board, Panel, Code, Metal | plato-runtime-kernel |
| `AlarmState` | variants | plato-engine-block |
| `AlarmSeverity` | variants | plato-dashboard |
| `FleetHealth` | variants | plato-dashboard |
| `HealthStatus` | variants | plato-fleet-manager |
| `CadenceType` | variants | plato-music-sync, plato-demo |
| `MotionType` / `Direction` | variants | plato-music-sync |
| `TempoMarking` | variants | plato-music-sync |
| `RoomState` | variants | plato-music-sync |
| `TernaryThreshold` | variants | plato-ternary-bridge |
| `AlarmPriority` | variants | plato-ternary-bridge |
| `MergeLine` | variants | plato-runtime-kernel |

---

## Appendix: Source File Inventory

### oracle1-workspace (already scanned)

```
data/tiles.jsonl                  → 9 tiles
training-data/training-pairs.jsonl → 200 tiles
training-data/zeroclaw-tiles/     → 160 tiles
training-data/gc-decisions.jsonl  → 12 tiles
training-data/knowledge/*.json    → 13 tiles
training-data/research/*.jsonl    → 464 tiles
artifacts/index/catalog.json      → 15 tiles
tile_buffers/                     → 0 tiles (all empty stubs)
experiments/                      → NOT YET EXTRACTED (~30-50 tiles)
```

### PLATO Repos (not yet scanned)

```
plato-agent-python/  → room.py, protocol.py, agent.py, config.py, escalation.py
plato-engine-block/  → engine.rs, protocol.rs, sensor.rs, actuator.rs, alarm.rs, history.rs, tick.rs
plato-runtime-kernel/ → lib.rs (RoomDepth, Baton, TutorLoop, GridBridge, AssertionTrap), delta.rs, merge.rs
plato-room-configs/  → 18 room JSON configs, schemas, loader, validator
plato-ternary-bridge/ → threshold.rs, alarm_ternary.rs
plato-fleet-manager/ → monitor.rs (HealthStatus)
plato-music-sync/    → polyrhythm.rs, counterpoint.rs, cadence.rs, tempo.rs, groove.rs
plato-flux-compiler/ → parser.rs, ast.rs, codegen.rs, optimizer.rs
plato-dashboard/     → render.rs, room_panel.rs, fleet_panel.rs
plato-demo/          → minimal_engine.rs, minimal_music.rs, scenario.rs
plato-fleet-chapel/  → FleetManager.chpl, PlatoEngine.chpl, GrooveTracker.chpl
plato-engine-block-c/ → plato_engine.h, server.c, sensors_dummy.c
plato-engine-block-elixir/ → room.ex, fleet.ex, ternary.ex, protocol.ex
plato-engine-block-gleam/ → room.gleam, fleet.gleam, ternary.gleam
plato-engine-block-zig/ → engine.zig, protocol.zig, ternary.zig
plato-quickstart/    → minimal example
```

---

## Conclusion

The PLATO Mythos project has a validated architecture and a solid conceptual mapping from PLATO knowledge structures to neural network components. The critical bottleneck is **tile data volume**:

- **1,009 tiles found** (11.5% of target)
- **~7,791 tiles missing** (88.5% of target)
- **390–615 tiles recoverable** from 16 unscanned plato-* repos
- **2,000–4,000 tiles recoverable** from 361 ternary-* repos
- **370–850 tiles recoverable** from 28 fleet/superinstance repos
- **1,000–2,000 tiles synthesizable** via LLM-assisted generation

The most efficient path to 8,800 tiles is a **phased approach**: quick wins from room configs and experiment data, deep scan of plato-* repos, fleet-wide ternary-* crawl, and LLM synthesis to fill remaining gaps.

The architecture is ready. The data is the fuel.
