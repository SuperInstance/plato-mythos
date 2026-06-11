# PLATO Architecture Summary

> PLATO is the model. Rooms are experts. Tiles are knowledge. Curriculum is depth.

## 1. Room Model Architecture

### Core Concept
A **Room** is a self-contained unit of sensor/actuator interaction — the atomic building block of the Plato Matrix. Rooms model physical spaces (engine rooms, server racks, greenhouses) or virtual spaces (game levels, process namespaces).

### Room Lifecycle
1. **Tick**: configurable Hz rate, reads all sensors each tick
2. **Snapshot**: captures current sensor values into a tick record
3. **History**: rolling circular buffer of N tick snapshots
4. **Alarm evaluation**: threshold rules evaluated against sensor data with cooldown semantics
5. **Stream**: live updates broadcast to subscribed TCP clients

### RoomDepth Enum (Five Depth Levels)
From the `plato-runtime-kernel`:

| Depth | Name | Analogy | What lives here |
|-------|---------|-------------|----------------------------------------------|
| 0 | Floor | Dance floor | Agents, humans, autonomous behavior |
| 1 | Board | DJ board | Instruments, tools, control surfaces |
| 2 | Panel | Instrument panel | Settings, presets, configurations |
| 3 | Code | Code editor | Functions, algorithms, logic |
| 4 | Metal | Transistors | Raw bits, hardware registers, firmware |

A room can zoom between depths. The engine room at Floor depth shows live gauges; at Metal depth it shows register values.

### Physical vs Virtual Rooms
- **Physical rooms**: backed by real hardware (ESP32, Raspberry Pi) with real sensors (temperature, humidity, pressure, motion)
- **Virtual rooms**: software-defined spaces (game worlds, process monitors, Kubernetes namespaces)
- Both use identical Room contracts (ROOM.json / room.json) and the same engine block runtime

### Tensor Grid Model
Rooms are organized into a tensor grid in the runtime kernel. Each room is a cell with a grid position (e.g., A1=Engine Room, B2=Bilge). Agents are **Batons** — immutable execution state passing between cells.

### RoomIdentity
Each room has a spatial identity: `room_id`, tensor hash, grid position, and depth level.

### RoomContract
The ROOM.json schema defining a room's borders, topology, and runtime assets.

### RoomTopology
Defines parent room, adjacent rooms, and traversal history with weights.

## 2. Agent Communication Protocol

### Bottles
Bottles are the message abstraction in Plato's agent communication. An agent composes a **bottle** — a typed, sealed message — and dispatches it to a room or fleet. The bottle pattern ensures messages are:
- **Immutable** once sealed (no tampering in transit)
- **Typed** (the protocol verifies the message type matches the destination)
- **Traceable** (each bottle carries origin and sequence metadata)

### Vessels
Vessels are the transport containers. While a bottle is the message content, the vessel is the envelope — it handles serialization, addressing (which room, which agent), and delivery guarantees (best-effort vs. confirmed). A vessel can carry multiple bottles in batch.

### Dispatch
The dispatch layer routes bottles/vessels to their destinations:
- **Local dispatch**: direct in-process call (same machine, same runtime)
- **TCP dispatch**: text-based protocol between engine blocks (`tick | history | actuator | alarm | subscribe`)
- **MQTT dispatch**: for distributed deployments across network boundaries
- **PGAS dispatch** (Chapel): transparent remote execution via `coforall` over locales

### Protocol Commands
The text protocol (`plato_handle_command`) supports:
- `tick` — request current sensor snapshot
- `history` — retrieve historical tick data
- `actuator` — read/write actuator state
- `alarm` — query/manage alarm rules
- `subscribe` — register for streaming updates

## 3. Conservation Law Implementation

### The Ternary Conservation Principle
In PLATO, sensor readings are reduced to ternary values `{-1, 0, +1}` (trits). The **conservation law** states that in a stable system, the sum of all ternary values across the fleet tends toward zero — anomalies balance out over time.

### How It's Implemented
1. **Threshold conversion** (`plato-ternary-bridge`): each sensor reading is converted via `to_trit(value, {low, high})` → `-1`, `0`, or `+1`
2. **Ternary room state**: 8 sensors → 8 trits packed into a single u16 (2 bits per trit)
3. **Fleet voting** (`fleet_vote`): rooms cast `{-1, 0, +1}` votes, consensus computed in O(N)
4. **Alarm vectors**: count of non-zero trits indicates anomaly magnitude
5. **Delta compression**: history is compressed via delta encoding on trits — most ticks are all-zeros (normal), so deltas are tiny

### Comptime Packing (Zig)
Zig's comptime verifies ternary correctness at compile time — invalid trit values produce `@compileError`. 16 trits pack into a single `u32`.

### Fleet Groove Score
The `plato-music-sync` crate measures fleet alignment as a **groove score** (0.0=chaos, 1.0=perfect sync). This is the conservation law's observable — when groove is high, the system is balanced.

## 4. TutorLoop and Educational Patterns

### What is TutorLoop?
TutorLoop is PLATO's educational feedback cycle: **observe → assess → instruct → practice → verify**. It mirrors the alarm → action → resolve cadence in the engine blocks.

### Tile System
- A **Tile** is a Q&A knowledge unit: `{question, answer, category, difficulty}`
- Categories: architecture, api, protocol, distributed-systems, sensor, actuator, room-model, ternary-logic
- Difficulty: 1-5 scale
- Tiles are the training data for plato-mythos (the RDT fine-tuning system)

### Curriculum = Depth
PLATO's curriculum maps to the Recurrent-Depth Transformer loop depth:
- More loop iterations = deeper reasoning
- ACT halting (from the deadband concept) stops reasoning when confidence is high
- Different LoRA adapters per loop depth = different "shells" of understanding

### The Isomorphism (plato-mythos)
```
PLATO Room       → MoE Expert Group    (routing, not random gates)
PLATO Tile       → MLA Compressed KV   (knowledge = latent representation)
PLATO Curriculum → RDT Loop Depth      (more rounds = deeper reasoning)
PLATO Deadband   → ACT Halting         (stop when confident)
PLATO Shell      → Depth-wise LoRA     (same base, different adapter per loop)
```

### RDT Architecture
1. **Prelude**: standard transformer layers × N — agent learns room context
2. **Recurrent Block**: `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)` — MoE routing, LoRA adapters, ACT halting, looped T times
3. **Coda**: standard transformer layers × M — final synthesis and output

### Cadence Pattern
From the demo and music-sync crates, the **cadence** pattern models alarm resolution:
- `alarm → action → resolve` — the fundamental cycle
- Cadences are graded: PERFECT (clean resolution), IMPERFECT (delayed), BROKEN (failed)
- The cadence module tracks alarm-to-resolution timing across the fleet

### Multi-Language Implementations
The thesis is proven across implementations:
- **Rust** (plato-engine-block): production reference, zero-cost abstractions
- **C** (plato-engine-block-c): bare-metal, <400 lines C99, zero malloc after init
- **Elixir/OTP** (plato-engine-block-elixir): BEAM actor model, supervision trees, hot code reload
- **Gleam** (plato-engine-block-gleam): type-safe BEAM, compile-time trit verification
- **Zig** (plato-engine-block-zig): comptime ternary packing, no hidden control flow
- **Chapel** (plato-fleet-chapel): PGAS distributed execution, locales-as-rooms
- **Python** (plato-agent-python): LLM-ready agent framework, async streaming

### Fleet Orchestration
The `plato-fleet-manager` provides:
- **Room discovery**: finds engine blocks across the network
- **TickAggregator**: merges tick streams by timestamp, detects cross-room correlations
- **FleetMonitor**: health status (🟢/🟡/🔴), stale tick detection, aggregate alarm state
- **Escalation policies**: configurable rules for when to alert human operators

### Room Configuration System
`plato-room-configs` provides:
- Declarative `.room.json` files (sensors, actuators, alarms, metadata)
- JSON schemas for validation
- Rust crate API for programmatic loading/validation
- 18 room configs and 5 fleet manifests covering boats, data centers, smart homes, game worlds, factories

### Flux Compiler
`plato-flux-compiler` compiles alarm conditions (e.g., `"coolant_temp_c > 95"`) into **FLUX bytecode** for deterministic execution on any platform — ESP32 to GPU cluster, zero behavioral difference.
