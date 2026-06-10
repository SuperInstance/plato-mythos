# PLATO Tile Dataset — Summary

**Generated:** 2026-06-10  
**Dataset:** `tiles.jsonl`  
**Purpose:** OpenMythos fine-tuning

---

## Overview

| Metric | Value |
|--------|-------|
| **Total unique tiles** | 1,009 |
| **Raw inputs processed** | 1,943 |
| **Duplicates removed** | 934 |
| **Unique domains** | 37 |
| **Tiles with confidence** | 396 / 1,009 (39.3%) |
| **Mean confidence** | 0.52 |

---

## Domains / Rooms Covered

| Domain | Tiles | Notes |
|--------|-------|-------|
| research | 490 | Largest domain; research summaries & analysis |
| Knowledge | 136 | Training-pairs with deadband protocol |
| Constraint | 64 | Constraint-based reasoning tiles |
| fleet-operations | 56 | Fleet management, captain's logs |
| dojo | 45 | Arena competition transcripts |
| Documentation | 39 | README & docs generation |
| Communication | 24 | Bottle messages, oracle comms |
| FleetHealth | 17 | Fleet health reports & audits |
| CodeArchaeology | 16 | Codebase analysis tiles |
| Memory | 16 | Memory & boot timeline |
| Research (lowercase) | 13 | Research room metadata |
| achievements | 12 | Achievement unlock data |
| gc-decisions | 12 | Garbage collection decision pairs |
| Integration | 8 | Integration test results |
| Prototyping | 7 | Maze/prototype experiments |
| Other (22 domains) | ~64 | Various small rooms |

---

## Confidence Distribution

| Range | Count | Percentage |
|-------|-------|-----------|
| 0.9–1.0 | 55 | 5.5% |
| 0.7–0.9 | 12 | 1.2% |
| 0.5–0.7 | 329 | 32.6% |
| 0.3–0.5 | 0 | 0% |
| 0.0–0.3 | 0 | 0% |
| No confidence | 613 | 60.7% |

**Note:** The majority of tiles (60.7%) have no confidence score. Most scored tiles cluster at 0.5 (default for training pairs and zeroclaw tiles). High-confidence tiles (0.9+) come from curated Q&A tiles and catalog metadata.

---

## Data Sources

| Source File | Unique Tiles |
|-------------|-------------|
| all-research.jsonl | 464 |
| training-pairs.jsonl | 200 |
| zeroclaw-tiles | 160 |
| all-operations.jsonl | 37 |
| all-dojo.jsonl | 24 |
| arena-competition.jsonl | 21 |
| npc-dialogue.jsonl | 18 |
| catalog.json (rooms) | 15 |
| all-achievements.jsonl | 12 |
| gc-decisions.jsonl | 12 |
| core-vision.jsonl | 10 |
| data/tiles.jsonl | 9 |
| enriched-fleet.jsonl | 8 |
| fleet-knowledge.jsonl | 6 |
| knowledge syntheses | 12 |
| curriculum.jsonl | 1 |

---

## Gaps & Missing Data

### ⚠️ Significant Gaps

1. **8,800+ tiles expected but only 1,009 found.** The design doc references 8,800+ tiles across 913 rooms. The `catalog.json` reports 8,316 source tiles, but most exist only as metadata (counts/keywords) — the actual tile content was never persisted to disk in extractable form, or was generated ephemerally during agent sessions.

2. **913 rooms expected, 37 found.** The `plato-matrix-bridge.json` lists ~60 rooms, but most contain only tile counts, not actual tile content. The catalog indexes 15 rooms. The remaining ~900 rooms from the design spec are not represented in the filesystem.

3. **Confidence scores sparse.** 60.7% of tiles have no confidence score. The default 0.5 score on most training pairs may not reflect actual quality.

4. **tile_buffers are test stubs.** The 108 tiles in `tile_buffers/` have empty `state`, `action`, and `outcome` fields — they're framework test fixtures, not real knowledge tiles.

5. **Zeroclaw tiles are thought fragments.** Many zeroclaw tiles are mid-reasoning thoughts rather than complete Q&A pairs.

6. **No room-level granular data.** The knowledge syntheses are room-level summaries, not individual tile Q&A pairs.

### 📋 Recommendations

- **Regenerate tiles from source repos:** Use the PLATO tile engine to extract tiles from all 452+ fleet repos
- **Harvest live PLATO server:** If the PLATO server has tiles in memory, export them
- **Add confidence calibration:** Score all tiles through a quality model before fine-tuning
- **Normalize domains:** Many domains appear in both Title Case and lowercase (e.g., "Research" vs "research")
- **Filter training pairs:** Remove very short or repetitive training-pairs entries

---

## Dataset Format

Each line in `tiles.jsonl` follows the OpenMythos fine-tuning format:

```json
{
  "messages": [
    {"role": "user", "content": "<question>"},
    {"role": "assistant", "content": "<answer>"}
  ],
  "domain": "<room-name>",
  "confidence": 0.85
}
```

Optional fields: `confidence`, `tags`
