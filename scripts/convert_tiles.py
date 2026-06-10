#!/usr/bin/env python3
"""
PLATO Tile Dataset Converter for OpenMythos Fine-tuning.
Collects all tile data across workspace and outputs unified JSONL.
"""
import json
import os
import hashlib
from pathlib import Path
from collections import defaultdict

BASE = "/home/phoenix/repos/oracle1-workspace"
OUTPUT_DIR = "/home/phoenix/repos/plato-mythos/data"

seen_hashes = set()
all_tiles = []
stats = defaultdict(lambda: 0)
domain_counts = defaultdict(int)
confidence_vals = []
gaps = []


def content_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]


def add_tile(question: str, answer: str, domain: str = "unknown",
             confidence: float = None, tags: list = None, source: str = ""):
    if not question or not answer or len(question.strip()) < 3 or len(answer.strip()) < 5:
        gaps.append(f"Empty/short from {source}")
        return
    stats["total_raw"] += 1
    h = content_hash(question)
    if h in seen_hashes:
        stats["duplicates"] += 1
        return
    seen_hashes.add(h)

    tile = {
        "messages": [
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ],
        "domain": domain,
    }
    if confidence is not None:
        tile["confidence"] = round(float(confidence), 2)
    if tags:
        tile["tags"] = tags

    all_tiles.append(tile)
    stats["total_output"] += 1
    stats[f"src:{source}"] += 1
    domain_counts[domain] += 1
    if confidence is not None:
        confidence_vals.append(float(confidence))


# === Source 1: data/tiles.jsonl ===
def src1_tiles_jsonl():
    p = os.path.join(BASE, "data/tiles.jsonl")
    if not os.path.exists(p): return
    with open(p) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            add_tile(d.get("question",""), d.get("answer",""),
                     d.get("domain","unknown"), d.get("confidence"),
                     d.get("tags",[]), "data/tiles.jsonl")


# === Source 2: training-pairs.jsonl ===
def src2_training_pairs():
    p = os.path.join(BASE, "training-data/training-pairs.jsonl")
    if not os.path.exists(p): return
    with open(p) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            msgs = d.get("messages", [])
            meta = d.get("metadata", {})
            user_c = "\n".join(m["content"] for m in msgs if m["role"] == "user")
            asst_c = "\n".join(m["content"] for m in msgs if m["role"] == "assistant")
            if user_c.strip() and asst_c.strip():
                tags = []
                if meta.get("deadband_tier"):
                    tags.append(meta["deadband_tier"])
                add_tile(user_c.strip(), asst_c.strip(),
                         meta.get("domain","training-pairs"),
                         meta.get("confidence"), tags, "training-pairs.jsonl")


# === Source 3: zeroclaw-tiles ===
def src3_zeroclaw():
    p = os.path.join(BASE, "training-data/zeroclaw-tiles/all-tiles.jsonl")
    if not os.path.exists(p): return
    with open(p) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            add_tile(d.get("question",""), d.get("answer",""),
                     d.get("domain","zeroclaw"), d.get("confidence"),
                     d.get("tags",[]), "zeroclaw-tiles")


# === Sources 4-7: instruction-format JSONL files ===
def src_instruction_jsonl(filepath: str, default_domain: str):
    if not os.path.exists(filepath): return
    src = os.path.basename(filepath)
    with open(filepath) as f:
        for line in f:
            if not line.strip(): continue
            try:
                d = json.loads(line)
            except: continue
            instr = d.get("instruction", "")
            inp = d.get("input", "")
            out = d.get("output", "")
            q = instr
            if inp:
                ctx = inp if isinstance(inp, str) else json.dumps(inp)
                q = f"{instr}\n\nContext: {ctx}" if instr else ctx
            meta = d.get("metadata", {})
            dom = meta.get("domain", default_domain)
            add_tile(q, str(out) if out else "", dom,
                     meta.get("confidence"), meta.get("tags",[]), src)


# === Source 8: gc-decisions ===
def src8_gc():
    p = os.path.join(BASE, "training-data/gc-decisions.jsonl")
    if not os.path.exists(p): return
    with open(p) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            inp = d.get("input", {})
            out = d.get("output", {})
            q = f"Given a {inp.get('file_type','?')} file ({inp.get('file_size_kb',0)}KB, disk {inp.get('disk_pressure_pct',0)}%), decide GC action."
            a = f"Action: {out.get('action','?')}. Tiles: {out.get('tiles_produced',0)}. Freed: {out.get('bytes_freed',0)}B. Quality: {out.get('quality_score',0)}"
            add_tile(q, a, "gc-decisions", out.get("quality_score"),
                     ["gc","decision"], "gc-decisions.jsonl")


# === Sources 9-10: knowledge & ensign syntheses ===
def src_syntheses():
    for subdir, tag in [("training-data/knowledge","knowledge"), ("training-data/ensigns","ensign")]:
        dp = os.path.join(BASE, subdir)
        if not os.path.isdir(dp): continue
        for fn in os.listdir(dp):
            if not fn.endswith(".json"): continue
            room = fn.replace(".json","").replace("_ensign","")
            try:
                with open(os.path.join(dp, fn)) as f:
                    data = json.load(f)
            except: continue
            if not isinstance(data, list): data = [data]
            for item in data:
                syn = item.get("synthesis","")
                if isinstance(syn, str):
                    try: syn = json.loads(syn)
                    except: pass
                if isinstance(syn, dict):
                    a = f"Summary: {syn.get('summary','')}"
                    pr = syn.get("principles",[])
                    if pr:
                        a += "\n\nPrinciples:\n" + "\n".join(f"- {p}" for p in pr)
                else:
                    a = str(syn)
                add_tile(f"What was learned in the {room} room? (group: {item.get('group','?')})",
                         a, room, None, [tag, "synthesis"], f"{subdir}/{fn}")


# === Source 11: tile_buffers ===
def src_tile_buffers():
    bp = os.path.join(BASE, "tile_buffers")
    if not os.path.isdir(bp): return
    for rd in os.listdir(bp):
        rp = os.path.join(bp, rd)
        if not os.path.isdir(rp): continue
        for fn in os.listdir(rp):
            if not fn.endswith(".json"): continue
            try:
                with open(os.path.join(rp, fn)) as f:
                    d = json.load(f)
            except: continue
            room_id = d.get("room_id", rd)
            st, act, out = d.get("state",""), d.get("action",""), d.get("outcome","")
            if st and out:
                add_tile(f"In room '{room_id}', given state '{st}', what happened?",
                         f"Action: {act}. Outcome: {out}", f"test-{room_id}",
                         d.get("reward",0), ["test-tile"], f"tile_buffers/{rd}")
            elif st:
                add_tile(f"What is the state of room '{room_id}'?",
                         f"State: {st}. Reward: {d.get('reward',0)}", f"test-{room_id}",
                         d.get("reward",0), ["test-tile"], f"tile_buffers/{rd}")


# === Source 12: catalog room metadata ===
def src_catalog():
    p = os.path.join(BASE, "artifacts/index/catalog.json")
    if not os.path.exists(p): return
    with open(p) as f:
        data = json.load(f)
    for name, rd in data.get("rooms",{}).items():
        tc = rd.get("tiles",0)
        kw = rd.get("keywords",[])
        fi = rd.get("files",{})
        add_tile(f"What does the PLATO room '{name}' contain?",
                 f"Room '{name}' has {tc} tiles. Topics: {', '.join(kw[:10])}. Files: {json.dumps(fi)}.",
                 name, 0.9, ["catalog","room-metadata"]+kw[:3], "catalog.json")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Processing data/tiles.jsonl...")
    src1_tiles_jsonl()
    print("Processing training-pairs.jsonl...")
    src2_training_pairs()
    print("Processing zeroclaw-tiles...")
    src3_zeroclaw()

    # Research files
    rd = os.path.join(BASE, "training-data/research")
    for fn in sorted(os.listdir(rd)):
        if fn.endswith(".jsonl"):
            print(f"  Processing research/{fn}...")
            src_instruction_jsonl(os.path.join(rd, fn), "research")

    # Fleet operations
    od = os.path.join(BASE, "training-data/fleet-operations")
    for fn in sorted(os.listdir(od)):
        if fn.endswith(".jsonl"):
            print(f"  Processing fleet-operations/{fn}...")
            src_instruction_jsonl(os.path.join(od, fn), "fleet-operations")

    # Achievements
    ad = os.path.join(BASE, "training-data/achievements")
    for fn in sorted(os.listdir(ad)):
        if fn.endswith(".jsonl"):
            print(f"  Processing achievements/{fn}...")
            src_instruction_jsonl(os.path.join(ad, fn), "achievements")

    # Dojo
    dd = os.path.join(BASE, "training-data/dojo-transcripts")
    for fn in sorted(os.listdir(dd)):
        if fn.endswith(".jsonl"):
            print(f"  Processing dojo/{fn}...")
            src_instruction_jsonl(os.path.join(dd, fn), "dojo")

    print("Processing gc-decisions...")
    src8_gc()
    print("Processing syntheses...")
    src_syntheses()
    print("Processing tile_buffers...")
    src_tile_buffers()
    print("Processing catalog...")
    src_catalog()

    # Write output
    out_path = os.path.join(OUTPUT_DIR, "tiles.jsonl")
    with open(out_path, "w") as f:
        for tile in all_tiles:
            f.write(json.dumps(tile, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote {len(all_tiles)} tiles to {out_path}")
    print(f"   Raw inputs: {stats['total_raw']}, Duplicates removed: {stats['duplicates']}")

    # Save stats for summary
    summary_data = {
        "total_output": stats["total_output"],
        "total_raw": stats["total_raw"],
        "duplicates": stats["duplicates"],
        "domains": dict(domain_counts),
        "confidence_vals": confidence_vals,
        "gaps": gaps[:20],
        "sources": {k: v for k, v in stats.items() if k.startswith("src:")},
    }
    with open(os.path.join(OUTPUT_DIR, "_stats.json"), "w") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
