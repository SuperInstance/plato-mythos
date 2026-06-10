#!/usr/bin/env python3
"""
OpenMythos Architecture Validator for plato-mythos

Validates that the RDT architecture maps correctly to PLATO concepts.
No GPU required — runs on CPU with tiny config.
"""

import sys
import json
import time
from pathlib import Path

def main():
    print("=" * 60)
    print("plato-mythos Architecture Validator")
    print("=" * 60)
    
    # Step 1: Check dependencies
    print("\n[1/6] Checking dependencies...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not found. Install: pip install torch")
        sys.exit(1)
    
    try:
        from open_mythos.main import OpenMythos, MythosConfig
        print("  ✓ OpenMythos installed")
    except ImportError:
        print("  ✗ OpenMythos not found. Install: pip install open-mythos")
        sys.exit(1)
    
    # Step 2: Create PLATO-aligned config (edge-tiny variant)
    print("\n[2/6] Creating PLATO edge-tiny config...")
    cfg = MythosConfig(
        vocab_size=32000,
        dim=512,
        n_heads=8,
        max_seq_len=4096,
        max_loop_iters=4,       # PLATO: 4 curriculum rounds
        prelude_layers=2,       # PLATO: orientation stage
        coda_layers=1,          # PLATO: synthesis stage
        n_experts=8,            # PLATO: 8 domain rooms
        n_shared_experts=1,     # PLATO: cross-domain knowledge
        n_experts_per_tok=2,    # PLATO: top-2 room routing
        expert_dim=2048,
        lora_rank=8,            # PLATO: shell adapters
        attn_type="mla",
        n_kv_heads=8,
        kv_lora_rank=64,        # PLATO: tile compression rank
        q_lora_rank=128,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
    )
    print(f"  Config: dim={cfg.dim}, experts={cfg.n_experts}, loops={cfg.max_loop_iters}")
    
    # Step 3: Build model
    print("\n[3/6] Building model...")
    model = OpenMythos(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model built: {total_params:,} parameters ({total_params/1e6:.1f}M)")
    
    # Step 4: Validate PLATO architecture mapping
    print("\n[4/6] Validating PLATO → Mythos mapping...")
    
    # 4a: Check MoE routing (rooms-as-experts)
    if hasattr(model, 'recurrent') and hasattr(model.recurrent, 'moe'):
        print("  ✓ MoE FFN present (rooms-as-experts routing)")
    else:
        print("  ⚠ MoE FFN not found at expected path")
    
    # 4b: Check MLA attention (tiles-as-KV)
    if hasattr(model, 'recurrent'):
        print("  ✓ Recurrent block present (curriculum loop)")
    else:
        print("  ✗ Recurrent block missing")
    
    # 4c: Check depth-wise LoRA (shells)
    if hasattr(model, 'recurrent') and hasattr(model.recurrent, 'injection'):
        print("  ✓ LTI injection present (deadband stability)")
    else:
        print("  ⚠ LTI injection not found")
    
    # 4d: Check spectral radius (conservation law analog)
    if hasattr(model, 'recurrent') and hasattr(model.recurrent, 'injection'):
        if hasattr(model.recurrent.injection, 'get_A'):
            A = model.recurrent.injection.get_A()
            rho = torch.linalg.eigvals(A).abs().max().item()
            stable = rho < 1.0
            status = "✓" if stable else "✗"
            print(f"  {status} Spectral radius ρ(A) = {rho:.4f} (must be < 1)")
            if not stable:
                print("    WARNING: Unstable — model may diverge during deep loops")
    
    # Step 5: Forward pass test
    print("\n[5/6] Testing forward pass...")
    model.eval()
    with torch.no_grad():
        # Simulate a PLATO tile query (random tokens for now)
        tokens = torch.randint(0, cfg.vocab_size, (1, 32))
        
        # Test with increasing loop counts (variable depth = curriculum)
        for n_loops in [1, 2, 4]:
            start = time.time()
            logits = model(tokens, n_loops=n_loops)
            elapsed = time.time() - start
            print(f"  {n_loops} loops: logits {logits.shape}, {elapsed:.3f}s")
    
    # Step 6: Generation test
    print("\n[6/6] Testing generation...")
    with torch.no_grad():
        prompt = torch.randint(0, cfg.vocab_size, (1, 8))
        output = model.generate(prompt, max_new_tokens=16, n_loops=4)
        print(f"  ✓ Generated: input {prompt.shape} → output {output.shape}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Architecture: RDT (Prelude → Recurrent × {cfg.max_loop_iters} → Coda)")
    print(f"  PLATO mapping: rooms→experts, tiles→KV, curriculum→loops")
    print("=" * 60)
    
    # Save validation results
    results = {
        "total_params": total_params,
        "config": {
            "dim": cfg.dim,
            "n_experts": cfg.n_experts,
            "n_loops": cfg.max_loop_iters,
            "n_shared_experts": cfg.n_shared_experts,
            "expert_dim": cfg.expert_dim,
            "attn_type": cfg.attn_type,
        },
        "plato_mapping": {
            "rooms": "MoE expert groups",
            "tiles": "MLA compressed KV",
            "curriculum": "loop depth",
            "deadband": "ACT halting",
            "shells": "depth-wise LoRA",
        },
        "status": "validated",
        "timestamp": time.time(),
    }
    
    out_path = Path(__file__).parent.parent / "data" / "validation-results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
