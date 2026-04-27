"""Tests for PlatoMythos model, deadband halting, and shell switching."""

import pytest
import torch

from plato_mythos.config import PlatoMythosConfig
from plato_mythos.deadband_act import DeadbandACT
from plato_mythos.model import PlatoMythos
from plato_mythos.shell_lora import ShellLoRA


def test_config_creation():
    """Basic sanity check that the config dataclass works."""
    config = PlatoMythosConfig(
        vocab_size=128,
        d_model=64,
        num_heads=4,
        max_loop_depth=4,
        num_rooms=2,
        experts_per_room=2,
    )
    assert config.head_dim == 16
    assert config.total_experts == 4


def test_forward_pass_random_tiles():
    """A forward pass with random integer tiles must return logits."""
    config = PlatoMythosConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        num_layers=1,
        prelude_layers=1,
        coda_layers=1,
        max_loop_depth=3,
        num_rooms=2,
        experts_per_room=1,
        lora_r=4,
        deadband_threshold=1e-2,
    )
    model = PlatoMythos(config)
    model.eval()

    batch, seq = 2, 8
    tiles = torch.randint(0, config.vocab_size, (batch, seq))
    logits = model(tiles)

    assert logits.shape == (batch, seq, config.vocab_size)


def test_forward_pass_dict_tiles():
    """Dict-form tiles should return a dict with logits inside."""
    config = PlatoMythosConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        num_layers=1,
        prelude_layers=1,
        coda_layers=1,
        max_loop_depth=3,
        num_rooms=2,
        experts_per_room=1,
        lora_r=4,
    )
    model = PlatoMythos(config)
    model.eval()

    batch, seq = 1, 4
    tiles = {
        "token_ids": torch.randint(0, config.vocab_size, (batch, seq)),
        "domain_id": torch.zeros(batch, dtype=torch.long),
        "confidence": torch.ones(batch) * 0.9,
    }
    out = model(tiles, rooms={"domains": ["room_0"] * seq, "confidences": [0.9] * seq})

    assert isinstance(out, dict)
    assert out["token_ids"].shape == (batch, seq, config.vocab_size)


def test_deadband_halting():
    """DeadbandACT must halt when state deltas become negligible."""
    act = DeadbandACT(threshold=1e-3, max_steps=10)

    # Simulate converging states.
    state = torch.randn(1, 4, 32)
    act.reset(state)

    halted = False
    for step in range(100):
        # Converge quickly by making the state barely change.
        state = state + torch.randn_like(state) * 1e-5
        if not act.should_continue(state, step):
            halted = True
            break

    assert halted, "DeadbandACT should halt when state stops changing"
    assert step < act.max_steps


def test_deadband_priority_tiers():
    """get_priority must map metadata onto P0/P1/P2."""
    act = DeadbandACT()

    assert act.get_priority({"critical": True}) == "P0"
    assert act.get_priority({"confidence": 0.99}) == "P0"
    assert act.get_priority({"confidence": 0.85}) == "P1"
    assert act.get_priority({"confidence": 0.3}) == "P2"
    assert act.get_priority(None) == "P1"


def test_shell_switching():
    """ShellLoRA must produce different outputs for different shell_ids."""
    config = PlatoMythosConfig(
        d_model=32,
        lora_r=4,
        lora_alpha=8.0,
        max_loop_depth=4,
    )
    shell_lora = ShellLoRA(config)

    x = torch.randn(1, 2, 32)
    out_0 = shell_lora(x, shell_id=0)
    out_1 = shell_lora(x, shell_id=1)
    out_2 = shell_lora(x, shell_id=2)

    # Different shells should yield different residuals.
    assert not torch.allclose(out_0, out_1, atol=1e-6)
    assert not torch.allclose(out_1, out_2, atol=1e-6)

    # Shape sanity check.
    assert out_0.shape == x.shape


def test_shell_out_of_range():
    """Accessing an invalid shell_id must raise an IndexError."""
    config = PlatoMythosConfig(d_model=16, lora_r=2, max_loop_depth=2)
    shell_lora = ShellLoRA(config)
    x = torch.randn(1, 2, 16)

    with pytest.raises(IndexError):
        shell_lora(x, shell_id=99)
