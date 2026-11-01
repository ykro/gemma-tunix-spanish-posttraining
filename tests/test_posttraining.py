"""Pytest suite for gemma-tunix-spanish-posttraining."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ephemeral_2d_mesh_sharding() -> None:
    exp01 = _load("exp01", "experiments/01_ephemeral_tpu_mesh.py")
    stats = exp01.inspect_mesh_topology(batch_size=8, seq_len=32, embed_dim=128, mlp_dim=512)
    assert stats["num_devices"] >= 1
    if stats["num_devices"] == 4:
        assert stats["mesh_shape"] == {"data": 2, "model": 2}
        assert stats["tokens_shard_shape"] == (4, 32)
        assert stats["q_kernel_shard_shape"] == (128, 64)
        assert stats["down_kernel_shard_shape"] == (256, 128)
        assert stats["per_device_weight_compression_ratio"] == 2.0


def test_gemma_nnx_architecture_and_softcap() -> None:
    exp02 = _load("exp02", "experiments/02_gemma_nnx_architecture.py")
    stats = exp02.verify_gemma_architecture()
    assert stats["total_params"] > 3_000_000
    assert stats["gqa_ratio"] == 2
    assert stats["logits_shape"] == (4, 16, 512)
    assert stats["logits_max_abs"] <= stats["softcap_bound"]


def test_surgical_lora_freezes_base_weights_bitwise() -> None:
    exp03 = _load("exp03", "experiments/03_lora_surgical_injection.py")
    res = exp03.verify_surgical_lora_step(lora_rank=4)
    assert res["reduction_pct"] >= 90.0
    assert res["base_weights_frozen_bitwise"] is True
    assert res["max_base_weight_diff"] == 0.0
    assert res["lora_weights_updated"] is True
    assert res["loss_after"] < res["loss_before"]


def test_grpo_group_advantage_normalization_and_alignment() -> None:
    exp04 = _load("exp04", "experiments/04_grpo_alignment_loop.py")
    rewards = jnp.array([[0.2, 0.6, 1.4, 1.8]], dtype=jnp.float32)
    adv = exp04.compute_group_normalized_advantages(rewards)
    assert jnp.abs(jnp.mean(adv)) < 1e-5
    assert jnp.abs(jnp.std(adv) - 1.0) < 1e-3

    res = exp04.run_grpo_alignment(num_steps=24, group_size=4, lora_rank=4)
    assert res["max_base_weight_diff"] == 0.0
    assert res["final_eval"]["mean_reward"] > res["initial_eval"]["mean_reward"]
    assert (
        res["final_eval"]["format_compliance_rate"]
        > res["initial_eval"]["format_compliance_rate"]
    )
