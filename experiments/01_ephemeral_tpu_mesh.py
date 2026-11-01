"""Experiment 01: Ephemeral Cloud TPU v5e 2D Device Mesh & Sharding Topology.

Configures a 2D ('data', 'model') XLA device mesh using jax.sharding.Mesh,
PartitionSpec, and NamedSharding. Runs identically on a 4-device local XLA host
simulation (--xla_force_host_platform_device_count=4) and a 4-chip or 8-chip
Cloud TPU v5e slice (v5litepod-4 / v5litepod-8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@dataclass(frozen=True)
class MeshShardingRules:
    """Canonical PartitionSpec mapping for 2D ('data', 'model') Gemma + LoRA training."""

    batch_tokens: PartitionSpec = PartitionSpec("data", None)
    embed_vocab_by_dim: PartitionSpec = PartitionSpec(None, "model")
    attn_qkv_proj: PartitionSpec = PartitionSpec(None, "model")
    attn_out_proj: PartitionSpec = PartitionSpec("model", None)
    mlp_gate_up_proj: PartitionSpec = PartitionSpec(None, "model")
    mlp_down_proj: PartitionSpec = PartitionSpec("model", None)
    rmsnorm_scale: PartitionSpec = PartitionSpec(None)
    lora_a: PartitionSpec = PartitionSpec(None, None)
    lora_b: PartitionSpec = PartitionSpec(None, "model")


def create_ephemeral_mesh(
    data_axis_size: int | None = None,
    model_axis_size: int | None = None,
) -> Mesh:
    """Builds a 2D ('data', 'model') XLA Mesh across available devices."""
    devices = jax.devices()
    num_devices = len(devices)

    if data_axis_size is None or model_axis_size is None:
        if num_devices >= 8 and num_devices % 4 == 0:
            data_axis_size = num_devices // 2
            model_axis_size = 2
        elif num_devices >= 4 and num_devices % 2 == 0:
            data_axis_size = num_devices // 2
            model_axis_size = 2
        elif num_devices == 2:
            data_axis_size = 1
            model_axis_size = 2
        else:
            data_axis_size = 1
            model_axis_size = 1

    if data_axis_size * model_axis_size != num_devices:
        raise ValueError(
            f"Invalid mesh shape ({data_axis_size}, {model_axis_size}) "
            f"for {num_devices} available XLA devices."
        )

    device_grid = np.array(devices).reshape(data_axis_size, model_axis_size)
    return Mesh(device_grid, axis_names=("data", "model"))


def inspect_mesh_topology(
    batch_size: int = 8,
    seq_len: int = 64,
    embed_dim: int = 256,
    mlp_dim: int = 1024,
) -> dict[str, Any]:
    """Allocates sharded batch and weight tensors on the 2D mesh and inspects shards."""
    mesh = create_ephemeral_mesh()
    rules = MeshShardingRules()

    batch_sharding = NamedSharding(mesh, rules.batch_tokens)
    qkv_sharding = NamedSharding(mesh, rules.attn_qkv_proj)
    down_sharding = NamedSharding(mesh, rules.mlp_down_proj)
    replicated_sharding = NamedSharding(mesh, rules.rmsnorm_scale)

    tokens = jax.device_put(
        jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape(batch_size, seq_len),
        batch_sharding,
    )
    q_kernel = jax.device_put(
        jnp.ones((embed_dim, embed_dim), dtype=jnp.float32),
        qkv_sharding,
    )
    down_kernel = jax.device_put(
        jnp.ones((mlp_dim, embed_dim), dtype=jnp.float32),
        down_sharding,
    )
    norm_scale = jax.device_put(
        jnp.ones((embed_dim,), dtype=jnp.float32),
        replicated_sharding,
    )

    token_shard_shape = tokens.addressable_shards[0].data.shape
    q_shard_shape = q_kernel.addressable_shards[0].data.shape
    down_shard_shape = down_kernel.addressable_shards[0].data.shape
    norm_shard_shape = norm_scale.addressable_shards[0].data.shape

    q_full_bytes = q_kernel.nbytes
    q_shard_bytes = q_kernel.addressable_shards[0].data.nbytes

    summary = {
        "backend": jax.default_backend(),
        "num_devices": len(jax.devices()),
        "mesh_shape": dict(mesh.shape),
        "axis_names": list(mesh.axis_names),
        "tokens_global_shape": tuple(tokens.shape),
        "tokens_shard_shape": tuple(token_shard_shape),
        "q_kernel_global_shape": tuple(q_kernel.shape),
        "q_kernel_shard_shape": tuple(q_shard_shape),
        "down_kernel_global_shape": tuple(down_kernel.shape),
        "down_kernel_shard_shape": tuple(down_shard_shape),
        "norm_scale_shard_shape": tuple(norm_shard_shape),
        "q_kernel_full_bytes": int(q_full_bytes),
        "q_kernel_shard_bytes": int(q_shard_bytes),
        "per_device_weight_compression_ratio": float(q_full_bytes / q_shard_bytes),
    }
    return summary


def main() -> None:
    info = inspect_mesh_topology()
    print("=" * 72)
    print(" Experiment 01: Ephemeral 2D XLA Mesh ('data', 'model') Verification")
    print("=" * 72)
    print(f" XLA Backend               : {info['backend']}")
    print(f" Total XLA Devices         : {info['num_devices']}")
    print(f" 2D Mesh Topology          : {info['mesh_shape']}")
    print(
        f" Batch Tokens Shape        : global={info['tokens_global_shape']} "
        f"-> per-device shard={info['tokens_shard_shape']} P('data', None)"
    )
    print(
        f" Q-Proj Kernel Shape       : global={info['q_kernel_global_shape']} "
        f"-> per-device shard={info['q_kernel_shard_shape']} P(None, 'model')"
    )
    print(
        f" MLP Down Kernel Shape     : global={info['down_kernel_global_shape']} "
        f"-> per-device shard={info['down_kernel_shard_shape']} P('model', None)"
    )
    print(
        f" Per-Device Weight Memory  : {info['q_kernel_shard_bytes']} B / "
        f"{info['q_kernel_full_bytes']} B "
        f"({info['per_device_weight_compression_ratio']:.1f}x reduction along model axis)"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
