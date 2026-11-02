"""Experiment 03: Tunix-Style Surgical LoRA Injection & Base Parameter Freezing.

Wraps target Gemma linear projections in Flax NNX with `nnx.LoRA`, separating
trainable `nnx.LoRAParam` low-rank adapters (A, B) from frozen base `nnx.Param`
weights using `nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))`. Verifies:
  1. >90% reduction in trainable parameters.
  2. Bit-for-bit preservation (max_abs_diff == 0.0) of all base weights after an
     Optax AdamW gradient update step via `nnx.Optimizer(..., wrt=nnx.LoRAParam)`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name,
        Path(__file__).resolve().parent / filename,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inject_lora_adapters(
    model: Any,
    lora_rank: int = 4,
    *,
    include_mlp: bool = True,
    rngs: nnx.Rngs,
) -> Any:
    """Surgically replaces target Linear modules with nnx.LoRA wrappers in-place."""
    for block in model.blocks:
        attn = block.attn
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base_linear: nnx.Linear = getattr(attn, proj_name)
            lora_wrapped = nnx.LoRA(
                in_features=base_linear.in_features,
                lora_rank=lora_rank,
                out_features=base_linear.out_features,
                base_module=base_linear,
                rngs=rngs,
            )
            setattr(attn, proj_name, lora_wrapped)

        if include_mlp:
            mlp = block.mlp
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                base_linear = getattr(mlp, proj_name)
                lora_wrapped = nnx.LoRA(
                    in_features=base_linear.in_features,
                    lora_rank=lora_rank,
                    out_features=base_linear.out_features,
                    base_module=base_linear,
                    rngs=rngs,
                )
                setattr(mlp, proj_name, lora_wrapped)

    return model


def count_lora_vs_base_parameters(model: Any) -> dict[str, float | int]:
    """Splits model state into LoRAParam vs frozen base Param and counts scalars."""
    base_filter = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))
    _, lora_state, base_state = nnx.split(model, nnx.LoRAParam, base_filter)

    lora_params = sum(x.size for x in jax.tree.leaves(lora_state))
    base_params = sum(x.size for x in jax.tree.leaves(base_state))
    total_params = lora_params + base_params
    trainable_ratio_pct = (lora_params / total_params) * 100.0
    reduction_pct = (1.0 - (lora_params / total_params)) * 100.0

    return {
        "base_params": int(base_params),
        "lora_params": int(lora_params),
        "total_params": int(total_params),
        "trainable_ratio_pct": float(trainable_ratio_pct),
        "reduction_pct": float(reduction_pct),
    }


def verify_surgical_lora_step(lora_rank: int = 4) -> dict[str, Any]:
    """Runs a single LoRA training step and verifies base weights remain bit-for-bit frozen."""
    mesh_mod = _load_module("ephemeral_tpu_mesh", "01_ephemeral_tpu_mesh.py")
    gemma_mod = _load_module("gemma_nnx_architecture", "02_gemma_nnx_architecture.py")

    mesh = mesh_mod.create_ephemeral_mesh()
    config = gemma_mod.GemmaConfig()
    model = gemma_mod.GemmaForCausalLM(config, rngs=nnx.Rngs(7))
    model = inject_lora_adapters(model, lora_rank=lora_rank, include_mlp=True, rngs=nnx.Rngs(13))
    model = gemma_mod.shard_gemma_model(model, mesh)

    counts = count_lora_vs_base_parameters(model)

    base_filter = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))
    _, lora_before, base_before = nnx.split(model, nnx.LoRAParam, base_filter)
    base_leaves_before = [jnp.copy(x) for x in jax.tree.leaves(base_before)]
    lora_leaves_before = [jnp.copy(x) for x in jax.tree.leaves(lora_before)]

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=3e-3), wrt=nnx.LoRAParam)

    tokens = jnp.array(
        [
            [2, 10, 14, 18, 22, 3, 12, 25],
            [2, 11, 15, 19, 23, 3, 13, 26],
            [2, 10, 16, 20, 24, 3, 12, 27],
            [2, 11, 17, 21, 22, 3, 13, 28],
        ],
        dtype=jnp.int32,
    )

    def loss_fn(mdl: Any, batch: jax.Array) -> jax.Array:
        logits = mdl(batch[:, :-1])
        targets = batch[:, 1:]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_lp = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
        return -jnp.mean(target_lp)

    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam))
    loss_before, grads = grad_fn(model, tokens)
    optimizer.update(model, grads)
    loss_after = loss_fn(model, tokens)

    _, lora_after, base_after = nnx.split(model, nnx.LoRAParam, base_filter)
    base_leaves_after = jax.tree.leaves(base_after)
    lora_leaves_after = jax.tree.leaves(lora_after)

    max_base_diff = max(
        float(jnp.max(jnp.abs(a - b))) for a, b in zip(base_leaves_before, base_leaves_after)
    )
    max_lora_diff = max(
        float(jnp.max(jnp.abs(a - b))) for a, b in zip(lora_leaves_before, lora_leaves_after)
    )

    return {
        **counts,
        "lora_rank": lora_rank,
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "max_base_weight_diff": max_base_diff,
        "max_lora_weight_diff": max_lora_diff,
        "base_weights_frozen_bitwise": max_base_diff == 0.0,
        "lora_weights_updated": max_lora_diff > 0.0,
    }


def main() -> None:
    res = verify_surgical_lora_step(lora_rank=4)
    print("=" * 72)
    print(" Experiment 03: Surgical LoRA Injection & State Filtering (Flax NNX)")
    print("=" * 72)
    print(f" Frozen Base Parameters    : {res['base_params']:,}")
    print(f" Trainable LoRA Parameters : {res['lora_params']:,} (rank={res['lora_rank']})")
    print(f" Total Parameters          : {res['total_params']:,}")
    print(
        f" Trainable Parameter Share : {res['trainable_ratio_pct']:.2f}% "
        f"({res['reduction_pct']:.2f}% of parameters frozen)"
    )
    print(f" Cross-Entropy Loss        : {res['loss_before']:.4f} -> {res['loss_after']:.4f}")
    print(
        f" Base Weight Max |Diff|    : {res['max_base_weight_diff']:.8f} "
        f"(frozen={res['base_weights_frozen_bitwise']})"
    )
    print(
        f" LoRA Adapter Max |Diff|   : {res['max_lora_weight_diff']:.6f} "
        f"(updated={res['lora_weights_updated']})"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
