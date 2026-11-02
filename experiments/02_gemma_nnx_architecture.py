"""Experiment 02: Gemma 2 style Architecture in Flax NNX with 2D Mesh Sharding.

Implements a Gemma 2 style causal Transformer (random weights, no checkpoint) in Flax NNX featuring:
  - Zero-centered GemmaRMSNorm: x * rsqrt(mean(x^2) + eps) * (1.0 + scale)
  - Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE)
  - Attention and Final Logit Soft-Capping: cap * tanh(logits / cap)
  - GeGLU Feed-Forward Network: gelu(gate(x)) * up(x)
  - 4-RMSNorm Decoder Block (pre/post attention, pre/post MLP)
  - Explicit 2D ('data', 'model') parameter sharding across XLA devices
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def _load_mesh_module():
    spec = importlib.util.spec_from_file_location(
        "ephemeral_tpu_mesh",
        Path(__file__).resolve().parent / "01_ephemeral_tpu_mesh.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class GemmaConfig:
    """Configuration for a Gemma 2 style architecture in Flax NNX."""

    vocab_size: int = 512
    embed_dim: int = 256
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 64
    mlp_dim: int = 1024
    num_layers: int = 4
    rope_theta: float = 10000.0
    attn_logit_softcap: float = 50.0
    final_logit_softcap: float = 30.0
    rms_norm_eps: float = 1e-6


class GemmaRMSNorm(nnx.Module):
    """Gemma zero-centered Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.eps = eps
        self.scale = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return normed * (1.0 + self.scale[...])


def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    head_dim: int,
    theta: float = 10000.0,
) -> jax.Array:
    """Applies Rotary Position Embeddings (RoPE) to query or key tensors.

    Args:
        inputs: Array of shape (batch, seq_len, num_heads, head_dim)
        positions: Array of shape (batch, seq_len)
        head_dim: Head dimension (even integer)
        theta: RoPE base period
    """
    half_dim = head_dim // 2
    freq_exponents = (2.0 * jnp.arange(half_dim, dtype=jnp.float32)) / float(head_dim)
    timescales = theta**freq_exponents
    sinusoid_inp = (
        positions[:, :, None, None].astype(jnp.float32) / timescales[None, None, None, :]
    )
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half = inputs[..., :half_dim]
    second_half = inputs[..., half_dim:]
    rotated_first = first_half * cos - second_half * sin
    rotated_second = second_half * cos + first_half * sin
    return jnp.concatenate([rotated_first, rotated_second], axis=-1).astype(inputs.dtype)


class GemmaGroupedQueryAttention(nnx.Module):
    """Grouped-Query Attention (GQA) with RoPE and Gemma-2 logit soft-capping."""

    def __init__(self, config: GemmaConfig, *, rngs: nnx.Rngs) -> None:
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = config.num_heads // config.num_kv_heads
        self.rope_theta = config.rope_theta
        self.attn_logit_softcap = config.attn_logit_softcap

        self.q_proj = nnx.Linear(
            config.embed_dim,
            config.num_heads * config.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            config.embed_dim,
            config.num_kv_heads * config.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            config.embed_dim,
            config.num_kv_heads * config.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            config.num_heads * config.head_dim,
            config.embed_dim,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = apply_rope(q, positions, self.head_dim, self.rope_theta)
        k = apply_rope(k, positions, self.head_dim, self.rope_theta)

        if self.num_queries_per_kv > 1:
            k = jnp.repeat(k, self.num_queries_per_kv, axis=2)
            v = jnp.repeat(v, self.num_queries_per_kv, axis=2)

        scale = float(self.head_dim) ** -0.5
        scores = jnp.einsum("bthd,bshd->bhts", q * scale, k)

        if self.attn_logit_softcap > 0.0:
            scores = self.attn_logit_softcap * jnp.tanh(scores / self.attn_logit_softcap)

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None, :, :]
        scores = jnp.where(causal_mask, scores, -1e9)
        attn_weights = jax.nn.softmax(scores, axis=-1)

        context = jnp.einsum("bhts,bshd->bthd", attn_weights, v)
        context = context.reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(context)


class GemmaGeGLUMLP(nnx.Module):
    """Gemma GeGLU Feed-Forward Network."""

    def __init__(self, config: GemmaConfig, *, rngs: nnx.Rngs) -> None:
        self.gate_proj = nnx.Linear(
            config.embed_dim, config.mlp_dim, use_bias=False, rngs=rngs
        )
        self.up_proj = nnx.Linear(
            config.embed_dim, config.mlp_dim, use_bias=False, rngs=rngs
        )
        self.down_proj = nnx.Linear(
            config.mlp_dim, config.embed_dim, use_bias=False, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        activated = jax.nn.gelu(self.gate_proj(x), approximate=True) * self.up_proj(x)
        return self.down_proj(activated)


class GemmaDecoderBlock(nnx.Module):
    """Gemma-2 style decoder block with 4 RMSNorm layers."""

    def __init__(self, config: GemmaConfig, *, rngs: nnx.Rngs) -> None:
        self.pre_attn_norm = GemmaRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.attn = GemmaGroupedQueryAttention(config, rngs=rngs)
        self.post_attn_norm = GemmaRMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        self.pre_mlp_norm = GemmaRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = GemmaGeGLUMLP(config, rngs=rngs)
        self.post_mlp_norm = GemmaRMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def __call__(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        attn_out = self.attn(self.pre_attn_norm(x), positions)
        x = x + self.post_attn_norm(attn_out)

        mlp_out = self.mlp(self.pre_mlp_norm(x))
        x = x + self.post_mlp_norm(mlp_out)
        return x


class GemmaForCausalLM(nnx.Module):
    """Gemma 2 style Causal Language Model in Flax NNX."""

    def __init__(self, config: GemmaConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        self.embedder = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.embed_dim,
            rngs=rngs,
        )
        self.blocks = nnx.List(
            [GemmaDecoderBlock(config, rngs=rngs) for _ in range(config.num_layers)]
        )
        self.final_norm = GemmaRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.lm_head = nnx.Linear(
            config.embed_dim, config.vocab_size, use_bias=False, rngs=rngs
        )

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array | None = None,
    ) -> jax.Array:
        bsz, seq_len = input_ids.shape
        if positions is None:
            positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (bsz, seq_len))

        hidden = self.embedder(input_ids) * (float(self.config.embed_dim) ** 0.5)
        for block in self.blocks:
            hidden = block(hidden, positions)

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        if self.config.final_logit_softcap > 0.0:
            logits = self.config.final_logit_softcap * jnp.tanh(
                logits / self.config.final_logit_softcap
            )
        return logits


def shard_gemma_model(model: GemmaForCausalLM, mesh: Mesh) -> GemmaForCausalLM:
    """Shards a GemmaForCausalLM instance across a 2D ('data', 'model') XLA Mesh."""
    graphdef, state = nnx.split(model)

    def _put_leaf(path: tuple[Any, ...], value: Any) -> Any:
        if not isinstance(value, jax.Array):
            return value
        path_str = "/".join(str(p.key if hasattr(p, "key") else p) for p in path)
        if value.ndim == 2:
            col_sharded = (
                "q_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "embedder",
                "lm_head",
                "lora_b",
            )
            if any(k in path_str for k in col_sharded):
                spec = PartitionSpec(None, "model")
            elif any(k in path_str for k in ("o_proj", "down_proj")):
                spec = PartitionSpec("model", None)
            else:
                spec = PartitionSpec(None, None)
        else:
            spec = PartitionSpec(None)
        return jax.device_put(value, NamedSharding(mesh, spec))

    sharded_state = jax.tree_util.tree_map_with_path(_put_leaf, state)
    return nnx.merge(graphdef, sharded_state)


def verify_gemma_architecture() -> dict[str, Any]:
    """Instantiates, shards, and runs a forward pass of GemmaForCausalLM on the 2D mesh."""
    mesh_mod = _load_mesh_module()
    mesh = mesh_mod.create_ephemeral_mesh()

    config = GemmaConfig()
    model = GemmaForCausalLM(config, rngs=nnx.Rngs(42))
    model = shard_gemma_model(model, mesh)

    _, params_state = nnx.split(model, nnx.Param)
    total_params = sum(leaf.size for leaf in jax.tree.leaves(params_state))

    batch_size, seq_len = 4, 16
    batch_sharding = NamedSharding(mesh, PartitionSpec("data", None))
    dummy_tokens = jax.device_put(
        jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        batch_sharding,
    )

    @nnx.jit
    def forward_step(mdl: GemmaForCausalLM, tokens: jax.Array) -> jax.Array:
        return mdl(tokens)

    logits = forward_step(model, dummy_tokens)
    q_kernel = model.blocks[0].attn.q_proj.kernel[...]
    down_kernel = model.blocks[0].mlp.down_proj.kernel[...]

    return {
        "total_params": int(total_params),
        "num_layers": config.num_layers,
        "embed_dim": config.embed_dim,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "gqa_ratio": config.num_heads // config.num_kv_heads,
        "logits_shape": tuple(logits.shape),
        "logits_max_abs": float(jnp.max(jnp.abs(logits))),
        "softcap_bound": config.final_logit_softcap,
        "q_kernel_global_shape": tuple(q_kernel.shape),
        "q_kernel_shard_shape": tuple(q_kernel.addressable_shards[0].data.shape),
        "down_kernel_global_shape": tuple(down_kernel.shape),
        "down_kernel_shard_shape": tuple(down_kernel.addressable_shards[0].data.shape),
    }


def main() -> None:
    stats = verify_gemma_architecture()
    print("=" * 72)
    print(" Experiment 02: Gemma 2 style Flax NNX Architecture & Sharding")
    print("=" * 72)
    print(f" Total Parameters          : {stats['total_params']:,}")
    print(
        f" Decoder Layers / Dim      : {stats['num_layers']} layers, "
        f"d_model={stats['embed_dim']}"
    )
    print(
        f" Grouped-Query Attention   : {stats['num_heads']} Q heads / "
        f"{stats['num_kv_heads']} KV heads ({stats['gqa_ratio']}:1 GQA)"
    )
    print(
        f" Block[0] Q-Proj Sharding  : global={stats['q_kernel_global_shape']} "
        f"-> shard={stats['q_kernel_shard_shape']}"
    )
    print(
        f" Block[0] MLP-Down Sharding: global={stats['down_kernel_global_shape']} "
        f"-> shard={stats['down_kernel_shard_shape']}"
    )
    print(
        f" Forward Logits Shape      : {stats['logits_shape']} "
        f"(max |logit|={stats['logits_max_abs']:.3f} <= cap {stats['softcap_bound']})"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
