"""Experiment 04: Group Relative Policy Optimization (GRPO) in JAX & Flax NNX.

Implements the critic-free GRPO post-training loop by hand, following the recipe used
by Google Tunix (Tunix itself is not imported):
  1. Group Rollout: Samples G candidate completions per Spanish prompt from policy pi_theta.
  2. Deterministic Spanish Reward: Evaluates XML structure (<razonamiento>...</razonamiento>
     <respuesta>...</respuesta>) and Spanish technical reasoning density.
  3. Group Relative Advantage: A_hat_i = (r_i - mean(r)) / (std(r) + eps) across group G.
  4. Reference KL Penalty: Computes per-token Schulman KL divergence against the frozen
     base reference policy pi_ref without allocating a separate critic model.
  5. Surgical LoRA Update: Optimizes only nnx.LoRAParam weights via Optax AdamW under @nnx.jit.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
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


SPANISH_TECH_VOCAB: dict[int, str] = {
    0: "<pad>",
    1: "<bos>",
    2: "<eos>",
    3: "<razonamiento>",
    4: "</razonamiento>",
    5: "<respuesta>",
    6: "</respuesta>",
    7: "malla",
    8: "bidimensional",
    9: "particiona",
    10: "tensores",
    11: "datos",
    12: "modelo",
    13: "tpu",
    14: "v5e",
    15: "reduce",
    16: "memoria",
    17: "hbm",
    18: "adaptadores",
    19: "lora",
    20: "congelan",
    21: "pesos",
    22: "base",
    23: "rango",
    24: "bajo",
    25: "grpo",
    26: "normaliza",
    27: "recompensas",
    28: "grupo",
    29: "elimina",
    30: "critico",
    31: "divergencia",
    32: "kl",
    33: "estabiliza",
    34: "politica",
    35: "flax",
    36: "nnx",
    37: "separa",
    38: "estado",
    39: "grafo",
    40: "compilacion",
    41: "xla",
}

ID_TO_TOKEN = SPANISH_TECH_VOCAB
TOKEN_TO_ID = {v: k for k, v in SPANISH_TECH_VOCAB.items()}

TAG_RAZ_OPEN = 3
TAG_RAZ_CLOSE = 4
TAG_RESP_OPEN = 5
TAG_RESP_CLOSE = 6


@dataclass(frozen=True)
class SpanishPromptItem:
    """Spanish technical prompt with target reasoning tokens and reference trajectory."""

    prompt_id: str
    prompt_text: str
    prompt_tokens: tuple[int, ...]
    target_keywords: tuple[int, ...]
    exemplar_completion: tuple[int, ...]


SPANISH_TRAINING_PROMPTS: tuple[SpanishPromptItem, ...] = (
    SpanishPromptItem(
        prompt_id="tpu_mesh_2d",
        prompt_text="Explica cómo la malla 2D en TPU v5e particiona tensores y reduce memoria HBM.",
        prompt_tokens=(1, 7, 8, 13, 14, 9, 10, 16),
        target_keywords=(7, 9, 10, 15, 16, 17),
        exemplar_completion=(3, 7, 9, 10, 15, 16, 4, 5, 17, 6),
    ),
    SpanishPromptItem(
        prompt_id="lora_surgical",
        prompt_text="Describe por qué los adaptadores LoRA de rango bajo congelan los pesos base.",
        prompt_tokens=(1, 18, 19, 23, 24, 20, 21, 22),
        target_keywords=(18, 19, 20, 21, 22, 15),
        exemplar_completion=(3, 18, 19, 20, 21, 22, 4, 5, 15, 6),
    ),
    SpanishPromptItem(
        prompt_id="grpo_critic_free",
        prompt_text=(
            "Analiza cómo GRPO normaliza recompensas por grupo y elimina el modelo crítico."
        ),
        prompt_tokens=(1, 25, 26, 27, 28, 29, 30, 32),
        target_keywords=(25, 26, 27, 29, 30, 33),
        exemplar_completion=(3, 25, 26, 27, 29, 30, 4, 5, 33, 6),
    ),
    SpanishPromptItem(
        prompt_id="nnx_split_state",
        prompt_text="Detalla cómo Flax NNX separa el grafo y el estado para compilación XLA.",
        prompt_tokens=(1, 35, 36, 37, 38, 39, 40, 41),
        target_keywords=(35, 36, 37, 38, 39, 41),
        exemplar_completion=(3, 35, 36, 37, 38, 39, 4, 5, 41, 6),
    ),
)


def decode_tokens(token_ids: list[int] | tuple[int, ...] | jax.Array) -> str:
    """Decodes integer token IDs into a readable Spanish string."""
    if isinstance(token_ids, jax.Array):
        token_ids = [int(t) for t in token_ids.tolist()]
    words = [ID_TO_TOKEN.get(int(t), f"tok_{int(t)}") for t in token_ids]
    return " ".join(words)


def score_spanish_completion(
    completion_ids: list[int] | tuple[int, ...] | jax.Array,
    target_keywords: tuple[int, ...],
) -> dict[str, float | bool]:
    """Deterministic reward function scoring XML structure and Spanish technical reasoning."""
    if isinstance(completion_ids, jax.Array):
        seq = [int(x) for x in completion_ids.tolist()]
    else:
        seq = [int(x) for x in completion_ids]

    reward = 0.0
    has_open_raz = TAG_RAZ_OPEN in seq
    has_close_raz = TAG_RAZ_CLOSE in seq
    has_open_resp = TAG_RESP_OPEN in seq
    has_close_resp = TAG_RESP_CLOSE in seq

    if has_open_raz and seq[0] == TAG_RAZ_OPEN:
        reward += 0.30
    elif has_open_raz:
        reward += 0.10

    valid_structure = False
    if has_open_raz and has_close_raz and has_open_resp and has_close_resp:
        idx_r1 = seq.index(TAG_RAZ_OPEN)
        idx_r2 = seq.index(TAG_RAZ_CLOSE)
        idx_a1 = seq.index(TAG_RESP_OPEN)
        idx_a2 = seq.index(TAG_RESP_CLOSE)
        if idx_r1 < idx_r2 < idx_a1 < idx_a2:
            valid_structure = True
            reward += 0.70
            reasoning_span = seq[idx_r1 + 1 : idx_r2]
            answer_span = seq[idx_a1 + 1 : idx_a2]

            matched_keywords = sum(1 for kw in target_keywords if kw in reasoning_span)
            reward += min(0.75, matched_keywords * 0.15)

            if len(answer_span) >= 1 and any(tok >= 7 for tok in answer_span):
                reward += 0.25
        else:
            reward -= 0.15
    else:
        matched_any = sum(1 for kw in target_keywords if kw in seq)
        reward += min(0.20, matched_any * 0.05) - 0.10

    return {
        "reward": float(reward),
        "valid_structure": valid_structure,
    }


def compute_group_normalized_advantages(
    rewards: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Computes group-relative normalized advantages per prompt: (r - mean) / (std + eps)."""
    mean_r = jnp.mean(rewards, axis=-1, keepdims=True)
    std_r = jnp.std(rewards, axis=-1, keepdims=True)
    return (rewards - mean_r) / (std_r + eps)


def compute_sequence_log_probs(
    model: Any,
    full_sequences: jax.Array,
    prompt_len: int,
) -> jax.Array:
    """Computes per-token log-probabilities over the generated completion slice."""
    logits = model(full_sequences[:, :-1])
    gen_logits = logits[:, prompt_len - 1 :, :]
    targets = full_sequences[:, prompt_len:]
    log_probs = jax.nn.log_softmax(gen_logits, axis=-1)
    token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return token_log_probs


@nnx.jit
def _jitted_forward_logits(model: Any, buffer: jax.Array) -> jax.Array:
    return model(buffer)


def greedy_generate(
    model: Any,
    prompt_tokens: jax.Array,
    gen_len: int = 10,
    vocab_limit: int = 42,
) -> jax.Array:
    """Autoregressively decodes `gen_len` tokens using a static-shape buffer for XLA cache reuse."""
    bsz, prompt_len = prompt_tokens.shape
    total_len = prompt_len + gen_len
    buffer = jnp.zeros((bsz, total_len), dtype=jnp.int32)
    buffer = buffer.at[:, :prompt_len].set(prompt_tokens)

    for step_idx in range(gen_len):
        pos = prompt_len + step_idx
        logits = _jitted_forward_logits(model, buffer)
        next_logits = logits[:, pos - 1, :vocab_limit]
        next_tok = jnp.argmax(next_logits, axis=-1).astype(jnp.int32)
        buffer = buffer.at[:, pos].set(next_tok)

    return buffer[:, prompt_len:]


def sample_group_completions(
    model: Any,
    prompts: tuple[SpanishPromptItem, ...],
    group_size: int,
    gen_len: int,
    step_rng: jax.Array,
    vocab_limit: int = 42,
    temperature: float = 0.85,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Autoregressively samples G candidate completions per prompt from policy pi_theta."""
    num_prompts = len(prompts)
    prompt_len = len(prompts[0].prompt_tokens)
    total_len = prompt_len + gen_len
    prompt_matrix = jnp.array([p.prompt_tokens for p in prompts], dtype=jnp.int32)
    repeated_prompts = jnp.repeat(prompt_matrix, repeats=group_size, axis=0)

    # Autoregressive token-by-token rollout over static-shape buffer (B * G, total_len)
    buffer = jnp.zeros((num_prompts * group_size, total_len), dtype=jnp.int32)
    buffer = buffer.at[:, :prompt_len].set(repeated_prompts)

    # Temperature schedule across the G candidates in each group:
    # g=0 greedy/low-temp (0.45), g=1 moderate (0.75), g=2 exploratory (1.05), g=3 broad (1.35)
    temps_per_group = jnp.linspace(0.45, 1.35, group_size, dtype=jnp.float32)
    temps_batch = jnp.tile(temps_per_group, reps=(num_prompts,))[:, None] * (temperature / 0.85)

    rng = step_rng
    for step_idx in range(gen_len):
        pos = prompt_len + step_idx
        rng, tok_rng = jax.random.split(rng)
        logits = _jitted_forward_logits(model, buffer)
        scaled_logits = logits[:, pos - 1, :vocab_limit] / temps_batch
        next_tok = jax.random.categorical(tok_rng, scaled_logits, axis=-1).astype(jnp.int32)
        buffer = buffer.at[:, pos].set(next_tok)

    sampled_completions = buffer[:, prompt_len:].reshape(num_prompts, group_size, gen_len)

    all_rewards = []
    for p_idx, p_item in enumerate(prompts):
        reward_list = []
        for g_idx in range(group_size):
            comp = sampled_completions[p_idx, g_idx]
            r_info = score_spanish_completion(comp, p_item.target_keywords)
            reward_list.append(float(r_info["reward"]))
        all_rewards.append(jnp.array(reward_list, dtype=jnp.float32))

    rewards_arr = jnp.stack(all_rewards, axis=0)
    return buffer, sampled_completions, rewards_arr


def run_grpo_alignment(
    num_steps: int = 24,
    group_size: int = 4,
    lora_rank: int = 4,
    kl_beta: float = 0.04,
    learning_rate: float = 1.5e-2,
    ppo_epochs: int = 2,
) -> dict[str, Any]:
    """Executes the full Tunix-style GRPO alignment loop in JAX / Flax NNX."""
    mesh_mod = _load_module("ephemeral_tpu_mesh", "01_ephemeral_tpu_mesh.py")
    gemma_mod = _load_module("gemma_nnx_architecture", "02_gemma_nnx_architecture.py")
    lora_mod = _load_module("lora_surgical_injection", "03_lora_surgical_injection.py")

    mesh = mesh_mod.create_ephemeral_mesh()
    config = gemma_mod.GemmaConfig(
        vocab_size=64,
        embed_dim=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        mlp_dim=256,
        num_layers=2,
    )

    base_model = gemma_mod.GemmaForCausalLM(config, rngs=nnx.Rngs(101))
    ref_model = gemma_mod.GemmaForCausalLM(config, rngs=nnx.Rngs(101))
    ref_model = gemma_mod.shard_gemma_model(ref_model, mesh)

    policy_model = lora_mod.inject_lora_adapters(
        base_model, lora_rank=lora_rank, include_mlp=True, rngs=nnx.Rngs(202)
    )
    policy_model = gemma_mod.shard_gemma_model(policy_model, mesh)

    base_filter = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))
    _, _, base_before = nnx.split(policy_model, nnx.LoRAParam, base_filter)
    base_leaves_before = [jnp.copy(x) for x in jax.tree.leaves(base_before)]

    optimizer = nnx.Optimizer(
        policy_model,
        optax.adamw(learning_rate=learning_rate, weight_decay=1e-4),
        wrt=nnx.LoRAParam,
    )

    prompts = SPANISH_TRAINING_PROMPTS
    prompt_len = len(prompts[0].prompt_tokens)
    gen_len = len(prompts[0].exemplar_completion)

    def evaluate_greedy_policy(mdl: Any) -> dict[str, Any]:
        prompt_mat = jnp.array([p.prompt_tokens for p in prompts], dtype=jnp.int32)
        gen_tokens = greedy_generate(mdl, prompt_mat, gen_len=gen_len, vocab_limit=42)
        rewards = []
        valid_flags = []
        decoded = []
        for idx, p_item in enumerate(prompts):
            info = score_spanish_completion(gen_tokens[idx], p_item.target_keywords)
            rewards.append(float(info["reward"]))
            valid_flags.append(bool(info["valid_structure"]))
            decoded.append(decode_tokens(gen_tokens[idx]))
        return {
            "mean_reward": float(sum(rewards) / len(rewards)),
            "format_compliance_rate": float(sum(1.0 for v in valid_flags if v) / len(valid_flags)),
            "rewards": rewards,
            "decoded": decoded,
        }

    initial_eval = evaluate_greedy_policy(policy_model)

    # Stage 4A: Brief format warmup (SFT cold-start, standard in Tunix/DeepSeek-R1 before GRPO)
    # so the randomly initialized 2-layer demo policy (312,448 base params) has non-zero
    # probability of emitting XML tags
    # during purely on-policy autoregressive group sampling.
    warmup_seqs = jnp.array(
        [p.prompt_tokens + p.exemplar_completion for p in prompts], dtype=jnp.int32
    )

    @nnx.jit
    def sft_warmup_step(mdl: Any, opt: nnx.Optimizer, seqs: jax.Array) -> jax.Array:
        def sft_loss(m: Any) -> jax.Array:
            log_p = compute_sequence_log_probs(m, seqs, prompt_len)
            return -jnp.mean(log_p)

        loss_v, grads = nnx.value_and_grad(
            sft_loss, argnums=nnx.DiffState(0, nnx.LoRAParam)
        )(mdl)
        opt.update(mdl, grads)
        return loss_v

    num_sft_warmup_steps = 4
    for _ in range(num_sft_warmup_steps):
        _ = sft_warmup_step(policy_model, optimizer, warmup_seqs)

    # Greedy decoding is deterministic and does not touch the PRNG or the optimizer,
    # so measuring here does not change the GRPO trajectory. It separates how much of
    # the final gain comes from the SFT warmup and how much from GRPO itself.
    post_sft_eval = evaluate_greedy_policy(policy_model)

    @nnx.jit
    def compute_frozen_log_probs(mdl: Any, full_seqs: jax.Array) -> jax.Array:
        return jax.lax.stop_gradient(compute_sequence_log_probs(mdl, full_seqs, prompt_len))

    @nnx.jit
    def grpo_update_epoch(
        mdl: Any,
        opt: nnx.Optimizer,
        full_seqs: jax.Array,
        old_log_probs: jax.Array,
        ref_log_probs: jax.Array,
        advantages: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        def loss_fn(policy_mdl: Any) -> tuple[jax.Array, jax.Array]:
            new_log_probs = compute_sequence_log_probs(policy_mdl, full_seqs, prompt_len)
            log_ratio = new_log_probs - old_log_probs
            ratio = jnp.exp(log_ratio)

            adv_expanded = advantages.reshape(-1, 1)
            unclipped = ratio * adv_expanded
            clipped = jnp.clip(ratio, 0.8, 1.2) * adv_expanded
            surrogate_obj = jnp.minimum(unclipped, clipped)

            ref_diff = ref_log_probs - new_log_probs
            per_token_kl = jnp.exp(ref_diff) - ref_diff - 1.0

            loss = -jnp.mean(surrogate_obj - kl_beta * per_token_kl)
            return loss, jnp.mean(per_token_kl)

        grad_fn = nnx.value_and_grad(
            loss_fn,
            argnums=nnx.DiffState(0, nnx.LoRAParam),
            has_aux=True,
        )
        (loss_val, kl_val), grads = grad_fn(mdl)
        opt.update(mdl, grads)
        return loss_val, kl_val

    prng = jax.random.PRNGKey(2026)
    history: list[dict[str, float]] = []

    for step in range(1, num_steps + 1):
        prng, step_rng = jax.random.split(prng)
        full_seqs, _, rewards_arr = sample_group_completions(
            policy_model, prompts, group_size, gen_len, step_rng, vocab_limit=42
        )
        advantages = compute_group_normalized_advantages(rewards_arr)
        old_log_probs = compute_frozen_log_probs(policy_model, full_seqs)
        ref_log_probs = compute_frozen_log_probs(ref_model, full_seqs)

        loss_val = jnp.array(0.0)
        kl_val = jnp.array(0.0)
        for _ in range(ppo_epochs):
            loss_val, kl_val = grpo_update_epoch(
                policy_model, optimizer, full_seqs, old_log_probs, ref_log_probs, advantages
            )

        if step == 1 or step % 6 == 0 or step == num_steps:
            step_eval = evaluate_greedy_policy(policy_model)
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss_val),
                    "kl_divergence": float(kl_val),
                    "group_mean_reward": float(jnp.mean(rewards_arr)),
                    "greedy_eval_reward": float(step_eval["mean_reward"]),
                    "format_compliance_rate": float(step_eval["format_compliance_rate"]),
                }
            )

    final_eval = evaluate_greedy_policy(policy_model)

    _, _, base_after = nnx.split(policy_model, nnx.LoRAParam, base_filter)
    base_leaves_after = jax.tree.leaves(base_after)
    max_base_diff = max(
        float(jnp.max(jnp.abs(a - b))) for a, b in zip(base_leaves_before, base_leaves_after)
    )

    return {
        "num_steps": num_steps,
        "group_size": group_size,
        "lora_rank": lora_rank,
        "kl_beta": kl_beta,
        "num_sft_warmup_steps": num_sft_warmup_steps,
        "initial_eval": initial_eval,
        "post_sft_eval": post_sft_eval,
        "final_eval": final_eval,
        "history": history,
        "max_base_weight_diff": max_base_diff,
        "policy_model": policy_model,
        "ref_model": ref_model,
    }


def main() -> None:
    res = run_grpo_alignment()
    print("=" * 72)
    print(" Experiment 04: Critic-Free GRPO Alignment Loop (JAX + Flax NNX)")
    print("=" * 72)
    print(
        f" Group Size (G) / Steps    : G={res['group_size']} completions/prompt, "
        f"{res['num_steps']} GRPO steps (KL beta={res['kl_beta']})"
    )
    print(
        f" Greedy Reward, base       : {res['initial_eval']['mean_reward']:+.4f} "
        f"(XML format compliance: {res['initial_eval']['format_compliance_rate'] * 100:.1f}%)"
    )
    print(
        f" Greedy Reward, after SFT  : {res['post_sft_eval']['mean_reward']:+.4f} "
        f"(XML format compliance: {res['post_sft_eval']['format_compliance_rate'] * 100:.1f}%)"
        f" <- {res['num_sft_warmup_steps']} SFT warmup steps, before any GRPO step"
    )
    for row in res["history"]:
        print(
            f"   [Step {row['step']:02d}] loss={row['loss']:+.4f} | "
            f"KL={row['kl_divergence']:.4f} | "
            f"group_reward={row['group_mean_reward']:+.4f} | "
            f"greedy_reward={row['greedy_eval_reward']:+.4f} | "
            f"XML_compliance={row['format_compliance_rate'] * 100:.0f}%"
        )
    print(
        f" Greedy Reward, SFT+GRPO   : {res['final_eval']['mean_reward']:+.4f} "
        f"(XML format compliance: {res['final_eval']['format_compliance_rate'] * 100:.1f}%)"
    )
    print(f" Base Weight Max |Diff|    : {res['max_base_weight_diff']:.8f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
