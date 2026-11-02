"""The Outside Judge (verify_posttraining.py).

Independently executes and audits all 5 stages of the Gemma-style, Tunix-style
Spanish post-training pipeline (Tunix is not imported; GRPO is written by hand)
on a 4-device XLA mesh:
  1. 2D ("data", "model") XLA Mesh Sharding (bytes per device by PartitionSpec)
  2. Gemma 2 style Flax NNX Architecture (GQA, GeGLU, RoPE, Soft-Capping)
  3. Surgical LoRA Injection (>90% reduction & bitwise frozen base weights)
  4. Critic-Free GRPO Group Advantage Normalization & KL Regularization
  5. Acid Test on the 4 training prompts (Base vs. LoRA SFT warmup + GRPO)

Writes real, reproducible telemetry to:
  - benchmarks/latest_results.json
  - benchmarks/terminal_output.txt
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


def _load_experiment(name: str, filename: str):
    exp_path = Path(__file__).resolve().parent / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, exp_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_verification_suite() -> dict[str, Any]:
    """Executes the full verification suite and asserts all engineering invariants."""
    t0 = time.perf_counter()

    exp01 = _load_experiment("exp01", "01_ephemeral_tpu_mesh.py")
    exp02 = _load_experiment("exp02", "02_gemma_nnx_architecture.py")
    exp03 = _load_experiment("exp03", "03_lora_surgical_injection.py")
    exp05 = _load_experiment("exp05", "05_acid_test_evaluation.py")

    mesh_stats = exp01.inspect_mesh_topology()
    gemma_stats = exp02.verify_gemma_architecture()
    lora_stats = exp03.verify_surgical_lora_step(lora_rank=4)
    acid_stats = exp05.run_acid_test_benchmark()

    elapsed_s = time.perf_counter() - t0

    checks = {
        "multi_device_mesh_active": mesh_stats["num_devices"] >= 4
        and mesh_stats["mesh_shape"] == {"data": 2, "model": 2},
        "weight_shard_compression_2x": mesh_stats["per_device_weight_compression_ratio"] == 2.0,
        "logit_softcap_respected": gemma_stats["logits_max_abs"] <= gemma_stats["softcap_bound"],
        "lora_reduction_above_90pct": lora_stats["reduction_pct"] >= 90.0,
        "base_weights_frozen_step": lora_stats["base_weights_frozen_bitwise"] is True,
        "lora_adapters_updated": lora_stats["lora_weights_updated"] is True,
        "grpo_reward_improved": acid_stats["grpo_mean_reward"] > acid_stats["base_mean_reward"],
        "grpo_format_compliance_improved": (
            acid_stats["grpo_format_compliance_pct"] > acid_stats["base_format_compliance_pct"]
        ),
        "grpo_base_weights_frozen": acid_stats["max_base_weight_diff"] == 0.0,
    }
    all_passed = all(checks.values())

    results = {
        "project": "gemma-tunix-spanish-posttraining",
        "all_checks_passed": all_passed,
        "execution_time_seconds": round(elapsed_s, 3),
        "checks": checks,
        "stage_01_mesh": mesh_stats,
        "stage_02_gemma_nnx": gemma_stats,
        "stage_03_surgical_lora": lora_stats,
        "stage_04_05_grpo_acid_test": acid_stats,
    }

    lines = [
        "=" * 78,
        " OUTSIDE JUDGE VERIFICATION: gemma-tunix-spanish-posttraining",
        "=" * 78,
        f" [STAGE 01] 2D XLA Mesh Topology       : {mesh_stats['num_devices']} devices "
        f"{mesh_stats['mesh_shape']} (backend={mesh_stats['backend']}"
        f"{', simulated' if mesh_stats['backend'] == 'cpu' else ''})",
        f"            Batch Token Sharding       : global={mesh_stats['tokens_global_shape']} "
        f"-> shard={mesh_stats['tokens_shard_shape']}",
        f"            Q-Proj Weight Sharding     : global={mesh_stats['q_kernel_global_shape']} "
        f"-> shard={mesh_stats['q_kernel_shard_shape']} "
        f"({mesh_stats['per_device_weight_compression_ratio']:.1f}x fewer bytes/device, "
        f"PartitionSpec arithmetic)",
        "-" * 78,
        f" [STAGE 02] Gemma 2 style NNX Model    : {gemma_stats['total_params']:,} base parameters "
        f"({gemma_stats['num_layers']} layers, d_model={gemma_stats['embed_dim']})",
        f"            GQA & Soft-Capping         : "
        f"{gemma_stats['num_heads']}Q/{gemma_stats['num_kv_heads']}KV "
        f"({gemma_stats['gqa_ratio']}:1 GQA), max |logit|={gemma_stats['logits_max_abs']:.3f} "
        f"<= {gemma_stats['softcap_bound']}",
        "-" * 78,
        f" [STAGE 03] Surgical LoRA Injection    : "
        f"{lora_stats['lora_params']:,} trainable LoRAParam / "
        f"{lora_stats['total_params']:,} total "
        f"({lora_stats['trainable_ratio_pct']:.2f}% trainable, "
        f"{lora_stats['reduction_pct']:.2f}% frozen)",
        f"            Base Parameter Drift       : "
        f"max |diff| = {lora_stats['max_base_weight_diff']:.8f} "
        f"(bitwise frozen = {lora_stats['base_weights_frozen_bitwise']})",
        f"            LoRA Adapter Update        : "
        f"max |diff| = {lora_stats['max_lora_weight_diff']:.6f} "
        f"(loss {lora_stats['loss_before']:.4f} -> {lora_stats['loss_after']:.4f})",
        "-" * 78,
        " [STAGE 04] Critic-Free GRPO Loop      : "
        "G=4 group rollouts/prompt, 24 steps, KL beta=0.04",
        f"            SFT warmup ({acid_stats['num_sft_warmup_steps']} steps)       : "
        f"greedy_reward={acid_stats['post_sft_mean_reward']:+.4f} | "
        f"XML_compliance={acid_stats['post_sft_format_compliance_pct']:.0f}% "
        "(before any GRPO step)",
    ]
    for h in acid_stats["history"]:
        lines.append(
            f"            Step {h['step']:02d}                    : "
            f"loss={h['loss']:+.4f} | KL={h['kl_divergence']:.4f} | "
            f"group_reward={h['group_mean_reward']:+.4f} | "
            f"greedy_reward={h['greedy_eval_reward']:+.4f} | "
            f"XML_compliance={h['format_compliance_rate'] * 100:.0f}%"
        )
    lines.extend(
        [
            "-" * 78,
            f" [STAGE 05] Acid Test (4 train prompts): "
            f"Greedy reward {acid_stats['base_mean_reward']:+.4f} -> "
            f"{acid_stats['grpo_mean_reward']:+.4f} "
            f"(Delta {acid_stats['reward_delta']:+.4f})",
            f"            XML Format Compliance      : "
            f"{acid_stats['base_format_compliance_pct']:.1f}% -> "
            f"{acid_stats['grpo_format_compliance_pct']:.1f}% "
            f"(+{acid_stats['compliance_delta_pct']:.1f}%)",
            f"            Base Weight Drift (end)    : "
            f"max |diff| = {acid_stats['max_base_weight_diff']:.8f}",
            "-" * 78,
        ]
    )
    for case in acid_stats["cases"]:
        lines.append(f"   Prompt [{case['prompt_id']}] : {case['prompt_text']}")
        lines.append(f"     Base ({case['base_reward']:+.2f}) : {case['base_output']}")
        lines.append(f"     SFT+GRPO ({case['grpo_reward']:+.2f}) : {case['grpo_output']}")
    lines.extend(
        [
            "=" * 78,
            f" VERIFICATION STATUS: {'PASS' if all_passed else 'FAIL'} "
            f"({sum(checks.values())}/{len(checks)} checks passed in {elapsed_s:.2f}s)",
            "=" * 78,
        ]
    )

    terminal_report = "\n".join(lines) + "\n"
    print(terminal_report, end="")

    benchmarks_dir = Path(__file__).resolve().parent / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    (benchmarks_dir / "latest_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (benchmarks_dir / "terminal_output.txt").write_text(
        terminal_report,
        encoding="utf-8",
    )

    if not all_passed:
        raise AssertionError(f"Verification checks failed: {checks}")

    return results


if __name__ == "__main__":
    run_verification_suite()
