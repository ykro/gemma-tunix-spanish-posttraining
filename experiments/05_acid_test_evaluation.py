"""Experiment 05: The Acid Test — Spanish Technical Reasoning Benchmark.

Compares the frozen Reference Gemma Model (before post-training) against the
LoRA + GRPO Post-Trained Checkpoint across the Spanish technical instruction suite.
Measures:
  - Mean Reward Improvement (Base vs. LoRA+GRPO)
  - XML Format Compliance (<razonamiento>...</razonamiento> <respuesta>...</respuesta>)
  - Technical Spanish Keyword Density inside the reasoning block
  - Zero Base Weight Drift (bitwise integrity of frozen base model)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_grpo_module():
    spec = importlib.util.spec_from_file_location(
        "grpo_alignment_loop",
        Path(__file__).resolve().parent / "04_grpo_alignment_loop.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_acid_test_benchmark() -> dict[str, Any]:
    """Runs the before/after Spanish benchmark and returns detailed per-prompt metrics."""
    grpo_mod = _load_grpo_module()
    alignment_res = grpo_mod.run_grpo_alignment(num_steps=24, group_size=4, lora_rank=4)

    prompts = grpo_mod.SPANISH_TRAINING_PROMPTS
    initial_eval = alignment_res["initial_eval"]
    final_eval = alignment_res["final_eval"]

    benchmark_cases = []
    for idx, p_item in enumerate(prompts):
        benchmark_cases.append(
            {
                "prompt_id": p_item.prompt_id,
                "prompt_text": p_item.prompt_text,
                "base_output": initial_eval["decoded"][idx],
                "grpo_output": final_eval["decoded"][idx],
                "base_reward": float(initial_eval["rewards"][idx]),
                "grpo_reward": float(final_eval["rewards"][idx]),
                "reward_delta": float(final_eval["rewards"][idx] - initial_eval["rewards"][idx]),
            }
        )

    reward_delta = float(final_eval["mean_reward"] - initial_eval["mean_reward"])
    compliance_delta = float(
        final_eval["format_compliance_rate"] - initial_eval["format_compliance_rate"]
    )

    return {
        "base_mean_reward": float(initial_eval["mean_reward"]),
        "post_sft_mean_reward": float(alignment_res["post_sft_eval"]["mean_reward"]),
        "post_sft_format_compliance_pct": float(
            alignment_res["post_sft_eval"]["format_compliance_rate"] * 100.0
        ),
        "num_sft_warmup_steps": int(alignment_res["num_sft_warmup_steps"]),
        "grpo_mean_reward": float(final_eval["mean_reward"]),
        "reward_delta": reward_delta,
        "base_format_compliance_pct": float(initial_eval["format_compliance_rate"] * 100.0),
        "grpo_format_compliance_pct": float(final_eval["format_compliance_rate"] * 100.0),
        "compliance_delta_pct": float(compliance_delta * 100.0),
        "max_base_weight_diff": float(alignment_res["max_base_weight_diff"]),
        "cases": benchmark_cases,
        "history": alignment_res["history"],
    }


def main() -> None:
    report = run_acid_test_benchmark()
    print("=" * 72)
    print(" Experiment 05: Acid Test (Base vs. LoRA SFT+GRPO, same 4 training prompts)")
    print("=" * 72)
    print(
        f" Mean Greedy Reward        : {report['base_mean_reward']:+.4f} (Base) -> "
        f"{report['post_sft_mean_reward']:+.4f} (SFT warmup) -> "
        f"{report['grpo_mean_reward']:+.4f} (SFT+GRPO) "
        f"[Delta: {report['reward_delta']:+.4f}]"
    )
    print(
        f" XML Format Compliance     : {report['base_format_compliance_pct']:.1f}% (Base) -> "
        f"{report['post_sft_format_compliance_pct']:.1f}% (SFT warmup) -> "
        f"{report['grpo_format_compliance_pct']:.1f}% (SFT+GRPO)"
    )
    print(" Scope                     : training prompts and reward fn; not generalization")
    print(f" Base Weight Max |Diff|    : {report['max_base_weight_diff']:.8f}")
    print("-" * 72)
    for case in report["cases"]:
        print(f" Prompt [{case['prompt_id']}]: {case['prompt_text']}")
        print(f"   Base ({case['base_reward']:+.2f}) : {case['base_output']}")
        print(f"   SFT+GRPO ({case['grpo_reward']:+.2f}) : {case['grpo_output']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
