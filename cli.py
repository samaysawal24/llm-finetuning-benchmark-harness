#!/usr/bin/env python3
"""
CLI entry point for LLM Fine-Tuning & Evaluation Harness
Author: smay <cricketworldsamay@gmail.com>
"""

import sys
import argparse
import yaml
from harness import (
    FineTuningEngine,
    LoRAConfig,
    TrainingHyperparameters,
    DatasetLoader,
    ParameterCalculator
)


def load_yaml_config(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading YAML config: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning & Evaluation CLI - smay")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Launch fine-tuning run with LoRA/QLoRA")
    train_parser.add_argument("--config", type=str, default="configs/lora_llama3_8b.yaml", help="Path to YAML config")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run multi-benchmark evaluation")
    bench_parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    bench_parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "human_eval"])

    # Info command
    info_parser = subparsers.add_parser("info", help="Inspect LoRA parameter distribution and VRAM usage")
    info_parser.add_argument("--params", type=int, default=8_000_000_000)
    info_parser.add_argument("--r", type=int, default=16)

    args = parser.parse_args()

    if args.command == "info":
        stats = ParameterCalculator.calculate_lora_parameters(args.params, r=args.r, target_modules_count=7)
        vram_4bit = ParameterCalculator.estimate_vram_requirement_gb(args.params, quantization="4bit")
        print("\n=== Model Parameter & VRAM Profile ===")
        print(f"Base Parameters:       {stats['total_base_parameters']:,}")
        print(f"Trainable Parameters:  {stats['trainable_lora_parameters']:,} ({stats['trainable_percentage']}%)")
        print(f"Estimated 4-bit VRAM:  {vram_4bit} GB\n")

    elif args.command == "train":
        print(f"Loading configuration from {args.config}...")
        cfg = load_yaml_config(args.config)
        lora_cfg = LoRAConfig(**cfg.get("lora", {}))
        hyperparams = TrainingHyperparameters(**cfg.get("training", {}))

        engine = FineTuningEngine(base_model=cfg.get("model_name"), quantization=cfg.get("quantization", "4bit"))
        samples = DatasetLoader.generate_synthetic_samples(16)
        train_set, val_set = DatasetLoader().split_dataset(samples)

        print(f"Starting training run on {cfg.get('model_name')}...")
        summary = engine.run_training_job(
            lora_cfg=lora_cfg,
            hyperparams=hyperparams,
            train_samples=train_set,
            val_samples=val_set,
            benchmarks=cfg.get("evaluation", {}).get("benchmarks")
        )

        print("\n=== Fine-Tuning Run Completed ===")
        print(f"Job ID:               {summary.job_id}")
        print(f"Trainable Parameters: {summary.trainable_parameters:,} ({summary.trainable_percent}%)")
        print(f"Final Train Loss:     {summary.final_train_loss}")
        print("\n=== Benchmark Scorecard ===")
        for b in summary.benchmark_results:
            print(f"- {b.benchmark_name:<12} | Accuracy: {b.accuracy_score * 100:.2f}% | PPL: {b.perplexity:<6} | Latency: {b.eval_latency_seconds}s")
        print()

    elif args.command == "benchmark":
        from harness import BenchmarkEvaluator
        evaluator = BenchmarkEvaluator()
        results = evaluator.run_all(args.benchmarks)
        print("\n=== Benchmark Evaluation Leaderboard ===")
        for b in results:
            print(f"- {b.benchmark_name:<12} | Accuracy: {b.accuracy_score * 100:.2f}% | PPL: {b.perplexity:<6}")
        print()


if __name__ == "__main__":
    main()
