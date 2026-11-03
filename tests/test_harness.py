"""
Unit tests for LLM Fine-Tuning & Evaluation Harness
Author: smay <cricketworldsamay@gmail.com>
"""

import pytest
from harness import (
    FineTuningEngine,
    LoRAConfig,
    TrainingHyperparameters,
    DatasetLoader,
    ParameterCalculator,
    BenchmarkEvaluator
)


def test_parameter_calculator():
    stats = ParameterCalculator.calculate_lora_parameters(
        base_params=8_000_000_000,
        r=16,
        target_modules_count=7
    )
    assert stats["trainable_lora_parameters"] > 0
    assert stats["trainable_percentage"] < 1.0  # LoRA should be < 1% of total weights


def test_dataset_chatml_formatting():
    loader = DatasetLoader()
    samples = DatasetLoader.generate_synthetic_samples(5)
    train_set, val_set = loader.split_dataset(samples, val_ratio=0.2)

    assert len(train_set) == 4
    assert len(val_set) == 1
    assert "<|im_start|>system" in train_set[0].formatted_prompt
    assert "<|im_end|>" in train_set[0].formatted_prompt


def test_training_and_evaluation_cycle():
    engine = FineTuningEngine()
    lora = LoRAConfig(r=8, lora_alpha=16)
    hypers = TrainingHyperparameters(epochs=1, batch_size=2)
    samples = DatasetLoader.generate_synthetic_samples(6)
    train, val = DatasetLoader().split_dataset(samples)

    summary = engine.run_training_job(
        lora_cfg=lora,
        hyperparams=hypers,
        train_samples=train,
        val_samples=val,
        benchmarks=["gsm8k", "mmlu"]
    )

    assert summary.job_id.startswith("ft_")
    assert summary.final_train_loss > 0.0
    assert len(summary.benchmark_results) == 2
