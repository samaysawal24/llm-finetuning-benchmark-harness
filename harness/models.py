"""
Data models and configurations for LLM Fine-Tuning & Evaluation Harness
Author: smay <cricketworldsamay@gmail.com>
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class TrainingHyperparameters(BaseModel):
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    epochs: int = 3
    weight_decay: float = 0.01


class DatasetSample(BaseModel):
    instruction: str
    input: str = ""
    output: str
    formatted_prompt: Optional[str] = None


class EvaluationMetrics(BaseModel):
    benchmark_name: str
    accuracy_score: float
    perplexity: float
    samples_evaluated: int
    eval_latency_seconds: float


class FineTuningJobSummary(BaseModel):
    job_id: str
    base_model: str
    quantization: str
    trainable_parameters: int
    total_parameters: int
    trainable_percent: float
    final_train_loss: float
    final_eval_loss: float
    benchmark_results: List[EvaluationMetrics] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=datetime.utcnow)
