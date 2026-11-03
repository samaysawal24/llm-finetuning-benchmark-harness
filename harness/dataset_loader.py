"""
Instruction Dataset Loader and ChatML Formatter
Author: smay <cricketworldsamay@gmail.com>
"""

import json
from typing import List, Tuple
from .models import DatasetSample


class DatasetLoader:
    """
    Parses instruction fine-tuning datasets and formats them into
    ChatML or Alpaca prompt templates.
    """

    CHATML_TEMPLATE = (
        "<|im_start|>system\n"
        "You are an expert enterprise software engineering assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "{response}<|im_end|>"
    )

    def format_sample(self, sample: DatasetSample) -> DatasetSample:
        full_user_prompt = sample.instruction
        if sample.input:
            full_user_prompt += f"\nContext:\n{sample.input}"

        formatted = self.CHATML_TEMPLATE.format(
            prompt=full_user_prompt.strip(),
            response=sample.output.strip()
        )
        sample.formatted_prompt = formatted
        return sample

    def split_dataset(self, samples: List[DatasetSample], val_ratio: float = 0.1) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        formatted_samples = [self.format_sample(s) for s in samples]
        val_count = max(1, int(len(formatted_samples) * val_ratio))
        return formatted_samples[:-val_count], formatted_samples[-val_count:]

    @classmethod
    def generate_synthetic_samples(cls, count: int = 20) -> List[DatasetSample]:
        samples = []
        for i in range(1, count + 1):
            samples.append(
                DatasetSample(
                    instruction=f"Optimize database index #{i} for write-heavy timeseries telemetry.",
                    input=f"Current query pattern: SELECT timestamp, cpu_percent FROM metrics WHERE host_id = {i * 10} ORDER BY timestamp DESC LIMIT 100;",
                    output=f"Recommended composite index: CREATE INDEX idx_host_timestamp ON metrics (host_id, timestamp DESC); This facilitates backwards index scans avoiding filesort."
                )
            )
        return samples
