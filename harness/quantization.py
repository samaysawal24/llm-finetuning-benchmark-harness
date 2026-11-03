"""
Quantization & Parameter Calculation Utilities
Author: Samay Sawal <cricketworldsamay@gmail.com>
"""

from typing import Dict, Any


class ParameterCalculator:
    """
    Computes parameter distributions, trainable parameter counts for LoRA matrices,
    and VRAM memory requirements for 16-bit, 8-bit, and 4-bit (QLoRA) models.
    """

    @staticmethod
    def calculate_lora_parameters(base_params: int, r: int, target_modules_count: int, hidden_size: int = 4096) -> Dict[str, Any]:
        """
        In LoRA: For each targeted linear weight W (d x k), we introduce A (d x r) and B (r x k).
        Trainable params per layer = 2 * hidden_size * r.
        """
        params_per_layer = 2 * hidden_size * r
        trainable_lora_params = params_per_layer * target_modules_count * 32  # Assuming 32 transformer layers

        trainable_pct = (trainable_lora_params / base_params) * 100.0

        return {
            "total_base_parameters": base_params,
            "trainable_lora_parameters": trainable_lora_params,
            "trainable_percentage": round(trainable_pct, 4)
        }

    @staticmethod
    def estimate_vram_requirement_gb(base_params: int, quantization: str = "4bit", batch_size: int = 4) -> float:
        # Base weights VRAM
        if quantization == "4bit":
            bytes_per_param = 0.5
        elif quantization == "8bit":
            bytes_per_param = 1.0
        else:
            bytes_per_param = 2.0  # FP16/BF16

        weights_gb = (base_params * bytes_per_param) / (1024 ** 3)
        # Activation + optimizer overhead per batch
        activation_gb = (batch_size * 0.8)
        total_vram = weights_gb + activation_gb + 1.2  # 1.2GB CUDA runtime overhead
        return round(total_vram, 2)

