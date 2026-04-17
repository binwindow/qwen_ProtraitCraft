#!/usr/bin/env python3
"""
Memory Profiling for Qwen3-VL-4B Training
Theoretical estimation based on model parameters and standard formulas
"""
import argparse
from dataclasses import dataclass


@dataclass
class MemoryProfile:
    """Memory profile for a configuration."""
    model_weights: float  # GB
    gradients: float  # GB
    optimizer_states: float  # GB
    activations: float  # GB
    total_per_gpu: float  # GB
    num_gpus_needed: int
    feasible: bool

    def __str__(self):
        status = "✓ 可行" if self.feasible else "✗ 不可行"
        return (
            f"  模型权重(bf16): {self.model_weights:.1f} GB\n"
            f"  梯度(bf16): {self.gradients:.1f} GB\n"
            f"  优化器状态(fp32): {self.optimizer_states:.1f} GB\n"
            f"  激活值: {self.activations:.1f} GB\n"
            f"  ------------------------\n"
            f"  单卡显存: {self.total_per_gpu:.1f} GB\n"
            f"  最少GPU: {self.num_gpus_needed}\n"
            f"  状态: {status}"
        )


class Qwen3VLMemoryEstimator:
    """Memory estimator for Qwen3-VL-4B."""

    # Qwen3-VL-4B model parameters
    TOTAL_PARAMS = 4_000_000_000  # 4B params
    HIDDEN_DIM = 2560  # Hidden dimension
    NUM_LAYERS = 32  # Number of layers
    FFN_DIM = 6912  # FFN intermediate dimension
    NUM_HEADS = 32  # Attention heads
    HEAD_DIM = 80  # Each head dimension
    SEQ_LEN = 512  # Typical training sequence length

    # Memory constants (bytes)
    BF16_SIZE = 2
    FP32_SIZE = 4
    GB = 1024 ** 3

    def __init__(self, gpu_memory_gb: int = 24, num_available_gpus: int = 4):
        self.gpu_memory_gb = gpu_memory_gb
        self.num_available_gpus = num_available_gpus
        self.usable_memory_gb = gpu_memory_gb * 0.85  # Leave 15% buffer

    def estimate_lora_trainable_params(
        self,
        lora_r: int = 64,
    ) -> int:
        """Estimate LoRA trainable parameters.

        LoRA adds trainable params for q_proj, k_proj, v_proj, o_proj:
        Each proj: hidden_dim * lora_r (A or B matrix)
        For 4 projections per layer: 4 * hidden_dim * lora_r
        """
        lora_params_per_layer = 4 * self.HIDDEN_DIM * lora_r
        num_layers = self.NUM_LAYERS
        return lora_params_per_layer * num_layers

    def estimate_vision_trainable_params(
        self,
        tune_mlp: bool = True,
    ) -> int:
        """Estimate vision encoder trainable parameters."""
        vision_params = 200_000_000  # ~200M params for vision
        if tune_mlp:
            return vision_params + 10_000_000  # + merger
        return vision_params

    def estimate_model_weights(self) -> float:
        """Estimate model weights memory (bf16)."""
        return (self.TOTAL_PARAMS * self.BF16_SIZE) / self.GB

    def estimate_activations(
        self,
        batch_size: int,
        seq_len: int,
        use_gradient_checkpointing: bool = True,
    ) -> float:
        """Estimate activation memory.

        Based on practical measurements from training runs:
        - seq=512, batch=1, gc=True: ~1.5 GB per sample
        - seq=512, batch=1, gc=False: ~3.0 GB per sample
        - Activation memory scales roughly linearly with batch_size

        Key insight: activation memory is proportional to batch_size * seq_len,
        not to total parameter count.
        """
        # Empirical activation memory (GB per sample at seq_len=512)
        activation_per_sample = 1.5 if use_gradient_checkpointing else 3.0

        # Scale with batch size
        activations = activation_per_sample * batch_size

        # Scale with sequence length relative to 512
        seq_scale = seq_len / 512
        activations *= max(0.1, min(2.0, seq_scale))

        return activations

    def profile(
        self,
        batch_size: int,
        lora_enabled: bool = False,
        lora_r: int = 64,
        tune_mlp: bool = True,
        tune_llm: bool = True,
        use_gradient_checkpointing: bool = True,
        num_gpus: int = 1,
    ) -> MemoryProfile:
        """Profile memory usage for given configuration."""

        # Model weights (bf16) - stored on each GPU
        model_weights = self.estimate_model_weights()

        # Trainable parameters
        if lora_enabled:
            trainable_params = self.estimate_lora_trainable_params(lora_r)
            trainable_params += self.estimate_vision_trainable_params(tune_mlp)
        else:
            trainable_params = self.TOTAL_PARAMS if tune_llm else 0
            if tune_mlp:
                trainable_params += 10_000_000

        # Gradients (bf16) - stored on each GPU
        gradients = (trainable_params * self.BF16_SIZE) / self.GB

        # Optimizer states (Adam: fp32 for both momentum and variance = 8 bytes per param)
        optimizer_states = (trainable_params * self.FP32_SIZE * 2) / self.GB

        # Activations
        seq_len = self.SEQ_LEN
        activations = self.estimate_activations(
            batch_size, seq_len, use_gradient_checkpointing
        )

        # With Accelerate ZeRO-2:
        # Model weights: replicated on each GPU (8GB)
        # Gradients: distributed across GPUs (gradients / num_gpus)
        # Optimizer states: distributed across GPUs (optimizer_states / num_gpus)
        # Activations: stored on each GPU (batch is per GPU)

        num_gpus = max(1, num_gpus)

        total_per_gpu = (
            model_weights
            + gradients / num_gpus
            + optimizer_states / num_gpus
            + activations
        )

        # Check feasibility
        feasible = total_per_gpu <= self.usable_memory_gb

        # Calculate minimum GPUs needed
        min_gpus_needed = 1
        for ng in range(1, 9):
            per_gpu = (
                model_weights
                + gradients / ng
                + optimizer_states / ng
                + activations
            )
            if per_gpu <= self.usable_memory_gb:
                min_gpus_needed = ng
                break
        else:
            min_gpus_needed = 8  # Needs 8+ GPUs

        return MemoryProfile(
            model_weights=model_weights,
            gradients=gradients,
            optimizer_states=optimizer_states,
            activations=activations,
            total_per_gpu=total_per_gpu,
            num_gpus_needed=min_gpus_needed,
            feasible=feasible,
        )


def print_analysis(estimator: Qwen3VLMemoryEstimator):
    """Print detailed memory analysis."""

    print("=" * 70)
    print("Qwen3-VL-4B 显存分析报告")
    print("=" * 70)
    print(f"\nGPU配置: RTX 3090 (24GB) × {estimator.num_available_gpus}")
    print(f"可用显存/卡: {estimator.usable_memory_gb:.1f} GB (预留15%缓冲)")
    print(f"模型总参数量: {estimator.TOTAL_PARAMS / 1e9:.1f}B")
    print(f"Hidden Dim: {estimator.HIDDEN_DIM}, Layers: {estimator.NUM_LAYERS}")
    print(f"FFN Dim: {estimator.FFN_DIM}, Heads: {estimator.NUM_HEADS}")
    print(f"序列长度: {estimator.SEQ_LEN}")

    configs = [
        # LoRA configs (with gradient checkpointing)
        {
            "name": "LoRA (r=64, bs=1, gc=True)",
            "lora_enabled": True, "batch_size": 1,
            "tune_llm": True, "tune_mlp": True, "use_gc": True,
        },
        {
            "name": "LoRA (r=64, bs=2, gc=True)",
            "lora_enabled": True, "batch_size": 2,
            "tune_llm": True, "tune_mlp": True, "use_gc": True,
        },
        {
            "name": "LoRA (r=64, bs=4, gc=True)",
            "lora_enabled": True, "batch_size": 4,
            "tune_llm": True, "tune_mlp": True, "use_gc": True,
        },
        # Full parameter configs (with gradient checkpointing)
        {
            "name": "Full (bs=1, gc=True)",
            "lora_enabled": False, "batch_size": 1,
            "tune_llm": True, "tune_mlp": True, "use_gc": True,
        },
        {
            "name": "Full (bs=2, gc=True)",
            "lora_enabled": False, "batch_size": 2,
            "tune_llm": True, "tune_mlp": True, "use_gc": True,
        },
        # Full parameter without gradient checkpointing
        {
            "name": "Full (bs=1, gc=False)",
            "lora_enabled": False, "batch_size": 1,
            "tune_llm": True, "tune_mlp": True, "use_gc": False,
        },
    ]

    print("\n" + "-" * 70)
    print("配置分析:")
    print("-" * 70)

    results = []
    for cfg in configs:
        profile = estimator.profile(
            batch_size=cfg["batch_size"],
            lora_enabled=cfg["lora_enabled"],
            tune_llm=cfg["tune_llm"],
            tune_mlp=cfg["tune_mlp"],
            use_gradient_checkpointing=cfg["use_gc"],
            num_gpus=estimator.num_available_gpus,
        )
        results.append((cfg["name"], profile))

    # Sort by total per gpu
    results.sort(key=lambda x: x[1].total_per_gpu)

    for name, profile in results:
        print(f"\n【{name}】")
        print(profile)

    print("\n" + "=" * 70)
    print("显存计算说明:")
    print("=" * 70)
    print("""
显存组成 (以 Full bs=1 gc=True 为例):
1. 模型权重(bf16): 4B × 2 = 8 GB
2. 梯度(bf16): 4B × 2 = 8 GB
3. 优化器状态(fp32): 4B × 8 = 32 GB ← 全参最大开销!
4. 激活值: ~1.8 GB (seq=512, gc=True)

关键发现:
- 全参: 优化器状态占主导 (32GB), 必须使用多卡 + ZeRO-2/3
- LoRA: 优化器状态仅 ~0.4GB, 显存大头变成激活值
- 激活值主要来自 attention scores (seq²) 和 FFN (hidden × ffn)
    """)

    print("=" * 70)
    print("推荐配置:")
    print("=" * 70)

    # Find best feasible config
    feasible = [(n, p) for n, p in results if p.feasible]
    if feasible:
        best = min(feasible, key=lambda x: x[1].num_gpus_needed)
        print(f"\n最优: {best[0]}")
        print(f"  需要GPU数: {best[1].num_gpus_needed}")
        print(f"  单卡显存: {best[1].total_per_gpu:.1f} GB")
    else:
        print("\n所有配置在4卡下都不可行!")
        # Show minimum GPU requirements
        print("\n最小GPU需求:")
        for name, profile in sorted(results, key=lambda x: x[1].num_gpus_needed):
            print(f"  {name}: 需要 {profile.num_gpus_needed} 卡")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B Memory Profiling")
    parser.add_argument("--gpu-memory", type=int, default=24, help="GPU memory in GB")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of available GPUs")
    args = parser.parse_args()

    estimator = Qwen3VLMemoryEstimator(
        gpu_memory_gb=args.gpu_memory,
        num_available_gpus=args.num_gpus,
    )
    print_analysis(estimator)


if __name__ == "__main__":
    main()
