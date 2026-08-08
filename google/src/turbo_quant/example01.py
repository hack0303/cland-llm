import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def test_polar_quantization(dim=1024, bits=3):
    # 1. 模拟 LLM 的 KV Cache 数据 (高斯分布)
    original_vector = torch.randn(dim)

    # 2. 计算模长 (Magnitude) - 通常用更高精度保留 (e.g., FP8/FP16)
    magnitude = torch.norm(original_vector)

    # 3. 计算单位方向向量 (Direction)
    direction = original_vector / magnitude

    # --- TurboQuant 核心逻辑模拟 ---
    # 4. 将连续的角度空间划分为 2^bits 个区间
    levels = 2**bits
    # 模拟量化：将方向向量投影到离散的格点上
    # 在高维空间，这相当于对超球面的表面进行划分
    quantized_direction = torch.round(direction * (levels / 2)) / (levels / 2)
    # 重新归一化以保持单位方向
    quantized_direction = F.normalize(quantized_direction, dim=0)

    # 5. 还原向量
    reconstructed_vector = magnitude * quantized_direction

    # 6. 精度评估
    cos_sim = F.cosine_similarity(
        original_vector.unsqueeze(0), reconstructed_vector.unsqueeze(0)
    )
    mse_error = torch.mean((original_vector - reconstructed_vector) ** 2)

    print(f"--- {bits}-bit 极坐标量化结果 ---")
    print(f"维度: {dim}")
    print(f"余弦相似度 (1.0 为完美): {cos_sim.item():.6f}")
    print(f"均方误差 (MSE): {mse_error.item():.6f}")


# 运行测试
test_polar_quantization(dim=1024, bits=3)
