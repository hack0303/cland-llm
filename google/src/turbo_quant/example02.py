import torch
import torch.nn.functional as F


def turbo_quant_sim(dim=1024, bits=3):
    # 1. 模拟原始向量
    original = torch.randn(dim)
    mag = torch.norm(original)
    vec = original / mag

    # 2. 模拟 TurboQuant 的随机旋转 (使用正交矩阵保持距离)
    # 这一步极其重要，它让数据分布均匀，防止量化崩塌
    q, _ = torch.linalg.qr(torch.randn(dim, dim))
    rotated_vec = q @ vec

    # 3. 在旋转空间进行 3-bit 量化 (模拟角度/分量压缩)
    # 我们模拟将每个分量压到 2^bits 个刻度
    levels = 2**bits
    scale = torch.max(torch.abs(rotated_vec))
    quantized = torch.round(rotated_vec / scale * (levels / 2 - 1))

    # 4. 反量化并逆旋转
    dequantized = (quantized / (levels / 2 - 1)) * scale
    reconstructed = q.T @ dequantized
    reconstructed = F.normalize(reconstructed, dim=0) * mag

    # 5. 计算指标
    cos_sim = F.cosine_similarity(original.unsqueeze(0), reconstructed.unsqueeze(0))
    print(f"--- 修正后的 {bits}-bit 模拟 ---")
    print(f"维度: {dim}")
    print(f"余弦相似度: {cos_sim.item():.6f} (这次应该接近 0.9+ 了)")


turbo_quant_sim(bits=3)
