def turbo_quant_with_qjl(dim=1024, bits=3):
    # --- 1. 原始 3-bit 极坐标量化 (接上一段代码逻辑) ---
    original = torch.randn(dim)
    mag = torch.norm(original)
    vec = original / mag

    q, _ = torch.linalg.qr(torch.randn(dim, dim))
    rotated_vec = q @ vec

    levels = 2**bits
    scale = torch.max(torch.abs(rotated_vec))
    quantized = torch.round(rotated_vec / scale * (levels / 2 - 1))

    # 第一次还原 (3-bit)
    dequantized_3bit = (quantized / (levels / 2 - 1)) * scale
    reconstructed_3bit = q.T @ dequantized_3bit

    # --- 2. QJL 补偿 (1-bit 秘密武器) ---
    # 计算残差 (Residual)
    residual = vec - reconstructed_3bit

    # 1-bit 量化：只保留残差的符号
    # 实际上 TurboQuant 会用一个更小的投影矩阵，这里简化模拟其核心：符号位记录
    qjl_compensation = torch.sign(residual) * torch.mean(torch.abs(residual))

    # 最终还原：3-bit 主干 + 1-bit 补偿
    final_reconstructed = (reconstructed_3bit + qjl_compensation) * mag
    final_reconstructed = F.normalize(final_reconstructed, dim=0) * mag

    # --- 3. 结果对比 ---
    cos_sim_old = F.cosine_similarity(
        original.unsqueeze(0), (reconstructed_3bit * mag).unsqueeze(0)
    )
    cos_sim_new = F.cosine_similarity(
        original.unsqueeze(0), final_reconstructed.unsqueeze(0)
    )

    print(f"维度: {dim}")
    print(f"仅使用 3-bit 极坐标: {cos_sim_old.item():.6f}")
    print(f"叠加 1-bit QJL 补偿后: {cos_sim_new.item():.6f}")


turbo_quant_with_qjl()
