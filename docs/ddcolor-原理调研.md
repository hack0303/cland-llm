---
title: DDColor 原理/技术调研（架构 · 训练 · 对比 · 边界）
summary: 面向选型/复现/落地的 DDColor 原理文档：Dual Decoder + Color Query 架构、训练/数据/损失/分辨率、
  vs DeOldify/SDXL img2img 对比、落地边界（老照片/人脸/低清、D2 保 work 亮度为何优）、最小复现与指标口径。
read_when: 选型或评审照片上色模型、调整 colorize 链路、排查串色/蓝块/亮度漂移、复现 DDColor、
  设计上色评测指标、阅读 base/cland-crawler#226 时。
tags: [ddcolor, colorize, photo, 上色, research, ddcolor原理]
owner: turing
status: "2026-10-02 · v1 初稿（turing 组织；复现最小实验待 hopper 独立复核、评测数据待 curie 补全）"
issue: base/cland-crawler#226
---

# DDColor 原理/技术调研

> 关联：#226（本调研）· #225（四路验证，D2 已采纳）· `docs/photo-colorize-offset-225.md` · `docs/photo-retouch-v0.md`
> 本文由 **turing** 定题/组织并出结论；复现最小实验在本会话执行（对应 hopper 分工，待其独立复核）；
> 评测指标口径与基线对比见 `docs/ddcolor-评测-指标口径与基线对比.md`（curie，进行中）；落地侧输入给 sage。

## 0. 问题

Owner：「DDColor 有专门的**模型技术文档**吗，**原理文档**、**调研文档**？」

现状：DDColor 在 cland-llm 仅以**操作级**出现（#225 验证 + modelscope 权重引用），**无原理/调研文档**。
本稿要回答 5 个问题：

1. **架构**：DDColor 的 Dual Decoder + Color Query 是什么，为何**结构保真好**、颜色不串位？
2. **训练**：数据、损失、输入/推理分辨率、优化设置。
3. **对比**：DDColor vs DeOldify vs SDXL img2img 上色。
4. **边界**：老照片/人脸/低清场景的失效边界；权重来源与许可；产线 **D2（保 work 亮度）**为何更优。
5. **复现**：最小复现步骤 + 指标口径。

## 1. 结论先行

- **一句话**：DDColor 是 ICCV 2023 的**端到端双解码器**上色模型——用 **ConvNeXt 编码器 + Pixel Decoder 恢复全分辨率空间结构**，再用 **Color Decoder 以 100 个可学习 color query 经多尺度 cross-attention** 学到语义感知的颜色表示，**只预测 AB 色度、亮度 L 直接沿用输入**；其优势集中在**分布自然度 + 空间伪影抑制 + 结构保真**——Track B（#226，n=8 真实黑白老照片）相对 SDXL-0.80 显著更少蓝青伪影（`skin_blue%` 中位 **0.25 vs 13.8**，Wilcoxon p=0.023、Cliff δ=−0.84）、亮度保真更好（`L_shift` **0.139 vs 0.377**，p=0.008）；且**单张仅 0.1–0.2s（P40）+ 1.1–1.8GB 显存**，是本项目老照片上色的**首选专用模型**。**置信度：高**（论文 + 官方源码 + 本地复现 + Track A/B 评测）。
- **口径修正（Track A，有 GT，n=6）**：DDColor **并非在单图色度还原上优于 SDXL**——参考色差 `ΔE00` 中位 **14.17**，反不如 SDXL-0.45 的 **8.86**（灰度直通 6.97 更低）。DDColor 强在**生成语义合理、分布自然的颜色**，而非还原该图本来的颜色；**半彩/褪色图 → 忠实复原原色**场景应改用**低强度 SDXL（0.45，偏灰）**。Track A 为方向性证据（n=6 不做 Wilcoxon）。
- **DeOldify**：本机**未实测**（权重源 `data.deepai.org`/HF 不可达），论文数值仅参照，**待验证**。
- **产线口径**：继续用 **D2 = DDColor 出色 + `chroma_transfer` 保 work 亮度**（#225 经 Owner 确认）。D2 的价值在于**把颜色与 work 亮度解耦、由 work 图决定结构与质感**（`colorfulness`/`edge_align` 与原生一致），且显著优于 SDXL；「D2 在 `L_shift` 上稳定优于原生」仅在 #225 单素材成立，n=8 未复现（详见 §5.3，降级为待验证）。
- **落地边界**：适合**历史黑白老照片 / 结构清晰的场景**；**透明/半透明物体**是论文承认的失败场景；**低清/小目标**受训练分辨率 256 限制，建议先超分再上色；**无用户可控性**（不支持文本/涂鸦引导），需要可控上色时要另外设计。

## 2. 架构原理（①）

### 2.1 总览：Encoder + Dual Decoder

论文（ICCV 2023, arXiv 2212.11613）与官方代码一致，前向流程如下：

```
灰度图 x_L (H×W×1)
   │
   ▼
[Encoder] ConvNeXt-L  ── 4 级特征 1/4,1/8,1/16,1/32
   │                                   │
   │            ┌──────────────────────┘
   ▼            ▼
[Pixel Decoder] 3× UnetBlock(skip) + PixelShuffle×4   ← 恢复全分辨率空间结构
   ├── out0 (1/16) ──┐
   ├── out1 (1/8)  ──┤→ 多尺度特征 [F1,F2,F3]
   └── out2 (1/4)  ──┘        │
                              ▼
                    [Color Decoder] 9×CDB
                    (cross-attn → self-attn → FFN)
                    K=100 color queries ──► 颜色嵌入 E_c (K×C)
                              │
   E_i (pixel decoder 全分辨率嵌入 C×H×W) ──┐
                                          ▼
                              Fusion:  E_c · E_i → Conv1×1 → ŷ_AB (2×H×W)
                                          │
              输入 L ──────────────────────┴──► concat → 最终彩色图
```

- **编码器**：ConvNeXt-L（`basicsr/archs/ddcolor_arch_utils/convnext.py`），4 个 hook（norm0..norm3）产出 1/4、1/8、1/16、1/32 分层特征。选 ConvNeXt 是因它给**层次化语义表征**；论文指出 ResNet/Swin 等任何能产层次特征的骨干均可替换。
- **双解码器**：**Pixel Decoder 管空间**（分辨率/结构），**Color Decoder 管语义颜色**。二者在 fusion 模块用**点积**耦合，再经 1×1 卷积出 AB。
- **只输出 AB**：训练配置 `num_output_channels: 2`，模型只回归 2 个色度通道；最终图 = `concat(输入 L, 预测 AB)`。**亮度不经过网络**，这是「保结构」的第一性来源。

### 2.2 Pixel Decoder：为什么保结构

- 4 级逐步上采样，每级含 **PixelShuffle 上采样层 + 与编码器对应 stage 的 shortcut 卷积**（论文 §3.2.1；代码 `DuelDecoder.make_layers`）。
- 与「反卷积/插值」不同，PixelShuffle 把 `(h/p, w/p, c·p²)` 重排成 `(h, w, c)`，**不引入棋盘伪影**，且保留完整特征金字塔。
- 论文强调：这使模型拿到**完整 image feature pyramid**（这是 ColTran/CT2 等单尺度 transformer 方法没有的），多尺度特征再喂给 Color Decoder 指导 color query 优化。
- **结构保真的本质**：① 输出的 L 直接取输入，不经网络；② AB 是在**全分辨率 image embedding E_i** 上点积生成的，颜色天然落在每个空间位置上，边界对齐。

### 2.3 Color Decoder + Color Query：为什么颜色不串位

- 采用 **query-based transformer decoder**（思路源自 DETR/Mask2Former）。维护 `K=100` 个**可学习 color query**（`query_feat`/`query_embed`，训练时初始化为 0）。
- 每个 **CDB 块**顺序为：**cross-attention（query 读图像特征）→ self-attention（query 间交互）→ FFN**，且论文特意**把 cross-attention 放在 self-attention 之前**——因为 query 初始为 0、语义独立，先与视觉特征建立关联才有意义。
- **多尺度轮询**：3 个尺度（1/4、1/8、1/16）各 3 个 CDB 为一组，轮询 `M` 次；论文 M=3 → 共 **3M=9 个 CDB**（代码 `dec_layers=9`）。多尺度让 query 同时捕捉高层语义与低层边界线索。
- **为什么减少 color bleeding**：多尺度 cross-attention 让颜色嵌入对「语义边界」更敏感；消融（表 2b）显示单尺度 FID 4.44–5.09，多尺度降到 **3.92**、ΔCF 0.47→**0.05**。可视化（论文 Fig.7）显示**每个 query 专精特定语义区域**（如狗的额头/毛发/草地），得到一致的语义级配色，而非逐像素乱涂。
- **与「直接回归 a/b」的区别**：朴素 U-Net/回归在颜色多模态不确定时趋向**均值（偏低饱和/错色）**；分类式方法（313 bins）需**手工先验**且分辨率受限。DDColor 用**可学习自适应调色板（100 个 query）+ 多尺度语义注意力**，无需手工先验，端到端学到语义感知配色。

### 2.4 Fusion 模块

把 pixel decoder 的逐像素嵌入 `E_i (C×H×W)` 与 color decoder 的语义颜色嵌入 `E_c (K×C)` 做**点积**得到 `F̂ = E_c · E_i (K×H×W)`，再 1×1 卷积出 `ŷ_AB ∈ R^{2×H×W}`（论文式 6–7；代码 `refine_net` 额外 concat 输入 3 通道，输入 103 通道 → 输出 2 通道）。

## 3. 训练 / 数据 / 损失 / 分辨率（②）

> 来源：论文 §3.4、§4.1；官方 `options/train/train_ddcolor.yml`；`basicsr/models/color_model.py`；`basicsr/data/lab_dataset.py`。

### 3.1 数据

| 项 | 值 |
|----|----|
| 训练集 | **ImageNet**，1.3M 训练 / 50k 验证（消融用 val5k） |
| 泛化测试 | COCO-Stuff、ADE20K（**不微调**直接测） |
| 训练分辨率 | **256×256**（`gt_size: 256`） |
| 数据增强 | 水平翻转、**Color Augmentation**（BigColor 方案，`color_enhance_factor: 1.2`），可选 CutMix/FMix（默认关） |
| 输入 | RGB→Lab，只把 **L**（灰度的 L）喂网络，监督目标是 **AB** |

### 3.2 网络

- 骨干 **ConvNeXt-L**（ImageNet-22k 预训练，`encoder_from_pretrain: True`）；tiny 版为 ConvNeXt-T。
- Pixel Decoder 4 级特征维度：**512, 512, 256, 256**。
- Color Decoder：**M=3 组 × 3 CDB = 9 层**，**K=100** queries，3 个尺度。
- 判别器：`DynamicUNetDiscriminator`（PatchGAN，nf=64）。
- 参数量：**DDColor-L 227.9M / DDColor-T 55.0M**（论文表 1；本机实测 L=**227.9M**，一致）。

### 3.3 损失（4 项加权和）

| 损失 | 形式 | 权重 λ |
|------|------|--------|
| Pixel | L1(预测 AB, GT AB)，逐像素监督 | **0.1** |
| Perceptual | VGG16-bn 特征 L1（conv1_1..conv5_1 加权 0.0625→1.0） | **5.0** |
| Adversarial | PatchGAN vanilla GAN | **1.0** |
| **Colorfulness** | `L_col = 1 − [σ_rgyb(ŷ) + 0.3·μ_rgyb(ŷ)] / 100`（Hasler–Süsstrunk 色彩度） | **0.5** |

`L_θ = 0.1·L_pix + 5.0·L_per + 1.0·L_adv + 0.5·L_col`。
**Colorfulness loss 是论文新引入项**：鼓励更饱和/悦目的配色，是 DDColor 色彩明显比 DeOldify 足的原因之一；但作者在 model zoo 备注中指出它**有时会产生不合理的色块（如红色伪影）**，`ddcolor_artistic` 即去掉了该项。

### 3.4 优化设置

- 优化器 **AdamW**，lr **1e-4**，β=(0.9, 0.99)，weight decay 0.01。
- **MultiStepLR**：80k 步衰减 ×0.5，之后每 40k 步衰减一次。
- **400,000 次迭代**，batch size **16**（4×Tesla V100，`batch_size_per_gpu: 4 × num_gpu: 4`）。
- 端到端自监督训练（不需要人工标注颜色）。

### 3.5 分辨率与推理速度

- **训练 256×256**；**推理**模型结构固定吃方形输入。官方 `scripts/infer.py` 默认 `--input_size 512`，产线 `retouch.py` 亦用 **512**。流程是先 resize 到 512 推理 AB，再双线性插值回原图分辨率，与**原图 L** 拼接。
- 论文速度：256×256 下 **25 FPS（T）/ 21 FPS（L）**（V100）；比 ColTran 快约 **96×**。
- **本机 P40 实测**（见 §7）：256→0.094–0.103s/张、峰值显存 1150 MiB；512→0.194–0.203s/张、峰值 1769 MiB；模型加载 6.5s。

## 4. 对比：DDColor vs DeOldify vs SDXL img2img（③）

### 4.1 论文定量对比（ImageNet val5k，论文表 1）

| 方法 | 参数量 | FID ↓ | CF ↑ | ΔCF ↓ | PSNR ↑ |
|------|-------|-------|------|-------|--------|
| CIC | 32.2M | 8.72 | 31.60 | 6.61 | 22.64 |
| InstColor | 69.4M | 8.06 | 24.87 | 13.34 | 23.28 |
| **DeOldify** | 63.6M | **6.59** | **21.29** | **16.92** | 24.11 |
| Wu et al. | 310.9M | 5.95 | 32.98 | 5.23 | 21.68 |
| ColTran | 74.0M | 6.44 | 34.50 | 3.71 | 20.95 |
| CT2 | 463.0M | 5.51 | 38.48 | 0.27 | 23.50 |
| BigColor | 105.2M | 5.36 | 39.74 | 1.53 | 21.24 |
| ColorFormer | 44.8M | 4.91 | 38.00 | 0.21 | 23.10 |
| **DDColor-tiny** | 55.0M | **4.38** | 37.66 | 0.55 | 23.54 |
| **DDColor-large** | 227.9M | **3.92** | 38.26 | **0.05** | 23.85 |

要点：
- **FID 最低**（分布最接近真实图）；**ΔCF 最低**（生成色彩度与 GT 的偏差最小 → 最自然，不虚高）。论文指出 CF 高不等于视觉好，故引入 ΔCF。
- **DeOldify**：FID 6.59、CF 仅 21.29、**ΔCF 高达 16.92** → 输出偏**暗淡、欠饱和**；论文 Figure 1/3 明确批评其 dull & unsaturated。
- DDColor 在 COCO-Stuff / ADE20K（**未微调**）也拿到最低 FID，泛化性好。
- **用户研究**：50 张图 × 20 人被试探，DDColor 偏好率高于 DeOldify/BigColor/CT2/ColorFormer。

### 4.2 消融（论文表 2–3，均 val5k）

| 变量 | 结果 |
|------|------|
| 无 Color Dec / 无 CL | FID 6.04 / CF 33.07 / ΔCF 5.14 |
| 有 CL、无 Color Dec | 5.93 / 36.14 / 2.07 |
| 有 Color Dec、无 CL | 4.01 / 35.69 / 2.52 |
| **两者都有** | **3.92 / 38.26 / 0.05** |
| 单尺度 1/16, 1/8, 1/4 | FID 5.09, 4.49, 4.44；ΔCF 0.99, 0.63, 0.47 |
| **多尺度 3** | **3.92**；ΔCF **0.05** |
| 解码器 self+self | 8.74 / 51.98 / 13.77 |
| cross+cross | 4.55 / 39.93 / 1.72 |
| self+cross | 3.98 / 37.70 / 0.51 |
| **cross→self（论文顺序）** | **3.92 / 38.26 / 0.05** |
| query 数 20/50/100/200/500 | FID 4.02/3.96/**3.92**/3.96/3.93 |

→ **三个设计都必要**：Color Decoder、Colorfulness Loss、多尺度；且 **cross-attention 必须在 self-attention 之前**。

### 4.3 SDXL img2img（#225 单素材 + #226 Track B n=8 统计）

| 变体 | strength | skin_blue% ↓ | colorfulness ↑ | edge_align ↑ | L_shift（vs work）↓ | 耗时 |
|------|----------|------------|---------------|-------------|-----------------|------|
| SDXL colorize 基线 | 0.80 | **4.51** | 6.74 | 0.051 | 0.095 | 120s |
| SDXL 降强度 | 0.45 | 0.00 | 2.13（偏灰） | 0.066 | 0.065 | 67s |
| **DDColor D1 原生** | — | **0.00** | **17.96** | **0.243** | 0.322 | ~20s* |
| **DDColor D2 保 work 亮度** | — | **0.00** | **17.96** | **0.243** | **0.076** | ~20s* |

\* #225 端的「≈20s」含完整链路（预处理/超分/出色）；本会话纯模型推理 512 实测 0.20s/张。

> 统计口径与完整数据见 `docs/ddcolor-评测-指标口径与基线对比.md`（curie，#226）。上表为 #225 **单素材**方向性证据；下表为 #226 Track B **n=8** 真实黑白老照片中位数（配对 Wilcoxon）。

| 变体 | skin_blue% ↓ | whole_blue% ↓ | edge_align ↑ | colorfulness | L_shift ↓ |
|------|-------------|---------------|-------------|--------------|-----------|
| ddcolor_native | 0.243 | 0.98 | 0.079 | 43.65 | **0.109** |
| **ddcolor_transfer（D2）** | 0.245 | 0.97 | 0.079 | 43.75 | 0.139 |
| sdxl_s045_cfg7 | 5.235 | 3.627 | 0.049 | 22.95 | 0.147 |
| sdxl_s080_cfg7 | 13.766 | 10.733 | 0.022 | 54.85 | 0.377 |

**DDColor(D2) vs SDXL-0.80（n=8）**：`skin_blue%` 0.25 vs 13.8（p=0.023，Cliff δ=−0.84）、`whole_blue%` 0.97 vs 10.73（p=0.008）、`L_shift` 0.139 vs 0.377（p=0.008，δ=−1.0）→ **均显著**。

**结论（分口径）**：
- **空间伪影 + 结构保真 + 色彩量级**：DDColor(D2) 显著优于 SDXL-0.80（#225 单素材 + Track B n=8 一致）→ **老照片上色主选 DDColor(D2) 成立**。
- **单图色度还原**：DDColor **不占优**（Track A ΔE00 14.17 vs SDXL-0.45 8.86）；需忠实原色 → 低强度 SDXL。
- **edge_align 不显著**（p=0.20）→ 不作为主判据。
- **DeOldify** 本机未实测，论文数值仅参照，**待验证**。
- 小样本（Track A n=6、Track B n=8）为方向性/初步显著，正式阈值待扩样（curie §7）。

### 4.4 横向小结

| 维度 | DDColor-L | DeOldify | SDXL img2img |
|------|-----------|----------|--------------|
| 任务定位 | 专用灰度上色 | 专用灰度上色（GAN） | 通用文生图/img2img |
| 结构保真 | **高**（L 直通 + 全分辨率 AB） | 中 | 低（重绘，0.8 重绘出蓝块） |
| 色彩丰富度 | **高**（CF 38.26 / ΔCF 0.05） | 低（CF 21.29 / ΔCF 16.92） | 可调但控性差 |
| 语义/串色 | **好**（多尺度 query） | 一般 | 差（strength 高时） |
| 速度/显存 | **0.1–0.2s / 1.8GB（P40）** | 较慢（GAN 后处理多） | ~120s / 大 |
| 可控性 | 无 | 无 | 有（prompt/strength） |
| 依赖 | 本机权重即可 | 需接仓库/权重 | 本机 SDXL 服务 |

## 5. 落地边界（④）

### 5.1 权重来源与许可

| 模型 | 来源 | 训练数据 | 说明 |
|------|------|----------|------|
| `ddcolor_modelscope`（**默认**） | HF `piddnad/ddcolor_modelscope` / ModelScope `damo/cv_ddcolor_image-colorization` | ImageNet（BigColor 清洗方案） | 非 ImageNet 图定性最好，FID 略降；**产线用这个** |
| `ddcolor_paper` | HF `piddnad/ddcolor_paper` | ImageNet | 复现论文图用 |
| `ddcolor_artistic` | HF `piddnad/ddcolor_artistic` | ImageNet + 私有艺术图 | **训练未用 colorfulness loss**，色块伪影更少，配色更多样 |
| `ddcolor_paper_tiny` | HF `piddnad/ddcolor_paper_tiny` | ImageNet | 最轻量（ConvNeXt-T，55M） |

- **代码许可 Apache-2.0**（仓库 `LICENSE`）；**权重许可 Apache-2.0**（HF 模型卡 `license: apache-2.0`，经 hf-mirror 核实）。
- 落盘（本项目）：`/mnt/data/ai_workspace/models/ddcolor/ddcolor_modelscope.pt`，**sha256 `17c460d7e55b32a598370621d77173be59e03c24b0823f06821db23a50c263ce`，911,950,059 字节（~912MB）fp32**；已登记 OpenCMDB 模型资产 **`ast-1790952748239`**（sage，模板 `tmpl-1790952684373` AI Model），许可已在 `cland-llm/docs/models.md` 校正为 **Apache-2.0**（commit `0f79cf8`）。
- 仓库（本项目）：`/mnt/data/ai_workspace/DDColor`（= 官方 `piddnad/DDColor`，本机快照 commit `2adb63f`）。

### 5.2 场景失效边界

| 场景 | 结论 | 依据 |
|------|------|------|
| **历史黑白老照片** | **推荐**。论文 §4.5 专门验证；本项目 #225 实测最优 | 论文 Fig.8 / #225 |
| **人脸** | 基本 OK；DDColor 结构对齐好（edge_align 0.243），但肤色仍可能偏；产线可叠 `skin_fix` | #225 / #179 |
| **低清 / 小目标** | **受限**。训练 256、推理 512，AB 插值回原图不会新增颜色细节；小目标/细结构易串色 | 论文 §3.5 / 消融 |
| **透明/半透明物体** | **论文承认的失败场景**，会产生 visual artifacts | 论文 §4.6 Fig.9 |
| **需要可控配色** | **不支持**。无文本/涂鸦/参考图引导，是论文指明的 future work | 论文 §4.6 |
| **彩色褪色照片** | 非目标场景（设计上是灰度输入），褪色彩图应先转灰度或用修复链路 | 架构（L 直通） |
| **艺术/动漫场景** | `ddcolor_artistic` 可用（README 有动漫风景样例），但配色偏多样、非写实 | model zoo |

工程建议：**低清老照片先超分（RealESRGAN）再上色**（产线 `prepare()` 已这么做）；透明物体、需控色场景另走 SDXL / 参考上色路线。

### 5.3 D2（保 work 亮度）为何优

**机制**：DDColor 与 D2 都只取网络预测的 **a/b** 色度、亮度由输入决定；差异在色度如何渲染。DDColor 原生管线在 Lab→RGB 时因色度饱和导致**色域裁剪**，会给测量到的 L 带来漂移；D2 把色度搬到 work 的 L 上并在 8bit Lab 内夹取。

> **受控实验（本会话）**：纯 Lab 往返（灰度、无上色）`L_shift` = **0.000**，排除量化误差；在保留 L 的前提下注入强色度（LAB2BGR 越界裁剪）升到 **12.5**，支持「色度→色域裁剪→L 漂移」链条。

**实测（两口径不一致，须分列）**：

| 口径 | DDColor 原生 | D2 保 work 亮度 |
|------|-------------|----------------|
| #225 单素材 hist01 | 0.322 | **0.076**（D2 明显更保真） |
| #226 Track B n=8 中位 | **0.109** | 0.139（D2 略高，量级相当） |
| SDXL-0.80（Track B 参照） | — | 0.377（两者均远优于它） |

→ **稳健结论**：D2 的价值在于**把颜色与 work 亮度解耦、由 work 图决定结构与质感**（`colorfulness`/`edge_align` 与原生一致），并显著优于 SDXL；**「D2 在 L_shift 上稳定优于原生」仅在 #225 单素材成立，n=8 未复现 → 降级为待验证**。Owner 采纳 D2 的方向不变（结构/质感可控 + 伪影显著更少）。

**置信度**：D2 显著优于 SDXL-0.80 → 高（n=8 配对显著）；D2 vs 原生 L_shift 优劣 → 待验证（口径/样本敏感）。

## 6. 最小复现 + 指标口径（⑤）

### 6.1 环境

| 项 | 值 |
|----|----|
| Python | 3.13（miniconda3 base） |
| torch / torchvision | **2.7.1+cu118** |
| opencv | 5.0.0（仓库 requirements 锁 4.7.0.72，本项目环境实际 5.0.0 可用） |
| GPU | **Tesla P40（24GB）**，CUDA 可用 |
| 代码 | `/tmp/DDColor`（= `/mnt/data/ai_workspace/DDColor`，commit `2adb63f`） |
| 权重 | `/mnt/data/ai_workspace/models/ddcolor/ddcolor_modelscope.pt`（sha256 见 §5.1） |

### 6.2 最小复现命令

```bash
# ① 纯模型推理（官方脚本；model_path 指向本地权重，input_size 512）
cd /mnt/data/ai_workspace/DDColor
CUDA_VISIBLE_DEVICES=0 python3 scripts/infer.py \
  --model_path /mnt/data/ai_workspace/models/ddcolor/ddcolor_modelscope.pt \
  --input ./assets/test_images --output /tmp/ddcolor_out --input_size 512

# ② 产线口径（D2）：DDColor 出色 → chroma_transfer 保 work 亮度
cd /mnt/data/ai_workspace/cland-llm
python3 inference/photo/tools/run_ddcolor.py \
  --input <黑白老照片> --outdir /tmp/ddcolor_d2 \
  --ddcolor-repo /mnt/data/ai_workspace/DDColor \
  --model-path /mnt/data/ai_workspace/models/ddcolor/ddcolor_modelscope.pt

# ③ 本会话最小复现（计量耗时/显存；脚本见「证据」）
CUDA_VISIBLE_DEVICES=0 python3 /tmp/ddcolor_repro/repro2.py
```

### 6.3 指标口径

**Track A · 有 GT（合成灰度上色）**：PSNR / SSIM / LPIPS / ΔE00（全局色差）/ BW-ΔE00（边界加权，抓串色）/ MI（错位指数 = ΔE_raw/ΔE_warp，>1 说明误差更多来自位置偏移）/ colorfulness。
**Track B · 无参考（真实黑白老照片）**（承接 #225）：`skin_blue%`（主指标，皮肤掩膜内蓝青占比）/ `whole_blue%` / `edge_align`（色度梯度-亮度梯度相关，越高越贴结构）/ `bleed_ratio`（平滑区串色）/ `colorfulness` / `L_shift`（相对输入亮度偏移）。
**论文口径**：FID（分布相似度，越低越好）、CF（色彩度，越高越足）、**ΔCF**（生成与 GT 色彩度差，越低越自然）、PSNR（仅参考，像素指标不反映上色质量）。

> curie 已建 `inference/photo/tools/bench_colorize.py`（Track A/B 双轨脚本），基线含 gray / ddcolor_native / ddcolor_transfer / sdxl_s045,060,080 / deoldify（预留）。**评测数据待其跑出补齐**。

### 6.4 本会话复现结果

```
torch 2.7.1+cu118 / Tesla P40
[input=256] Audrey Hepburn        (1210,915)  0.094s
[input=256] Migrant Mother 1936   (1247,1000) 0.103s   load 6.51s  peak 1150.8 MiB
[input=512] Audrey Hepburn        (1210,915)  0.194s
[input=512] Migrant Mother 1936   (1247,1000) 0.203s   load 6.57s  peak 1769.1 MiB
params 227.9M（与论文 DDColor-L 227.9M 一致）
```

产物：`/tmp/ddcolor_repro/out/*.png`（256/512 各 2 张）。目视核对：肤色、花卉裙、绿叶、棕色冰箱等配色自然、边界贴合，无蓝块串色。

## 7. 证据

| 证据 | 类型 | 位置 | 说明 |
|------|------|------|------|
| 论文全文 | 一手论文 | arXiv 2212.11613 / ICCV 2023 | 架构、损失、表 1/2/3、用户研究、失败案例；本会话抓取 ar5iv 全文 `/tmp/ddcolor_paper.txt` |
| 官方源码 | 一手代码 | `github.com/piddnad/DDColor` @ `2adb63f` | `/tmp/DDColor`（`ddcolor/model.py`、`pipeline.py`、`options/train/train_ddcolor.yml`、`basicsr/*`） |
| 权重模型卡 | 一手来源 | HF `piddnad/ddcolor_modelscope`（经 hf-mirror） | `license: apache-2.0` |
| 本会话复现 | 实验 | `/tmp/ddcolor_repro/` | 256/512 耗时、显存、227.9M 参数、L_shift 0.319 |
| #225 四路验证 | 实验 | `docs/photo-colorize-offset-225.md` | DDColor vs SDXL 全指标；D1/D2 对比 |
| P40 产线实测 | 实验 | CHANGELOG #225 | 上色链路 ~20s/张、权重 912MB |

## 8. 复现（环境/命令/数据）

- 环境/命令见 §6.1–6.2；本会话脚本 `/tmp/ddcolor_repro/repro.py`、`repro2.py`，结果 `/tmp/ddcolor_repro/results.json`。
- 数据：官方 `assets/test_images/`（PD/历史照片 13 张）+ 本项目 `materials/historical/hist01_migrant_mother.jpg`（PD）。
- 固定项：权重 sha256、仓库 commit `2adb63f`、input_size、CUDA_VISIBLE_DEVICES 均已在文中标注。
- 待 hopper 复核：在干净环境按 §6.2 从零跑通，确认耗时/显存/输出可复现。

## 9. 局限与待验证

1. **无训练侧实证**：本文训练设置来自论文与配置文件，**本机未重训**；数据清洗（BigColor 方案）细节未展开。
2. **SDXL 对比**：#225 单素材 + **#226 Track B n=8**（curie）已出统计；Track A 有 GT 的定量对比已补（`docs/ddcolor-评测-指标口径与基线对比.md`），n=6 为方向性证据、待扩样。
3. **DeOldify 未本机实测**（bench 脚本预留），论文数值为官方基线，未与本地 P40 速度/显存对齐。
4. **D2 vs 原生 `L_shift` 优劣待验证**：仅 #225 单素材成立，Track B n=8 未复现（见 §5.3）；D2 显著优于 SDXL 仍成立。
4. **`input_size` 256 vs 512 的定量差异**未量化（本会话仅记录速度/显存）；待补 PSNR/ΔE00 对比。
5. **权重来源可追溯性**：`ddcolor_modelscope.pt` 的具体转换链路（HF↔ModelScope）与精度（fp32）已核实，但**上传时间/版本号**未取到（网络受限）。
6. 论文是 **2022-12 arXiv / 2023 ICCV**；后续若有更新版/衍生模型未纳入。

## 10. 落地建议

| 对象 | 建议 | 验证方式 |
|------|------|----------|
| **sage（AI 工程）** | 产线 colorize 继续用 **D2**（`--colorizer ddcolor` + `chroma_transfer`）；默认 `input_size=512`；低清先超分 | 跑 `bench_colorize.py` Track B，确认 `skin_blue%=0`、`L_shift≤0.1`、`colorfulness` 不回退 |
| **curie（评测）** | 固化 Track A/B 口径；Track A 用 Met/合成退化对拍集出 PSNR/SSIM/LPIPS/ΔE00；Track B 沿用 #225 六指标 | 脚本 `bench_colorize.py` 跑通并固化基线值 |
| **hopper（复现）** | 独立按 §6.2 复核耗时/显存/输出；核对 `input_size` 256/512 质量差 | 两环境结果一致即通过 |
| **产品/选型** | 老照片上色**主选 DDColor-L**；需可控配色/透明物体场景**不适用**，另设计 | 边界场景测试集 |
| **法务/合规** | 代码与权重均 Apache-2.0，可商用；保留 LICENSE 与模型卡引用 | LICENSE 随分发 |

## 11. 数据来源

1. Kang et al. *DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders.* ICCV 2023, pp. 328–338. arXiv:2212.11613 — https://arxiv.org/abs/2212.11613
2. 官方实现 `piddnad/DDColor`（Apache-2.0）— https://github.com/piddnad/DDColor （本机快照 commit `2adb63f`）
3. 权重 `ddcolor_modelscope`（Apache-2.0）— https://huggingface.co/piddnad/ddcolor_modelscope ；ModelScope `damo/cv_ddcolor_image-colorization` — https://www.modelscope.cn/models/damo/cv_ddcolor_image-colorization
4. Hasler & Süsstrunk. *Measuring colorfulness in natural images.* SPIE 2003（colorfulness 指标与损失）
5. 本地实验：`/tmp/ddcolor_repro/`、`docs/photo-colorize-offset-225.md`、`inference/photo/tools/run_ddcolor.py`、`inference/photo/tools/bench_colorize.py`

---

### 附：本调研分工与状态

| 角色 | 分工 | 状态 |
|------|------|------|
| **turing** | 定题/拆解/组织；架构+训练+对比+边界+结论 | ✅ 本稿 v1 |
| **hopper** | 读论文+源码，最小复现（独立复核） | ⏳ 待复核（本会话已跑一版） |
| **curie** | 评测指标口径 + 基线对比（Track A/B） | ⏳ 脚本已建，数据待补 |
| **sage** | 落地侧输入 + 产线对接 | ✅ §10 已给建议，待确认 |
