---
title: DDColor 评测 —— 指标口径与基线对比（#226）
summary: 为 DDColor 原理调研固化上色评测口径（Track A 有 GT 参考 + Track B 无参考真实老照片）并出基线数据：
  实测 DDColor 在真实黑白老照片上蓝青伪影 skin_blue 0.25% vs SDXL-0.80 13.8%（p=0.023）、亮度保真 L_shift 0.14 vs 0.38（p=0.008）；
  参考保真（有 GT）SDXL-0.45 最佳而偏灰，DDColor 色彩更足但不还原具体 GT 色。
read_when: 评审/调整老照片上色链路、设计上色评测指标、比较 DDColor/DeOldify/SDXL、阅读 base/cland-crawler#226 时。
tags: [ddcolor, colorize, photo, 上色, evaluation, benchmark, curie]
owner: curie
status: "2026-10-02 · v1（Track A n=6 + Track B n=8 已出数；DeOldify 待接入）"
issue: base/cland-crawler#226
---

# DDColor 评测：指标口径与基线对比（#226）

> 配合 **base/cland-crawler#226**（DDColor 原理/技术调研，turing 组织）。
> 原理/架构/训练/边界见 `docs/ddcolor-原理调研.md`；#225 单素材验证见 `docs/photo-colorize-offset-225.md`。
> 本文职责（curie）：**固化指标口径** + **出 DDColor vs SDXL vs（DeOldify）可比基线**。
> 脚本：`inference/photo/tools/bench_colorize.py`；数据：`/mnt/data/ai_workspace/outputs/colorize-bench-226/`。

## 0. 问题

1. 老照片上色的**评测指标怎么定**才能同时抓住「串色/错位」与「色彩自然度/结构保真」？
2. DDColor 相对 **SDXL img2img**（本机现有链路基线）与 **DeOldify**（行业参照）到底强在哪、弱在哪？
3. 这些差异**是否统计显著**，还是小样本噪声？

## 1. 结论先行（含置信度）

- **口径**：采用**双轨**——**Track A · 有 GT（合成灰度上色，考色度还原/结构）** + **Track B · 无参考（真实黑白老照片，考伪影/保真/自然度）**；每指标给出公式与方向（§2），统计用**配对 Wilcoxon + bootstrap 95% CI + Cliff's δ**（§2.3）。**置信度：高**（可复现脚本 + 落盘数据）。
- **Track B（真实老照片，无 GT，n=8）**：DDColor 相对 SDXL-0.80 **显著更少蓝青伪影**（`skin_blue%` 中位 **0.25 vs 13.77**，Wilcoxon p=0.023、Cliff δ=−0.84）且**结构保真更好**（`L_shift` **0.139 vs 0.377**，p=0.008、δ=−1.0），色彩量级充足（`colorfulness` 43.8）。SDXL-0.80 色彩更高（54.9）**但主要来自蓝块/过饱和**。**置信度：高**（与 #225 单素材结论一致，样本扩到 8 张仍显著）。
- **Track A（有 GT，n=6）**：**DDColor 并不优于 SDXL**——参考色差 `ΔE00` 中位 **14.17**（SDXL-0.45 **8.86**、灰度直通 6.97），色彩最足（`colorfulness` 44.8）。即 DDColor 擅长**生成"合理的"颜色**，而非**还原该图本来的颜色**；SDXL-0.45 参考保真最好但**偏灰**（CF 15.8）。**置信度：中**（n=6 且素材为油画/商品，非老照片；`n<8` 不做 Wilcoxon，仅 bootstrap CI + δ）。
- **DeOldify**：本机**未跑通**（权重源 `data.deepai.org`/HF 从本机不可达），仅有论文数值，**结论待验证**（§5、§7）。
- **一句话给选型**：**老照片上色主选 DDColor（D2）结论成立且更稳**；但若任务是「半彩/褪色图 → 忠实复原原色」，DDColor 会**编造颜色**，应改用低强度 SDXL 或其它保真路线。

## 2. 指标口径（固化）

所有指标在同一 `work` 分辨率（默认长边 512）上计算；每个模型只换「上色引擎」，**其余链路一致**（`fit_for_work` → 出色 → `chroma_transfer` 绑回输入亮度）。脚本落盘每图每变体的预测 PNG，指标可由 PNG **离线重算**（`--recompute`）。

### 2.1 Track A · 有 GT（参考基准）

做法：取彩色 GT → `L` 通道灰度化为模型输入（**输入 L 与 GT L 完全相同**）→ 各模型上色 → 与 GT 逐指标比。这样 PSNR/SSIM 里的结构项被"免疫"，指标反映**色度质量**。

| 指标 | 方向 | 定义 | 抓什么 |
|------|------|------|--------|
| PSNR | ↑ | skimage `peak_signal_noise_ratio(gt, pred, data_range=255)` | 像素保真（参考，不反映上色质量） |
| SSIM | ↑ | skimage `structural_similarity(channel_axis=2)` | 结构相似 |
| LPIPS | ↓ | AlexNet LPIPS（256²，输入归一化到 [−1,1]） | 感知相似 |
| **ΔE00** | ↓ | 均值 CIEDE2000（skimage `rgb2lab` + `deltaE_ciede2000`） | **全局色差**（颜色准不准） |
| **BW-ΔE00** | ↓ | 只在 GT 亮度梯度 **top10%** 结构边界上求 ΔE00 均值 | **边界处串色/错位**（误差集中在轮廓=错位） |
| **MI 错位指数** | →1 | `ΔE00_raw / ΔE00_shift`，用相位相关在 GT/pred 的 a、b 通道估最优**全局平移**后重算 ΔE00，取 `max(1, ·)` | 颜色层是否**整体平移**（>1=可被全局平移解释） |
| colorfulness | 参考 | Hasler & Süsstrunk (2003) | 色彩自然度/饱和度代理 |

> **MI 的实测说明**：本批全部变体 MI=1.00（含 DDColor 与 SDXL）→ **不存在全局颜色层平移**；#225 观测到的"颜色覆盖偏移"是**局部/语义级串色**，应由 `BW-ΔE00` 与 Track B `skin_blue%` 量化，而非全局位移。MI 保留为**下游诊断项**（若未来模型出现整体套色偏移会 >1）。

### 2.2 Track B · 无参考（真实黑白老照片）

承接 #225 六指标（口径**与 `run_ddcolor.py` 的 `blue_artifact_pct` 对齐**）：

| 指标 | 方向 | 定义 | 抓什么 |
|------|------|------|--------|
| **skin_blue%**（主） | ↓ | DWPose 脸/手/前臂掩膜内，Lab 色度 `sat=hypot(a,b)>18` 且 `hue=atan2(b,a)∈(−140°,−40°)` 占比 | **皮肤区蓝青串色**（要求饱和度，避免暗部误判） |
| whole_blue% | ↓ | 同上，全图 | 全图蓝青伪影 |
| edge_align | ↑ | 在输入 `L` 梯度 top10% 结构边缘处，`corr( gL, gChroma(pred) )` | 色度**贴合结构**程度 |
| bleed_ratio | ↓ | 低梯度（平滑）区 `mean(gChroma)/mean(gChroma)` 全图 | 平滑区**无谓串色** |
| colorfulness | 参考 | Hasler & Süsstrunk | 色彩是否偏灰 |
| L_shift | ↓ | `mean(|L_src − L_pred|)`（Lab L） | **结构/质感保真**（越低越好；D2 口径应 ≤0.1 量级） |

> 皮肤掩膜来自复用产线 `retouch.build_skin_mask`（DWPose）；无人物图掩膜为空 → `skin_blue%` 置 0，以 `whole_blue%` 为主。

### 2.3 统计口径

- **每图配对**（同图同输入比较变体），报告**中位数**（抗偏）。
- `n≥8`：**Wilcoxon 符号秩**（p 值）；`n<8`：不做 Wilcoxon，仅报 **bootstrap 95% CI（中位差，10000 次，seed=20261002）** 与 **Cliff's δ**（|δ|≥0.474 为大效应）。
- 门槛：正式结论要求 **Wilcoxon p<0.05 且 CI 不含 0**；仅 CI+δ 的为**方向性证据**。

## 3. 基线定义

| 变体 | 定义 | 备注 |
|------|------|------|
| `gray` | 灰度直通（**下界对照**） | colorfulness≡0；PSNR/SSIM 上界参照 |
| `ddcolor_native` | DDColor-L 原生（ImageNet，512，只回 AB、L 直通） | 权重 `ddcolor_modelscope.pt` |
| `ddcolor_transfer` | DDColor 出色 + `chroma_transfer` 保 work 亮度（**产线 D2 口径**） | #225 已采纳 |
| `sdxl_s045_cfg7` | SDXL img2img colorize strength=0.45 CFG=7，80 步，seed=42 | 低强度（保守） |
| `sdxl_s080_cfg7` | SDXL img2img colorize strength=0.80 CFG=7，80 步，seed=42 | #225 原基线（高强度） |
| `deoldify` | DeOldify Artistic/Stable | 脚本已留接口，本机权重缺失（§5） |

## 4. 结果

### 4.1 Track A · 有 GT（n=6：3 幅油画 + 3 件商品图）

中位数（`summary_A.md`）：

| variant | PSNR↑ | SSIM↑ | LPIPS↓ | ΔE00↓ | BW-ΔE00↓ | MI | colorfulness |
|---|---|---|---|---|---|---|---|
| gray（下界） | 26.00 | 0.969 | 0.153 | **6.97** | 9.77 | 1.0 | 0.0 |
| ddcolor_native | 20.75 | 0.928 | 0.206 | 14.18 | 14.24 | 1.0 | 44.75 |
| ddcolor_transfer | 20.73 | 0.925 | 0.206 | 14.17 | 14.23 | 1.0 | **44.75** |
| sdxl_s045_cfg7 | **25.08** | 0.942 | **0.159** | **8.86** | 13.22 | 1.0 | 15.80 |
| sdxl_s080_cfg7 | 19.28 | 0.830 | 0.452 | 13.16 | 15.56 | 1.0 | 51.10 |

**解读（关键）**：
- **参考色差赢家是"少改"的方法**：`gray`（不加色）ΔE00 最低，其次 `s045`。这说明 Track A 度量的是**还原具体 GT 色**，而**扩散/生成式上色本就不以还原为目**。
- **DDColor 参考色差反而不如 SDXL-0.45/0.80**（14.17 vs 8.86/13.16），但**结构最好于 s080**（SSIM 0.925 vs 0.830）、**色彩远足于 s045**（44.8 vs 15.8）——即 **DDColor = 色彩足 + 结构稳，但颜色是"编的"**。这与论文口径互补：论文用 **ΔCF**（色彩度差）而非逐图像素保真，DDColor 的强项是**分布层自然度**，不是单图复原。
- **统计功效不足**：n=6 不做 Wilcoxon；bootstrap 中位差 CI 多数跨 0（如 ΔE00 DD−s080 CI=[−6.9, +7.6]），故 Track A **不构成对 DDColor 的胜负判定**，只作**方向性证据**。唯一强效应：`s045` 的 colorfulness 显著低于 `s080`（δ=−1.0，CI=[−40.1,−17.0]）。
- **样本外域提醒**：本集是**油画/商品图**，DDColor（ImageNet 训练）在油画上会**过饱和/偏色**（如 pair02 梵高麦田 DDColor ΔE00 24.0 vs gray 11.7）。**不能用该集给老照片选型下定论**。

### 4.2 Track B · 无参考真实黑白老照片（n=8）

中位数（`summary_B.md`）：

| variant | skin_blue%↓ | whole_blue%↓ | edge_align↑ | bleed_ratio↓ | colorfulness | L_shift↓ |
|---|---|---|---|---|---|---|
| ddcolor_native | **0.243** | **0.98** | 0.079 | **0.425** | 43.65 | **0.109** |
| ddcolor_transfer | 0.245 | 0.97 | 0.079 | 0.440 | **43.75** | 0.139 |
| sdxl_s045_cfg7 | 5.235 | 3.627 | 0.049 | 0.695 | 22.95 | 0.147 |
| sdxl_s080_cfg7 | 13.766 | 10.733 | 0.022 | 0.778 | 54.85 | 0.377 |

**DDColor(D2) vs SDXL-0.80 配对检验（n=8，Wilcoxon）**：

| 指标 | DDColor(D2) | SDXL-0.80 | p | Δ 中位 [95% CI] | Cliff δ |
|---|---|---|---|---|---|
| skin_blue% | 0.245 | 13.766 | **0.023** | −24.3 [−24.3, −1.7] | −0.84（大） |
| whole_blue% | 0.97 | 10.733 | **0.008** | −21.8 [−21.8, −3.8] | −0.94（大） |
| L_shift | 0.139 | 0.377 | **0.008** | −0.38 [−0.38, −0.17] | −1.00（大） |
| colorfulness | 43.75 | 54.85 | **0.008** | −20.0 [−20.0, −6.5] | −0.72（大） |
| edge_align | 0.079 | 0.022 | 0.195 | +0.18 [−0.02, +0.18] | +0.31 |

**解读**：
- **DDColor 在"少蓝块 + 保结构 + 色彩足"上同时占优且统计显著**（skin_blue/whole_blue/L_shift 均 p<0.05，大效应），与 #225 单素材结论**一致**、样本扩到 8 张仍成立。这是**老照片主链路选 DDColor 的量化依据**。
- SDXL-0.80 的 colorfulness 更高（54.85），但**主要来自蓝块与过饱和**（skin_blue 13.8%），不是"更好"。
- `s045` 是 SDXL 最保守档：蓝块已降到 5.2%，但色彩进一步偏灰（22.95）——**DDColor 在两项上都优于它**。
- `edge_align` 差异不显著（p=0.20），说明 DDColor 的"更贴结构"优势**弱于**蓝块/保真两项，不宜单独作为主判据。

### 4.3 论文口径对照（DDColor vs DeOldify，ImageNet val5k，论文表 1）

| 方法 | FID↓ | CF↑ | ΔCF↓ | PSNR↑ |
|---|---|---|---|---|
| DeOldify | 6.59 | 21.29 | 16.92 | 24.11 |
| DDColor-tiny | 4.38 | 37.66 | 0.55 | 23.54 |
| DDColor-large | **3.92** | **38.26** | **0.05** | **23.85** |

> 论文口径（FID/ΔCF）与本文 Track A/B **不可直接混用**：FID 是**分布**相似度、ΔCF 是**色彩度差**，都不衡量单图位置保真。本文 Track B 的 `skin_blue%` 是论文未覆盖的**空间伪影**维度。

## 5. DeOldify 本机状态（待验证）

- **未跑通**。原因：官方权重托管在 `data.deepai.org`，从本机**不可达**（连接超时）；`huggingface.co` 亦不可达。`github.com` 可达但仓库不含权重（仅 README 指向外部）。
- **可用安装路径（已核实）**：PyPI 有可用的 torch≥2.0 兼容分叉 `deoldify==0.0.1`（含内置 fastai，来源 GitHub `leonelhs/DeOldify`，MIT），权重放 `models/ColorizeArtistic_gen.pth`；**只差权重文件**。
- **现状结论**：DeOldify 对比**只有论文数值**，**不得**当作本机实测结论；需在能取到权重的环境补跑后再回填 `deoldify` 变体。

## 6. 复现

**环境**：Tesla P40 ×2（用 GPU0）· Python 3.13 · torch 2.7.1+cu118 · torchvision 0.22.1+cu118 · CUDA 11.8 · skimage 0.26.0 · OpenCV 5.0.0 · lpips 0.1.4 · 本机 SDXL fp16、DDColor-L。

**资产**：
- DDColor 仓库 `/mnt/data/ai_workspace/DDColor` @ `2adb63f`；权重 `models/ddcolor/ddcolor_modelscope.pt`（sha256 前缀 `17c460d7e55b32a5`，870MB）。
- SDXL `models/stable-diffusion-xl-base-1.0-fp16`。
- Track A 清单 `/mnt/data/ai_workspace/outputs/photo-retouch/materials/bench-a-226.json`；Track B 图源 `assets/test_images` 的 `Abandoned Boy/Audrey Hepburn/Einstein/Helen Keller/Louis Armstrong/Detroit 1915/Buffalo Bank/February 1936` 8 张（PD）。

**命令**：

```bash
cd /mnt/data/ai_workspace/cland-llm

# 完整跑（Track A + Track B）
python3 inference/photo/tools/bench_colorize.py \
  --track-a-manifest /mnt/data/ai_workspace/outputs/photo-retouch/materials/bench-a-226.json \
  --track-b-dir "/mnt/data/ai_workspace/DDColor/assets/test_images" \
  --track-b-list /tmp/track-b-list-226.json \
  --variants gray,ddcolor_native,ddcolor_transfer,sdxl_s045_cfg7,sdxl_s080_cfg7 \
  --out /mnt/data/ai_workspace/outputs/colorize-bench-226 --work-res 512

# 只从已落盘 PNG 重算指标（改口径/修 bug 后回填，不重跑模型）
python3 inference/photo/tools/bench_colorize.py --recompute \
  --track-b-dir "/mnt/data/ai_workspace/DDColor/assets/test_images" \
  --out /mnt/data/ai_workspace/outputs/colorize-bench-226 --work-res 512
```

**产物**：`/mnt/data/ai_workspace/outputs/colorize-bench-226/`（`per_image.json` 全量逐图指标 + `summary_A.{md,json}` / `summary_B.{md,json}` 含显著性与每图预测 PNG）。运行日志 `runAB.log`、`runB.log`。

## 7. 局限与待验证

1. **样本量**：Track A n=6、Track B n=8，均低于论文级评测；Track B 已显著但**每类场景覆盖不足**（缺低清/严重划痕/群体人脸）。**待验证**：扩到 ≥30 张分布采样后再定阈值。
2. **Track A 素材外域**：3 幅油画 + 3 商品，非"老照片"。**待验证**：换 Met/PD 彩色老照片做参考集（这也是 turing §10 建议）。
3. **DeOldify 未实测**（§5）——补齐后需与本地 P40 速度/显存对齐，不能只引论文。
4. **MI 全 1**：本批无全局平移；**局部错位**只能由 BW-ΔE00/skin_blue 间接量化，缺"错位像素定位"指标。**可补**：语义掩膜（SAM）内/外的色度误差分解。
5. **`edge_align` 未显著**：单像素梯度相关对噪声敏感；**待验证**：改用多尺度或边界带 ΔE00 更稳。
6. **随机性**：SDXL 固定 seed=42，仅代表单次采样；**待验证**：同图多 seed（≥3）报方差。
7. **Track A 的色度迁移口径**：所有上色输出都做了 `chroma_transfer` 绑回输入 L，故 DDColor-native 与 D2 在 Track A 几乎相同；这**不测**原生输出在 Lab→RGB 的色域裁剪/亮度漂移。

## 8. 落地建议

| 对象 | 建议 | 验证方式 |
|------|------|----------|
| **sage（AI 工程）** | 产线继续用 **D2**；门槛用本文 Track B：`skin_blue%<1`、`L_shift≤0.15`、`colorfulness≥35`、`edge_align` 不劣化 | 每次换代跑 `bench_colorize.py Track B` 出这 4 值 |
| **product/选型** | 老照片上色**主选 DDColor**（证据：Track B 显著）；**半彩/褪色"复原原色"** 场景改低强度 SDXL（Track A 参考保真更好） | 任务分诊：全黑白 vs 需忠实原色 |
| **turing（#226 报告）** | 采纳本文 Track B 数据佐证"结构保真/少串色"；**修正**"全面优于"为"分布自然度与空间伪影优、单图色度还原不占优" | 引用 §4.1/4.2 |
| **hopper（复现）** | 复核 DDColor 指标时用**同一脚本同 work-res**；注意 `edge_align` 需输入 L 作结构基准 | 重跑得到一致中位数量级 |
| **后续评测** | 扩 Track B 到 ≥30 张 + 补 DeOldify + 多 seed，再定正式验收阈值 | 见 §7 |

## 9. 数据来源

1. 论文 Kang et al. *DDColor*, ICCV 2023, arXiv:2212.11613（对比表与 ΔCF 口径）。
2. Hasler & Süsstrunk. *Measuring colorfulness in natural images.* SPIE 2003（colorfulness）。
3. 本地实验：`inference/photo/tools/bench_colorize.py`（本文脚本）、`cmp_colorize_offset.py`、`run_ddcolor.py`、`retouch.py`。
4. 数据落盘：`/mnt/data/ai_workspace/outputs/colorize-bench-226/`（`per_image.json` + `summary_A/B.*` + 逐图 PNG）。
5. #225 `docs/photo-colorize-offset-225.md`（单素材四路验证，口径来源）。
6. DeOldify 权重源：`https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth`（本机不可达）；PyPI `deoldify==0.0.1`（leonelhs 分叉）。
