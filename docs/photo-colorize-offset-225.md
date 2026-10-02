---
title: 老照片上色「颜色覆盖位置偏移」四路验证（#225）
summary: 对老照片上色颜色错位/蓝块伪影，用同一素材对 A(SDXL调参)/B(色度配准)/C(掩膜约束)/D(DDColor专用模型) 四路做对拍；结论=D(DDColor)最优，A(strength≤0.45)为 SDXL 最小改动兜底。
read_when: 调整老照片上色链路、排查颜色错位/蓝晕伪影、选型上色模型、评审 #225 时。
tags: [photo, colorize, research, 老照片上色, ddcolor, sdxl]
owner: sage
status: "2026-10-02 · 四路验证完成，待评审"
issue: base/cland-crawler#225
---

# 老照片上色「颜色覆盖位置偏移」四路验证（#225）

## 1. 问题

Owner：「老照片上色**脸没问题了**，**手臂/衣服还是有问题，颜色覆盖的位置有偏移**。」

实测（`hist01_migrant_mother`，PD）表现为：手臂/衣物/孩童面部出现**成片蓝色块状串色**，
颜色与结构边界不对齐。现用链路 = SDXL img2img colorize（**strength=0.80**）→ `chroma_transfer`
（只取 AI 出色色度、盖到 work 亮度，**无几何配准**）。

## 2. 方法

- 素材：`materials/historical/hist01_migrant_mother.jpg`（1920×2496，PD），全路同一 work（784×1024）。
- 四路：
  - **A SDXL 调参**：strength 0.80(基线)/0.45/0.35 + CFG 7/5
  - **B 色度层配准**：`chroma_transfer` 前用稠密光流把 SDXL 出色层对齐到 work；再加引导滤波
  - **C 掩膜约束**：DWPose 脸/手/前臂 mask + 躯干/衣物区域色度稳定
  - **D 专用模型**：DDColor-L（ImageNet，modelscope 权重）出色；D2 再按产线口径做色度迁移
- 复现：`inference/photo/tools/cmp_colorize_offset.py`（A/B/C）、`inference/photo/tools/run_ddcolor.py`（D）

指标：
- `skin_blue%` = DWPose 皮肤掩膜内蓝/青色像素占比 → **直接量化手臂/脸「蓝晕错位」**
- `whole_blue%` = 全图蓝/青占比（含合法蓝衣）
- `colorfulness` = 平均色度（自然度代理，越高色彩越足）
- `edge_align` = 结构边缘处色度梯度相关（越高颜色越贴结构）
- `L_shift` = 相对 work 的亮度偏移（保真度代理，越低越保结构）

## 3. 客观结果（同一素材）

| 变体 | 路 | strength | CFG | skin_blue% | whole_blue% | colorfulness | edge_align | L_shift | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| A0 基线 | A | 0.80 | 7 | **4.51** | 2.43 | 6.74 | 0.051 | 0.095 | 120s |
| A1 | A | 0.45 | 7 | **0.00** | 0.00 | 2.13 | 0.066 | 0.065 | 67s |
| A2 | A | 0.35 | 7 | 0.00 | 0.00 | **0.73** | 0.068 | 0.083 | 53s |
| A3 | A | 0.45 | 5 | 0.00 | 0.00 | 1.60 | 0.071 | 0.074 | 67s |
| B1 | B | 0.80 | 7 | 2.83 | 2.47 | 6.73 | 0.073 | 0.094 | 120s |
| B2 | B | 0.80 | 7 | 2.74 | 2.33 | 6.80 | 0.198* | 0.152 | 120s |
| B3 | B | 0.45 | 7 | 0.00 | 0.00 | 2.12 | 0.093 | 0.065 | 67s |
| C1 | C | 0.80 | 7 | 4.51 | 2.43 | 6.75 | 0.053 | 0.118 | 120s |
| C2 | C | 0.80 | 7 | 0.00 | 1.81 | 6.62 | 0.035 | 0.130 | 120s |
| D1 DDColor 原生 | D | — | — | 0.00 | 0.74 | **17.96** | **0.243** | 0.322 | ≈20s |
| D2 DDColor→work 亮度 | D | — | — | 0.00 | 0.74 | **17.96** | **0.243** | **0.076** | ≈20s |

\* B2 的 `edge_align` 经引导滤波机械抬高（滤波本身让色度贴合 L 边缘），**不代表蓝块被消除** —— 视觉上 B1/B2 蓝块仍在（skin_blue% 2.7–2.8）。指标须结合 `skin_blue%` 与目视判读。

## 4. 结论（逐路）

- **A SDXL 调参**：strength 0.80→0.45 可**彻底消除蓝块**（skin_blue 4.51→0）；但色度大幅下降
  （colorfulness 6.74→1.6–2.1），画面偏灰、色彩不足；0.35 更过淡（0.73）。**是零依赖速修，非最优**。
- **B 色度层配准**：在 0.80 下光流配准**不能**消除已生成的蓝块（sdxl 输出本身错色），
  only 在 0.45 上叠加略优（B3 align 0.093 vs A1 0.066），**增益有限**，不推荐单独上。
- **C 掩膜约束**：0.80 下掩膜+区域中值**无改善**（C1 4.51）；C2 叠加现有 `fix_skin_color_cast`
  可清皮肤蓝晕，但**衣物蓝块仍在**（whole_blue 1.81）。受限于无 SAM/衣物分割，**仅能护肤色**。
- **D DDColor（推荐）**：skin_blue 0、色度最足最自然（17.96）、结构对齐最好（align 0.243）、
  串色最低（bleed 0.567）；D2 保 work 亮度（L_shift 0.076）→ **结构与质感无损**。**综合最优**。

## 5. 推荐

1. **主选 D**：老照片上色改用 **DDColor-L**，再按产线口径 `chroma_transfer(DDColor出色, work)`
   保住原图亮度/质感（D2）。权重 ~912MB（modelscope `damo/cv_ddcolor_image-colorization`），
   本机 P40 出色 ≈20s/张（含模型加载，全流程含预处理 ≈60s），无外部服务依赖。
2. **兜底 A**：若短期不动模型，SDXL colorize 预设 **strength 0.80→0.45、CFG 5–7**，
   即时消除蓝块（代价：色彩变淡），可叠加 DWPose `skin_fix`。
3. **暂不采用 B/C 单独方案**：B 增益有限、C 无法覆盖衣物；可作为 D 上的补充后处理。

## 6. 复现

```bash
python3 inference/photo/tools/cmp_colorize_offset.py \
  --input .../materials/historical/hist01_migrant_mother.jpg \
  --outdir /mnt/data/ai_workspace/outputs/photo-retouch-225/hist01
python3 inference/photo/tools/run_ddcolor.py \
  --input .../hist01_migrant_mother.jpg --outdir .../hist01 \
  --ddcolor-repo /tmp/DDColor --model-path /tmp/ddcolor_w/pytorch_model.pt
```

产物（对比图 `compare.png` + `meta.json`）：`/mnt/data/ai_workspace/outputs/photo-retouch-225/hist01/`。
