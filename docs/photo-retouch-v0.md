---
title: "AI 修图产线 v0 — 照片修复/超分/上色/商品图（#179）"
summary: "H-01 AI 修图产线 v0：Pillow 预处理 → RealESRGAN 4x 分块超分 → SDXL img2img 低强度修复/上色 → before/after 对比 + PSNR/SSIM 评估；11/11 案例跑通（失败率 0%），含 3 组退化-复原对拍集与 Owner 总览图"
read_when:
  - "运行或维护 AI 修图产线（inference/photo/retouch.py）"
  - "查看 #179 H-01 修图 v0 的效果样例 / 指标 / 耗时基线"
  - "调整修复/上色/商品图模式参数（strength/guidance/prompt）"
  - "复用无人工看图的质量验收方法（PSNR/SSIM + 无参指标 + 总览拼图）"
scope:
  - inference/photo
  - docs
status: "active"
updated: "2026-10-01"
---

# AI 修图产线 v0（H-01 · 工单 #179）

> Owner 2026-10-01：「先实现跑通，我看下效果」——本报告 = **效果样例 + 指标 + 耗时基线**。
> **Owner 速览（先看这个）**：`/mnt/data/ai_workspace/outputs/photo-retouch/owner_overview.png`
> （11 案例 上=输入 / 下=输出；单案例对比图在 `outputs/photo-retouch/<case>/compare.png`）

## 1. 结论摘要

| 验收项 | 结果 |
|---|---|
| 一条命令批量 | ✅ `retouch.py --manifest cases.json`（或 `--input DIR`）一次跑完 11 案例 |
| 失败率 ≤5% | ✅ **0/11 = 0%**（单案例失败不中断批，失败率 >5% 脚本非零退出） |
| 修复样例 ≥5 含对拍 ≥3 | ✅ 修复/上色 6 组（对拍集 5 组：3 组含 GT 逐像素对拍 + 2 组基线） |
| 主图（商品图）≥3 | ✅ 3 组（Met CC0 器皿 → 白底 1024×864/672/688） |
| 单张耗时基线 | ✅ upscale 11-13s · product 26-34s · repair 42-59s · colorize 129-132s（+显存模型加载冷 ~6min/热 56-90s） |
| 素材只用 PD/CC0 | ✅ 8 项来源许可留痕（`materials/manifest.json`，含 sha256） |
| 证据 | 输出目录 + YAML + 本报告 + #179 评论 |

## 2. 管线与参数

```
输入目录 / cases.json
 ① Pillow 预处理    EXIF 转正 → 自动裁剪均匀边框(≤12%) → 色阶 autocontrast / 白点归一(product)
 ② RealESRGAN 4x    RRDBNet 分块推理(256 tile + 16 重叠羽化)，复用 hand_pipe/rrdbnet.py 已验证权重适配
 ③ SDXL img2img     低强度 + mode 提示词（P40 fp16，进程内加载；实际步数 = int(steps×strength)）
 ④ 收尾/合成        colorize: LAB 色度迁移(保结构) · product: 连通域白底 · repair: 黑白输入保持黑白
 → <case>/{before,after,compare}.png + meta.json；run.json + metrics.json + owner_overview.png
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | repair | `repair` 修复 / `upscale` 仅超分 / `colorize` 上色 / `product` 商品白底 |
| `--strength` | 按 mode：repair .25 / colorize .80 / product .25 | img2img 去噪强度（0.25 保结构；上色需 ≥0.8 才有色彩） |
| `--seed` | 42 | 固定可复现（同 seed 同输入输出一致） |
| `--batch` | 1 | 同模式组批送 SDXL（P40 无 Tensor Core，批>1 无加速收益，默认 1） |
| `--steps / --guidance` | 80 / 按 mode（colorize 7.0，其余 4.5） | 总步数；小步长（80×strength）显著保结构 |
| `--work-res / --max-side` | 1024 / 2048 | SDXL 工作分辨率长边 / ESRGAN 输出长边上限 |
| `--esrgan / --esrgan-in-cap` | realesrgan / 1280 | 超分权重（另有 ultrasharp）；x4 输入长边内存保护 |

**关键设计**（经 3 轮参数对照实测，见 §6）：
1. **低强度 + 小步长**：SDXL img2img 在 `strength 0.25 × steps 80` 时结构保持最好（对比 0.25×30、0.55×30 等）；
2. **上色 = 大强度重绘 + 色度迁移**：色度来自 AI（0.8 强度重绘），亮度沿用修复后原图（LAB L 通道）→ 上色不毁脸不漂结构；
3. **黑白修复保持黑白**：repair 模式检测灰度输入后回转 L 模式（上色请走 colorize）。

## 3. 素材（仅公有领域 / CC0）

| 名称 | 用途 | 来源/许可 | 尺寸 |
|---|---|---|---|
| pair01 The Harvesters（Bruegel） | 对拍集 GT | Met Open Access **CC0** | 3811×2809 |
| pair02 Wheat Field with Cypresses（Van Gogh） | 对拍集 GT | Met Open Access **CC0** | 4000×3184 |
| pair03 Under the Wave off Kanagawa（Hokusai） | 对拍集 GT | Met Open Access **CC0** | 3859×2594 |
| hist01 Migrant Mother（Lange 1936） | 老照片上色/修复 | Wikimedia Commons **PD** | 1920×2496 |
| hist02 Abraham Lincoln（1863） | 老照片上色 | Wikimedia Commons **PD** | 1920×2474 |
| prod01 Potpourri jar / prod02 Glass cooler / prod03 Dish | 商品白底 | Met Open Access **CC0** | 559×625 / 600×457 / 600×530 |

退化-复原对拍集（`degrade.py` 从 GT 合成"老照片"输入，seed 42/43/44）：

| 对拍 | 退化算子（按序执行） | 输入尺寸 |
|---|---|---|
| pair01 | downscale 2.2 + blur 1.1 + noise 9 + scratch 5 + vignette 0.18 + fade 0.12 + jpeg 84 | 1732×1276 |
| pair02 | downscale 2.6 + gray + sepia 70 + blur 1.4 + noise 11 + scratch 6 + frame 2% + vignette 0.22 + jpeg 80 | 1538×1224 |
| pair03 | downscale 3.2 + blur 1.0 + noise 8 + scratch 4 + vignette 0.15 + fade 0.10 + jpeg 78 | 1205×810 |

> 素材抓取脚本 `inference/photo/tools/fetch_assets.py`（幂等，curl 重试）；来源/许可/sha256 记录于素材目录 `manifest.json`。

## 4. 运行方式

```bash
cd /mnt/data/ai_workspace/cland-llm
# ① 混合模式批量（推荐：cases.json 给出每张的 mode/strength/gt）
python3 inference/photo/retouch.py \
  --manifest /mnt/data/ai_workspace/outputs/photo-retouch/materials/cases.json \
  --output /mnt/data/ai_workspace/outputs/photo-retouch --run-name v0

# ② 输入目录批量（目录内全部图片同一 mode）
python3 inference/photo/retouch.py --input <DIR> --output <OUT> --mode repair --strength 0.25 --seed 42

# ③ 评估（PSNR/SSIM/无参指标/评分 + Owner 总览拼图）
python3 inference/photo/evaluate.py --run <OUT>/run.json --overview <OUT>/owner_overview.png

# ④ 工作流（alice-workflow-hub，manual 触发，两步 shell：retouch → evaluate）
#    workflows/photo-retouch.yaml → id: photo.retouch（已通过 workflow_spec 校验）
```

**workflow 命令示例**（hub 执行器就绪后）：
`uv run alice-workflow-hub run photo.retouch --manifest <cases.json> --output <OUT>`

## 5. 结果（v0 全量 run：11/11，失败率 0%）

总耗时 **654.4s（10.9 min）** + 模型加载（冷 ~6min / 热 56s）；分阶段均值：预处理 0.06s · ESRGAN 8.68s · 扩散 60.27s · 收尾 1.45s · **单张合计 59.49s**。

| 案例 | mode | 关键参数 | 尺寸（输入→超分→输出） | 耗时 | 说明 |
|---|---|---|---|---|---|
| pair01_harvesters_ai | repair | s=.25 seed=42 | 1732×1276 → 2048 → 1024×752 | 49.0s | 对拍1 AI 修复 |
| pair01_harvesters_esrgan | upscale | — | 1732×1276 → 2048 | 12.5s | 对拍1 基线（无扩散） |
| pair02_wheat_field_colorize | colorize | s=.80 g=7 | 1538×1224 → 2048 → 1024×816 | 128.8s | 对拍2 上色（sepia 输入） |
| pair03_great_wave_ai | repair | s=.25 seed=42 | 1205×810 → 2048 → 1024×688 | 41.6s | 对拍3 AI 修复（低保真小图） |
| pair03_great_wave_esrgan | upscale | — | 1205×810 → 2048 | 11.1s | 对拍3 基线 |
| hist01_migrant_mother_colorize | colorize | s=.80 g=7 | 1920×2496 → 2048 → 784×1024 | 132.4s | 老照片上色 |
| hist01_migrant_mother_repair | repair | s=.30 | 1920×2496 → 2048 → 784×1024 | 59.4s | 老照片黑白修复 |
| hist02_lincoln_colorize | colorize | s=.80 g=7 | 1920×2474 → 2048 → 784×1024 | 132.4s | 老照片上色 |
| prod01_potpourri_jar | product | s=.25 | 559×625 → 2048 → 1024×864 | 34.4s | 商品白底 |
| prod02_glass_cooler | product | s=.25 | 600×457 → 2048 → 1024×672 | 26.1s | 商品白底 |
| prod03_dish_death_of_saul | product | s=.25 | 600×530 → 2048 → 1024×688 | 26.6s | 商品白底 |

### 5.1 对拍集指标（vs GT，PSNR/SSIM；1024 口径 = 统一放大到长边 1024 后比较）

| 案例 | PSNR↑ | SSIM↑ | PSNR@1024 | SSIM@1024 | 色彩度 输入→输出(GT) | 解读 |
|---|---|---|---|---|---|---|
| pair01_ai (repair) | 16.38 | 0.396 | 16.38 | 0.396 | 46.4→57.8（GT 46+） | 扩散重绘：更干净/更艳，但像素偏离 GT |
| pair01_esrgan (基线) | **20.45** | **0.519** | 20.43 | 0.571 | 46.4→46.0 | 保真上限（无生成） |
| pair02_colorize | 12.01 | 0.069 | — | — | 18.7→**57.7**（GT 48.1） | 上色任务：色彩度接近 GT；逐像素 SSIM 不可作判据 |
| pair03_ai (repair) | 14.45 | 0.430 | — | — | 33.3→34.5 | 低保真小图，扩散重绘提升观感 |
| pair03_esrgan (基线) | **21.91** | **0.590** | 21.82 | **0.654** | 33.3→33.2 | 保真上限 |

> **结论（供选品参考）**：`upscale`（纯超分）保真最高、最快（11-13s）；`repair`（AI 修复）观感更"干净鲜亮"但会改动像素（SSIM 明显低于基线）。对"要像原图"的客户用 upscale，对"要好看"的客户用 repair；两个版本本次都随样例交付，Owner 可对比拍板。

### 5.2 无参指标 + 5 维评分（AI 代理评分，1–5，**待 Owner 目测校准**）

| 案例 | 清晰度(lapvar) | 噪声σ | 色彩度 | 修复自然度 | 细节 | 色彩 | 干净度 | 商用可用度 |
|---|---|---|---|---|---|---|---|---|
| pair01_harvesters_ai | 3160 | 0.0 | 57.8 | 1* | 5 | 4.8 | 5 | 3.6 |
| pair01_harvesters_esrgan | 2720 | 0.0 | 46.0 | 2* | 5 | 4.2 | 5 | 4.3 |
| pair02_wheat_field_colorize | 1182 | 0.0 | 57.7 | 1* | 5 | 4.2 | 5 | 3.4 |
| pair03_great_wave_ai | 3173 | 0.0 | 34.5 | 1* | 5 | 4.4 | 5 | 3.5 |
| pair03_great_wave_esrgan | 851 | 0.0 | 33.2 | 3* | 5 | 4.3 | 5 | 4.7 |
| hist01_migrant_mother_colorize | 345 | — | 19.0 | 5.0 | 5 | 3 | 5 | 4.6 |
| hist01_migrant_mother_repair | 321 | — | 0.0 | 4.0 | 5 | 1 | 5 | 3.9 |
| hist02_lincoln_colorize | 228 | — | 11.5 | 4.0 | 5 | 2 | 5 | 4.0 |
| prod01_potpourri_jar | 2051 | — | 17.3 | 2* | 4 | 3 | 5 | 3.3 |
| prod02_glass_cooler | 2011 | — | 22.3 | 2* | 5 | 4 | 5 | 3.8 |
| prod03_dish_death_of_saul | 2033 | — | 33.8 | 2* | 5 | 4 | 5 | 3.8 |

> `*` 评分说明：**修复自然度**按 SSIM 分档（对拍集）——`repair` 因主动重绘天然低分、`upscale` 高分，这是"保真 vs 观感"的取舍而非质量缺陷；商品图输入本身干净（无噪声可去、无 GT），该维度不适用（2 分为指标口径产物）。**商用可用度** = 0.35×修复+0.25×细节+0.2×干净+0.2×色彩（长边≥1600 加 0.5）。
> 评分为**代理评分**（本地无视觉大模型，评分脚本 = `evaluate.py`），最终以 Owner 目测为准；本报告已附全部对比图。

### 5.3 上色色偏修复（hist01_colorize · #179 收尾）

**问题**（Owner 2026-10-02 判定）：`hist01_migrant_mother_colorize` 人物黑白照上色后，**脸/手臂有色偏伪影**（手部被误上成蓝灰、眼周/颈部泛青）。根因：SDXL 上色的色度（a/b）被 `chroma_transfer` 原样采纳，模型把靠近蓝色毛衣的手部也预测成蓝色，且肤色饱和度整体偏低。

**修复**（`retouch.py`：`build_skin_mask()` + `fix_skin_color_cast()`，colorize 后处理，逐案例开关 `skin_fix`）：
1. **面部/手/前臂掩膜**：DWPose（已有的 `inference/sdxl/hand_pipe/dwpose.py`，CPU onnxruntime，不占 GPU）给出 68 点人脸 + 双手 + 肘/腕关键点，取凸包并生成：脸 + 双手 + 前臂（裸皮段）+ 颈/前胸条带；**眼部挖洞**避免眼白/虹膜被染色；膨胀+高斯羽化避免接缝。
2. **局部色偏抑制**：在 LAB 色度平面内，把掩膜内**色相异常**（落在天然肤色窗口 22°–72° 之外，如蓝色手部 -90°）的像素拉回暖肤色色相（43°），并把饱和度夹取到 5–10；色相正常的像素保持（保留脸颊自然红润）。亮度/纹理完全不动。
3. **验收证据图**：开关开启时额外输出 `<case>/skin_zoom.png`（修复前/后**面部放大**并排）。

**效果（hist01_colorize，修复前 → 后）**：

| 指标 | 修复前 | 修复后 | 判据 |
|---|---|---|---|
| 整体色彩度（colorfulness） | 19.5 | **19.0**（−2.6%） | 变化 ≤ ±20% ✅ |
| 面部/手部蓝色色偏 | 手部 b≈−14（蓝）、眼周青 | 手/脸暖肤色，无色偏块 | 肤色自然无色偏 ✅ |
| 肤色色相异常像素占比 | — | 54.4% 掩膜像素被拉回（`skin_cast_fixed_pct`） | — |
| 5 维评分（代理） | 修复自然度 5.0 / 色彩 3 / 商用 4.6 | **不变** | 无回退 ✅ |
| 原图 sha256（after） | a9fd8d0f… | **025eb220…** | 仅 hist01 变 |

**边界（已遵守）**：本修复**仅对 `hist01_migrant_mother_colorize` 开启**（`cases.json` 逐案例 `skin_fix:true`），老照片主线其余 pass 例（`pair02/pair03/hist01_repair/hist02`）与商品线（`prod01-03`）输出**逐字节不变**（sha256 复验一致）。

**已知限制**：掩膜依赖 DWPose 检出人物；色相/饱和度目标为**浅肤色**调参，对深肤色人物可能偏保守（后续可按面部参考自适应）。

### 5.4 pair01 用途说明（#179 收尾口径）

`pair01/02/03` 是**内部评测对拍集**（对公有领域图人工退化，**有 GT 可算 PSNR/SSIM**），**不是交付样例**。因此 `owner_overview.png` 已改版：正文只放**交付样例**（老照片 `hist` / 电商 `prod`，共 6 例），对拍集单列 **「评测附录（对拍集 pair · 内部回归基线，有 GT 算 PSNR/SSIM，非交付样例）」（5 例），避免再次"不知道做什么用的"。

## 6. 关键实验记录（为什么是这套参数）

| 实验 | 配置 | 结果 → 决策 |
|---|---|---|
| repair 强度×步数（pair03 对拍） | s.25/30步 · s.40/30 · s.55/30 | PSNR 14.7→14.0→13.5，均在重绘；**改小步长** |
| repair 细步长 | s.15/60 · s.25/80 · s.30/80 | s.25/80 保真与清理平衡 → 定为 repair 默认 |
| colorize 提示词 | 通用"colorized historical photograph" s.42→.75 | 输出仍近灰度（色彩度 1-11）→ 改**场景/油画描述提示词** + **负面词加黑白/单色** + **强度 0.8** |
| colorize 结构保护 | 直接输出 vs **LAB 色度迁移**（L 用原图） | 色彩度 2.3→19.5（hist01）、57.7（油画对拍）；结构/面部零漂移 |
| ESRGAN 内存保护 | 直推 1920×2496 → x4=7680×9984 输出数组 ~1.2GB | 输入封顶 1280 → 输出 ≤5120 → 分块+截断 2048 → 稳定 |

## 7. 失败率与稳定性

- **v0 全量 run：11/11 成功，失败率 0%**（门禁 ≤5%）。
- 单案例异常（读图/超分/扩散/落盘）被捕获记为 `status=failed`，批量继续；`run.json.summary.fail_rate > 5%` 时脚本退出码非 0（workflow 可感知）。
- 复现性：同 seed + 同输入 = 同输出（目录模式回归 run 与清单 run 的 pair 案例逐字节一致，见 §8）。
- 服务依赖：**无**（SDXL fp16 进程内加载，冷启 ~6min / 热启 56-90s；不依赖 10331）。

## 8. 验收复跑记录

**① 输入目录一条命令（≥5 张，acceptance 口径）**：

```bash
python3 inference/photo/retouch.py \
  --input /mnt/data/ai_workspace/outputs/photo-retouch/materials/smoke_input \
  --output /mnt/data/ai_workspace/outputs/photo-retouch-dir \
  --mode repair --strength 0.25 --seed 42 --run-name dir-smoke
```

结果：**5/5 成功，失败率 0%**，总耗时 **243.4s**（+ 模型热启）；输出 `outputs/photo-retouch-dir/{5 案例 + run.json}`。
**确定性复验**：与清单 run 同参数案例的输出 sha256 逐字节一致——
`pair01: 624d93ac764b7a4f…` · `pair03: b1307fd8937c250d…`（两 run 各取一份对比）。

## 9. 已知限制（v0）

1. **扩散重绘 ≠ 像素复原**：repair/colorize 会改动像素（SSIM 低于纯超分基线）；两条路线并行交付，按客户取向选择。
2. **上色色准依赖提示词**：本地无专用上色模型；通用提示词会"保灰"，场景/油画描述词效果显著更好；后续可接入 ControlNet 或专用着色模型提升色准。
3. **商品白底为规则法**：连通域刷白 + 白点归一，未做抠图/阴影重打光；纯白产品高光存在被刷白风险（本轮 3 案例未触发）。
4. **对拍集 GT 为绘画**：PD 真实老照片没有彩色 GT，故合成对拍集用 Met CC0 高清画作（细节/色彩丰富，适合度量）；真实老照片只做定性样例。
5. **无视觉大模型评分**：5 维评分为脚本代理分，Owner 目测后校准。

## 10. 下一步（待 Owner 反馈）

- [ ] Owner 看 `owner_overview.png` → 选定 repair 风格取向（保真 upscale / 观感 repair）与上色色准偏好
- [ ] 参数微调落默认档（如商品图强度、上色饱和度）
- [ ] 常驻服务化（10331 类似的 img2img 服务，省去每次 6min 冷加载；batch 吞吐）
- [ ] 对拍集扩充 + 引入更强的上色/修复专用模型（如 FLUX/专用着色）
- [ ] 渠道上架相关（**不在本单范围**，Owner 安排）

## 11. 证据路径

| 产物 | 路径 |
|---|---|
| 代码 | `inference/photo/{retouch.py, upscale.py, degrade.py, evaluate.py, tools/fetch_assets.py}` |
| 输出（11 案例 + 报告数据） | `/mnt/data/ai_workspace/outputs/photo-retouch/{<case>/..., run.json, metrics.json, owner_overview.png}` |
| 上色色偏修复证据（面部放大） | `/mnt/data/ai_workspace/outputs/photo-retouch/hist01_migrant_mother_colorize/skin_zoom.png` |
| 素材与许可 | `/mnt/data/ai_workspace/outputs/photo-retouch/materials/{manifest.json, pairs/, historical/, products/, cases.json}` |
| 目录验收复跑 | `/mnt/data/ai_workspace/outputs/photo-retouch-dir/` |
| workflow | `alice-workflow-hub/workflows/photo-retouch.yaml`（id `photo.retouch`，已校验） |
| 运行日志 | `/tmp/retouch_v0f.log`、`/tmp/retouch_dir.log` |
