---
title: "老照片上色 v0.2：成品分辨率修复 + 色度清理（2026-10-18 实单驱动）"
summary: "两张实单老照片暴露 colorize 两个缺陷：①成品落盘在 work 分辨率（1024），ESRGAN 的 2048 细节被丢弃；②DDColor 在明亮低纹理区猜出紫/品红色块、细纹理上出彩虹条带。v0.2 用 --color-base up（色度上采样铺到超分亮度）与 chroma_transfer 的色度中值 + 亮区抑制修复；雪景天空色度 8.4→0.0、人物肤色逐像素不变，2 张实单耗时 28.6s。"
read_when:
  - "运行/维护 colorize 上色链路（--colorizer ddcolor / --color-base / --chroma-smooth / --flat-damp）"
  - "成品分辨率不对（上色结果比 upscale 小）或上色出现紫/品红色块、彩虹条带"
  - "评估 v0.2 与 v0.1 上色差异，或回滚口径"
tags: [photo, colorize, ddcolor, chroma, regression, 老照片上色]
owner: sage
status: "v0.2 · 2026-10-18（实单验证，待 Owner 目测）"
issue: base/cland-ai#2
---

# 老照片上色 v0.2（2026-10-18）

## 1. 触发：实单暴露的两个缺陷

工单 [base/cland-ai#2](http://192.168.1.7:3001/base/cland-ai/issues/2)（Owner 上传 2 张老照片：2096×3209 雪景人像、3513×2397 室内合影），
用 v0 标准管线 `retouch.py --mode colorize --colorizer ddcolor --max-side 2048 --seed 42` 跑，结果**在判据上通过、在观感上有两处硬伤**：

| # | 现象 | 根因 | 量化 |
|---|---|---|---|
| D1 | 成品 `after.png` 只有 **1024** 长边（输入 3513×2397 → 输出 1024×688） | colorize 收尾把色度铺到 `_work`（SDXL 工作分辨率 1024），ESRGAN 出来的 `_up`（2048）**只用于上色源、没进成品** | 见 §3 表：v0.1 成品 1024 vs v0.2 成品 2048 |
| D2 | 雪景**天空被猜成紫/品红+黄绿射线**；室内合影**白墙/白衣出现彩虹条带** | ① DDColor 在明亮低纹理区（天空/雪/白布）无依据地"补色"；② 色度在 512 口径下带噪，直接 LANCZOS 上采样到 2048 → 细纹理上呈彩色摩尔纹 | 雪景天空平均色度 **8.4 → 0.0**（LAB C*），白墙 12.9 → 9.6，人脸/肤色 **不变**（差 <0.3） |

> **口径纪律**：D1/D2 都在**标准管线内部**修复（新参数 + 文档 + CHANGELOG），没有另写临时脚本；实单命令与日志可核。

## 2. 改法（`inference/photo/retouch.py`）

| 参数 | 默认 | 作用 |
|---|---|---|
| `--color-base {up,work}` | `up`（新默认） | colorize 成品亮度基底：`up` = ESRGAN 超分图（保留 `--max-side` 细节），`work` = v0.1 旧口径（长边 = `--work-res`）。色度是低频信息，从 work 分辨率上采样足够，**亮度必须用 up** 才能不丢细节 |
| `--chroma-smooth N` | 3 | 在**色度原生分辨率**上做 N×N 中值滤波，再上采样 —— 压掉 512 口径色度噪点/彩虹条带（0 = 关） |
| `--chroma-strength S` | 1.0 | 色度强度：整体压/放模型上色的饱和度（`0.7~0.85` 可减轻过度饱和与臆造色块，色相不变） |
| `--flat-damp F` | 1.0 | 明亮（L>215 全抑制，L<170 不抑制）且低纹理（局部 L 标准差 <4 全抑制，>12 不抑制）区域把色度向中性回拉 F 比例；暗部/高纹理区不受影响 |

关键实现点（`chroma_transfer()`）：

```
a,b = LAB(color)[a,b]         # DDColor 出色层（512/1024 口径）
a,b = median(a,b, N)          # 色度去噪（在原生分辨率做，别放大后再滤）
a,b = resize(a,b → base)      # 上采样到 ESRGAN 亮度基底
w   = 1 - flat_damp * smooth_bright_mask(L_base)   # 亮区保留系数
out = LAB(L_base, 128+(a-128)*w, 128+(b-128)*w)
```

- 掩膜**只看亮度通道**（L 的局部标准差 + 亮度），不依赖语义分割/人脸检测 → 无额外模型、无显存成本（post 阶段 ~4s/张）；
- 掩膜的"平滑/明亮"判定**必须是 0..255 口径**（`cv2.cvtColor` 对 uint8 输入 L∈[0,255]，对 float32 输入 L∈[0,100]）——本轮第一版把阈值写成 0..255 却喂了 float32，掩膜恒为 0（等于没生效），见 PITFAILLOG。

## 3. 回归证据（同 2 张实单，跑 4 版）

| 版本 | 命令要点 | 成品尺寸（雪景/合影） | 雪景天空 C\* | 白墙 C\* | 耗时 |
|---|---|---|---|---|---|
| v0.1 旧口径 | 默认参数 | 656×1024 / 1024×688 | 8.4 | 12.9 | 20.0s / 13.5s |
| v0.2-a 中间态 | `--color-base up` | 1337×2048 / 2048×1396 | 8.3（掩膜写反，无效） | 13.0 | 15.9s / 14.1s |
| **v0.2 最终** | `+ --chroma-smooth 3 --flat-damp 1` | 1337×2048 / 2048×1396 | **0.0** | **9.6** | 15.2s / 14.1s |

- 人脸/肤色回归（母亲/孩子/父亲 ROI 平均 LAB）：v0.2 与 v0.2-a 差 ≤0.3（`L`/`a`/`b` 逐项），**亮区抑制不碰暗部与皮肤**；
- 对比拼图（内部回归基线，非交付物料）：`/mnt/data/ai_workspace/outputs/photo-retouch/colorize-v02-20261018/evidence.png`；
- 实单产物：`/mnt/data/upload/ai_picture_fix/out_20261018/<case>/{before,after,compare}.png + meta.json`、`run.json`；
- 运行日志：`/mnt/data/upload/ai_picture_fix/out_20261018.log`。

## 4. 第三张实单（纯黑白儿童照，2026-10-18 追加）

同命令再跑 1 张 1212×2017 纯黑白照片：成品 `after.png` 1230×2048（12.5s）。
- 量化：输入通道差 R-G 0.2 / R-B 0.3（确认纯黑白），上色后皮肤/衣物/木雕屏风均有色，无紫红天空类伪影（本图无大面积天空）；
- 观感：**颜色鲜艳但属"推测色"** —— 裤子被猜成粉紫条纹、白外套上出现淡粉/橙色斑点、屏风彩绘偏青绿饱和；人脸偏白偏平（原片大光比 + 顶灯光斑）；
- `--chroma-strength 0.75/0.6` 与 `--chroma-smooth 9` 复跑对比：饱和度小幅下降，**裤子色相与外套斑点不变** ⇒ 属 DDColor 的语义猜测误差，调后处理参数解决不了。

## 5. 剩余局限（如实说明，未解决）

1. **颜色是"猜"的**：DDColor 只依据亮度推理色度，**没有真实色彩 GT**。合影里沙发的红、右侧布帘的青色都是模型猜的，画面整体偏暖/偏红；判据只能到"无明显串色/伪影"，不能保证"还原当年真实颜色"。
2. **细纹理彩虹条带只减轻未根除**：`--chroma-smooth 7` 与 3 观感接近，说明残留来自 DDColor 在 512 口径的结构性缺失，不是噪声；彻底解决要上更高输入尺寸的上色模型或对色度做引导滤波（`inference/photo/tools/cmp_colorize_offset.py` 里 B/C 路线已验证过，成本/收益未定）。
3. **白衣服/浅衣物会被"染色"**：DDColor 在白色衣物褶皱处会点状补色（2026-10-18 第三张实单：白外套出现淡粉/橙斑）；`--flat-damp` 只按亮度+纹理判定，褶皱区纹理高 → 未被抑制。根治需衣物/材质语义掩膜。
4. **亮区抑制是启发式**：若照片里真有"大面积、明亮、低纹理、有颜色"的主体（白色彩色墙面、雪地红衣等），会被拉灰；`--flat-damp 0` 可关，但天空/白布伪影会回来。
5. **成品上限 = `--max-side`**：`--max-side 2048` 时 3513×2397 的合影仍是 2048×1396（0.58×），不是原分辨率。真要原尺寸需 `--max-side` 调高 + 显存/耗时上升（P40 24G，ESRGAN 分块，未实测上限）。

## 6. 复跑与回滚

```bash
cd /mnt/data/ai_workspace/cland-llm
# v0.2（现默认）
python3 inference/photo/retouch.py --input <DIR|FILE> --output <OUT> \
    --mode colorize --colorizer ddcolor --max-side 2048 --seed 42
# 回滚 v0.1 口径（成品退回 work 分辨率、关颜色清理）
python3 inference/photo/retouch.py ... --color-base work --chroma-smooth 0 --flat-damp 0
# 色彩更收敛（少臆造色块，色相不变）
python3 inference/photo/retouch.py ... --chroma-strength 0.75
```
