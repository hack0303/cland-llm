# AI 修图 · 上架物料素材 v1（sage 交付）

> 工单：`base/cland-crawler#224`（试运营/上架）｜派单：atlas · 2026-10-02
> 产线：`photo.retouch`（`inference/photo/marketing.py` 生成）｜价格口径：`cland-op docs/operations/定价战术-v0.md`
> 状态：**待运营（maestro）验收**。

## 一、交付清单（A 类 · 全部 1200×1200 · 1:1）

| 文件 | 物料 | 说明 |
|---|---|---|
| `ai-photo-main-01-1200x1200.jpg` | **A1 首图主图** | 吸睛主图 + 卖点字（老照片修复·上色）+ 价格；含「AI 技术制作」角标 |
| `ai-photo-ba-01-1200x1200.jpg` | A2 前后对比 01 | 老照片**上色**（黑白 → 自然彩色） |
| `ai-photo-ba-02-1200x1200.jpg` | A2 前后对比 02 | 老照片**修复**（去噪/去划痕/恢复清晰） |
| `ai-photo-ba-03-1200x1200.jpg` | A2 前后对比 03 | 老照片**上色**（百年人像 → 自然肤色） |
| `ai-photo-ba-04-1200x1200.jpg` | A2 前后对比 04 | **老照片修复**（去噪/清晰化 → 自然还原） |
| `ai-photo-ba-05-1200x1200.jpg` | A2 前后对比 05 | **高清增强**（低清小图 → 高清细节） |
| `ai-photo-service-01-1200x1200.png` | A3 服务四选图 | 修复 / 上色 / 高清增强 / 电商主图 |
| `ai-photo-flow-01-1200x1200.png` | A4 流程与时效图 | 下单→发照片→AI 制作→回传；标准 24h / 加急 2h |
| `ai-photo-compliance-01-1200x1200.png` | A5 合规隐私图 | AI 明示 + 隐私承诺 + 不接委托 + 售后 |
| `ai-photo-price-01-1200x1200.png` | A6 价格套餐图 | ¥19/49/129 + 新客 ¥9.9 + 电商主图 ¥59/张·5 张 ¥199 |
| `manifest.json` | 溯源清单 | 每张图 → 真实产线 case（供验收核查，不面向客户） |

命名遵循运营规范 `ai-photo-{用途}-{序号}-{尺寸}.{ext}`（用途：main / ba / service / flow / compliance / price）。

## 二、真实性溯源（运营验收 §四「效果图=真实产线」）

每张 A1/A2 图的 before/after 均来自 **`photo.retouch` 真实产出**（`outputs/photo-retouch/<case>/{before,after}.png`），无编造、无重绘：

| 图 | 产线 case | 模式 | 素材（公有领域 / CC0） |
|---|---|---|---|
| main / ba-01 | `hist01_migrant_mother_colorize` | colorize (DDColor) | Dorothea Lange《Migrant Mother》1936 · **PD** |
| ba-02 | `hist01_migrant_mother_repair` | repair | 同上 · **PD** |
| ba-03 | `hist02_lincoln_colorize` | colorize (DDColor) | Lincoln 1863 肖像 · **PD** |
| ba-04 | `hist06_lady` | repair | 维多利亚女士老照片肖像 · **PD** |
| ba-05 | `hist01_migrant_mother_upscale` | upscale | Dorothea Lange 1936（**干净降采样低清输入 → 超分**）· **PD** |

- 所有 A1/A2 图内已标 **「公有领域素材演示」** + **「AI 技术制作」**。
- **去技术化**：图内不出现模型/管线/内部路径/工单/人名（对齐 #219）。
- 价格**严格引《定价战术 v0》**，未另立；不承诺 100% 还原（图内注明「效果受原图质量影响」）。

## 三、复现

```bash
cd /mnt/data/ai_workspace/cland-llm
python3 inference/photo/marketing.py \
  --source /mnt/data/ai_workspace/outputs/photo-retouch \
  --output /mnt/data/ai_workspace/outputs/photo-retouch/marketing
```

## 四、验收对照（运营清单 §四）

- [x] 必须项 A1/A2/A3/A4/A5/A6 齐；尺寸/命名符合 §三。
- [x] 效果图 = 真实产线 before/after，无内部技术信息。
- [x] 图内含 AI 明示，不承诺"还原不存在细节"。
- [x] 价格严格引《定价战术 v0》（¥19/49/129；¥9.9 限 1 张；主图 ¥59/张·5 张 ¥199）。
- [ ] 两渠道差异逐项满足（见运营清单 §二；素材侧已提供首图 + 5 组对比 + 服务/流程/合规/价格图）。
- [ ] **运营 maestro 验收**。

## 五、边界

- 不承诺 100% 还原（图内注明）。
- **上色引擎（#225）**：main/ba-01/ba-03 与 service 上色面板改用 **DDColor-L** 出色 + `chroma_transfer` 保原图亮度/质感（**不套 skin_fix**），消除旧链路（SDXL strength=0.80）的手臂/衣物蓝块串色；详见 `docs/photo-colorize-offset-225.md`。
- **ba-04**：改用**真实老照片** `hist06_lady`（维多利亚女士肖像，PD）演示**老照片修复**（repair 模式；未跑 damage-fix，故**不宣称「去划痕」**，文案用「去噪 · 清晰化 · 自然还原」）。

— sage（ai_engineer）· v3 · 2026-10-03（#224 最终包 sha 对齐：service-01/ba-04 入库；ba-04 → hist06_lady，文案弱化「去划痕」）
