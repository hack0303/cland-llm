# Case 002：《最后一颗火种》小灯灵 Lumo（全流水线：角色 → 分镜 → 8 镜头 → 配音合成）

> 日期：2026-08-15 · 管线：prompt-hub 提示词 → char_sheet 角色 → 手动分镜 → run_story 全流水线
> 目的：验证完整 IP 生产链路（角色一致性锚定 + 分镜驱动 + 配音合成）
> 状态：✅ 已完成（8 镜头 + 配音 + 合成，17.6s 全片）

## 一、角色（Lumo v4）

| 项 | 值 |
|---|---|
| desc | `a tiny round glowing firefly fairy with a warm yellow cloak covering her feet and a small glowing wick on her chest, big bright eyes, holding a large lantern bigger than herself` |
| style | `cute healing fantasy` |
| lighting | `soft warm glow in darkness` |
| 资产 | front（锚定）+ walking/holding/waving + happy/sad/surprised（全透明版） |
| 提示词源 | prompt-hub（唯一真源） |

**提示词演进记录**（本轮踩坑）：
- v1（firefly fairy）：脸最好，但脚丑（光脚/鞋混出）
- v2（palm-sized/chubby/light fairy）：脸变丑（chubby 脸型变形 + 负面词 ugly 反向效应）
- v3（同上 + 增强负面词）：最丑
- **v4（v1 脸 + 长斗篷遮脚 + 简单负面词）**：✅ 定稿

## 二、分镜（8 镜头 × 2s = 16s）

故事《最后一颗火种》：Lumo 发现火种 → 保护前行 → 雪怪 → 点亮雪怪 → 机器人 → 小女孩 → 揭示真相 → 光明扩散

| 镜头 | 场景 | 配音 |
|---|---|---|
| 1 | 废墟发现火种 | 旁白：世界曾拥有太阳… |
| 2 | 保护火种前行 | 旁白：小灯灵一族世代收集遗落的光 |
| 3 | 遇雪怪 | 别害怕 |
| 4 | 点亮雪怪 | 好…好暖和 |
| 5 | 废弃机器人 | 滴——检测到…光 |
| 6 | 小女孩 | 你是谁？ |
| 7 | 递灯揭示 | 是你心里的希望 |
| 8 | 光明扩散 | 当希望被点亮… |

## 三、执行参数与实测

| 环节 | 参数 | 实测 |
|---|---|---|
| SDXL 出图 | 30 步，negative 简单版（去 ugly） | ~46s/张 |
| I2V | 512×512（甜点）、16 帧、20 步、denoise 0.8 | **~140s/镜头**（1024 会 15min/镜头——run_story 已内置 512 缩放） |
| 角色锚定 | IPAdapter weight 0.7 + front.png | 跨镜头锁角色 |
| TTS | Spark-TTS，中文 | 秒级/句 |
| SFX | **跳过**（15GB RAM 限制，能力边界决策） | — |
| 合成 | concat + adelay 对齐 | <30s |

**总耗时**：8 镜头 ≈ 30 分钟（角色生成 9 分钟另计）

## 四、关键修复（本轮）

| 问题 | 修复 |
|---|---|
| SDXL 并发 500（scheduler sigmas 越界） | server.py 全局锁串行化 |
| 负面词 ugly/deformed 反向降质 | 回归简单负面词 |
| I2V 1024 跑 15min/镜头 | generate.py --size 512 缩放（run_story 默认） |
| run_story 误判 ComfyUI 未运行 | healthy() 兼容 /system_stats |
| bash 工具后台启动参数丢失 | 启动脚本 + cmdline 验证 |
| pkill -f 自杀（第二次踩） | [x] 字符类技巧 |

## 五、产物

```
cland-llm/outputs_video/lumo/
├── storyboard.json        # 分镜（v4 前缀）
├── frames/scene001~008.png
├── clips/scene001~008.mp4 # 512×512 @8fps
├── audio/scene00X_voice.wav
└── lumo_final.mp4         # 16s 合成片（配音 + 无音效）

outputs_character/lumo/    # 角色资产（16 文件）
```

## 六、结论

1. **角色提示词是迭代出来的**：v1→v4 四轮，负面词和体型词是脸质量的两大变量
2. **能力边界裁剪有效**：SFX 跳过、三视图只 FRONT、动作 3 个——没影响主线出片
3. **512 甜点纪律**：一切 I2V 输入必须 512（1024 代价 6.6 倍）
4. **锚定链路完整**：prompt-hub 提示词 → char_sheet 资产 → front.png 锚定 → 8 镜头角色一致
