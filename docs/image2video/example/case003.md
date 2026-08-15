# Case 003：V2 多管线架构验证（Lumo《最后一颗火种》）

> 日期：2026-08-16 · 架构：V2 工业级多管线分离（bg/char/anim/audio/merge）
> 目的：验证 V2 架构能力（多管线/幂等/编排/端到端），非画质验收
> 结论：**架构能力全部验证通过**；画质限制（白板脸等）为模型/分辨率已知边界，非架构缺陷

## 一、验证对象（V2 架构）

```
inference/i2v2/
├── pipeline.py    # 编排（--only/--scene 部分执行，幂等）
├── pipe_bg.py     # 背景管线：纯场景出图（无角色，bg_prompt 字段）
├── pipe_char.py   # 角色管线：白底出图（脸部锚定）→ RMBG 透明
├── pipe_anim.py   # 动画管线：16 帧 I2V + Motion LoRA → 透明序列
├── pipe_audio.py  # 音频管线：预配音（克隆锚定尽力而为）
└── merge.py       # 合成管线：overlay 角色→背景 + compose 复用
```

## 二、架构能力验证矩阵

| 能力 | 验证方式 | 结果 |
|---|---|---|
| 多管线分离 | 背景/角色/动画/音频/合成独立运行 | ✅ |
| 单目标可调 | 背景 456 修复仅重出 bg（不动其他管线） | ✅ |
| 幂等断点 | 每管线产物文件检查，重跑跳过已完成 | ✅ |
| 编排控制 | `--only anim,merge` / `--scene 1` 部分执行 | ✅ |
| 端到端出片 | 分镜 → 5 管线 → 25s 成片（画面+配音） | ✅ |
| 音频链路 | 16kHz / PTS 无爆炸 / loudnorm 响度一致 | ✅ |
| 运动增强 | Motion LoRA（ZoomIn 0.6）幅度 3.3→32.4 | ✅ |
| 角色一致性 | face_anchor 脸部锚定链路 | ✅ |
| 速度 | 16 帧 + Motion LoRA = 135s/镜（vs 32 帧 500s+） | ✅ |

## 三、管线产物

```
/mnt/data/ai_workspace/outputs_video/lumo/assets/
├── bg/           8 张纯场景背景（456 纯场景化修复版）
├── char_white/   8 张白底角色（脸部锚定）
├── char_alpha/   8 张透明角色（RMBG）
├── anim/         128 帧透明序列（16×8）
├── anim_white/   8 个白底动画 mp4
├── audio/        8 句配音 wav
└── merged/       8 个合成镜头片段
lumo_final.mp4    25.0s 成片
```

## 四、已知质量限制（非架构缺陷）

| 限制 | 根因 | 路线 |
|---|---|---|
| 白板脸（脸细节弱） | 512² 全身构图 → 脸 ~50px + AnimateDiff 时序平滑 + 动漫扁平风 | C 分镜特写 + B FaceDetailer 补脸；D（SDXL 系）性价比最低（1024 视频 P40 跑不动） |
| 背景细节 | 512 场景出图 | 背景可独立升级（768/更高）——管线可替换性正是为此 |
| 次角色（雪怪/机器人/小女孩） | 未纳入角色管线（bg_prompt 已纯场景化） | 独立次角色管线或镜头留白 |
| 声纹一致性 | Spark-TTS 克隆效果有限（0.82 vs 0.83） | 保留克隆逻辑，多角色区分后续 GPT-SoVITS |

## 五、修复链记录（V1 遗留 → V2）

| 问题 | 修复 |
|---|---|
| 音频 PTS 爆炸（无声） | loudnorm 后 asetpts + amix 后 asetpts + aresample 16000 |
| 32 帧慢 4 倍 | 16 帧固定 + tpad 补时长 |
| 猫脸 | face_anchor 脸部锚定（出图+视频统一） |
| 无动作 | Motion LoRA（ZoomIn） |
| 动画提示词残留猫 | workflow 节点 5 动态写 motion_prompt（模板残留坑，见 PITFAILLOG） |
| 背景混入角色 | bg_prompt 纯场景化 |

## 六、结论

1. **V2 架构成立**：多管线分离解决了 V1 的"单管线多目标互相妥协"结构性问题
2. **画质是可调参数，架构是可验证事实**——当前可进入"按需打磨"阶段
3. 下一步优先级建议：分镜特写化（C）> FaceDetailer（B）> 背景 768 升级 > 次角色管线
