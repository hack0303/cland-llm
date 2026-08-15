# TODO-image2video: 图生视频实施看板

> 遵循 GFM 任务列表规范（todoprompt skill）
> 依据：`docs/image2video/DESIGN_V2.md`（V2 工业架构：多管线分离）· `docs/image2video/RESEARCH.md`（选型）
> 格式：`- [ ]` 待办 | `- [x]` 已完成 | `- [/]` 执行中 | `- [-]` 取消 | `- [!]` 阻塞

---

# 目标：P40 机器图生视频管线（V1 跑通 → V2 工业架构 → 质量打磨）

## 一、V1 基线（单管线，已验证）

- [x] ComfyUI headless（GPU1:10337）+ AnimateDiff-Evolved + IPAdapter_plus 节点
- [x] 模型下载：SD1.5 + mm_sd_v15_v2 + IP-Adapter + CLIP-ViT-H（4.4GB）
- [x] I2V 工作流 `workflow_i2v.json` + `generate.py`（首条出片 512px 140s）
- [x] V1 修复链：音频 PTS 爆炸 / 采样率 / 16 帧提速（500s→140s）/ face_anchor 猫脸
- [x] Motion LoRA（ZoomIn 0.6）运动增强（幅度 3.3→32.4）

## 二、V2 工业架构（多管线分离，已验证 case003）

- [x] `inference/i2v2/` 五管线：pipe_bg / pipe_char / pipe_anim / pipe_audio / merge
- [x] `pipeline.py` 编排（--only/--scene 幂等断点）
- [x] 背景纯场景化（bg_prompt 字段，456 无人物）
- [x] 白底角色 + 脸部锚定 + RMBG 透明序列 + overlay 合成
- [x] 动画提示词动态化（motion_prompt 写入节点 5，修猫残留）
- [x] case003 架构验证记录 + PITFAILLOG ×6
- [x] gitignore 生成物清理

## 三、质量打磨（按 ISSUE.md 优先级）

- [ ] 分镜特写化（C 方案：多中景/近景，脸占画面大）[priority:: high] [ref:: ISSUE.md #1]
- [ ] FaceDetailer 补脸（B 方案：首帧脸区域高清重绘，Impact Pack）[priority:: high] [ref:: ISSUE.md #1]
- [ ] 次角色管线（雪怪/机器人/小女孩独立处理）[priority:: medium] [ref:: ISSUE.md #3]
- [ ] 背景分辨率升级（512→768，管线可替换性验证）[priority:: medium] [ref:: ISSUE.md #2]
- [ ] RIFE 帧插值 16→48 帧 @24fps [priority:: low]

## 四、LoRA 训练（角色固化）

- [ ] 数据生成 `prepare_data.py`（IPAdapter 锚定批量，LORA_DATA.md）[priority:: medium]
- [ ] 数据筛选（自动预筛 + 人工终审，≥18 张）[priority:: medium]
- [ ] 训练 lumo_v1（diffusers/kohya，30-60min）[priority:: medium] [ref:: LoRA.md]
- [ ] 验证（触发词/姿态泛化/视频兼容）+ 接入 char_sheet [priority:: medium]

## 五、服务化与集成

- [ ] 薄 FastAPI 网关（POST /generate：分镜→成片）[priority:: low]
- [ ] QUICK_START / CHANGELOG 更新 [priority:: low]
- [ ] （可选）LTX-Video 2B 写实向补全（~14GB）[priority:: low]

---

## 状态汇总

| 状态 | 计数 | 说明 |
|------|------|------|
| `- [x]` 已完成 | 14 | V1 基线 + V2 架构全部落地 |
| `- [/]` 执行中 | 0 | — |
| `- [ ]` 待执行 | 11 | 打磨/LoRA/服务化 |
| `- [!]` 失败/阻塞 | 0 | — |
| **总计** | **25** | — |

> 最后更新：2026-08-16
