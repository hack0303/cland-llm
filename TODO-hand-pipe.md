# TODO-hand-pipe: AI 生图手脚修复管线实施看板

> 遵循 GFM 任务列表规范（todoprompt skill）
> 依据：ShowDoc「AI生图手脚崩坏原因与解决方案」（LLM/251514559）· `docs/sdxl/hand_pipe.md`（技术手册）
> 格式：`- [ ]` 待办 | `- [x]` 已完成 | `- [/]` 执行中 | `- [-]` 取消 | `- [!]` 阻塞

---

# 目标：SDXL 手脚崩坏工业修复管线（组件测试 → 管线搭建 → 端到端验证 → 质量打磨）

## 一、组件环节测试（已完成）

- [x] 基线测试：4 场景 × 提示词方案对比 8 图（DWPose 手部关键点量化评估）[verify:: outputs/hand_pipe/results.json]
- [x] 模型资产下载：ControlNet 三件套（openpose/depth/canny 11.8GB）+ SDXL-inpaint（20GB）+ DWPose（350MB）+ ESRGAN（134MB）
- [x] DWPose 检测器（yolox 标准 grid+stride decode + RTMPose SimCC + 仿射预处理）[verify:: python3 hand_pipe/dwpose.py <img> <out>]
- [x] 超分组件（RRDBNet 纯 torch 实现，1024→4096 双模型验证）[verify:: python3 hand_pipe/rrdbnet.py <pth> <img> <out>]
- [x] ControlNet OpenPose 生成（373s，条件图为 DWPose 骨架含手部 21 点）
- [x] Inpaint 局部重绘（149.7s，自动检测手部 bbox + 羽化 mask）

## 二、管线服务搭建（已完成）

- [x] `pipe_server.py` 六接口服务（GPU1:10335：generate/pose/cn_generate/inpaint/upscale/pipeline）
- [x] 8bit 量化配置（PipelineQuantizationConfig，base+inpaint 8bit / CN fp16）
- [x] 显存优化：只常驻 openpose CN + empty_cache + expandable_segments（19.5→14.3GB）
- [x] 端到端 `/pipeline` 4 场景全通（329-358s/场景，含超分无 OOM）[verify:: pipeline_report.json]
- [x] 质量评估：4 场景 step1 双手 21/21 点 conf 0.97，inpaint 后 0.98
- [x] 启动脚本 `start_pipe.sh`（setsid 稳定启动）+ 技术文档 `docs/sdxl/hand_pipe.md`
- [x] QUICK_START.md 服务登记（10335）

## 三、质量打磨（待办）

- [x] 输出图客观验收（DWPose 21 点 + 5 指几何，step1-3 全优；超分图修复后手部可检出）[verify:: outputs/hand_pipe/review_*.png]
- [ ] 输出图人工复核（查看 review_*.png 拼图，确认手指形态/数量真实改善）[priority:: high] [owner:: 用户]
- [ ] depth 条件图补齐（MiDaS onnx 下载 + cn_generate depth 分支实测）[priority:: medium]
- [ ] canny 条件图实测（懒加载路径验证，含 cn_pipe 重建显存回收）[priority:: medium]
- [ ] ControlNet 8bit 量化提速（当前 281-309s/张，P40 无 Tensor Core 慢）[priority:: medium]
### □ 3.3 LoRA 手部增强（调研完成，待接入）
- [x] LoRA 调研：5 候选评估（HF 多关键词扫描 + 格式判定）[verify:: models/sdxl_loras/]
- [x] 候选排除记录：jlsim/GoodHands-beta2 gated 需申请；casque 镜像实为 SD1.5 格式；Muapi all-in-one 实为 FLUX 格式；ntc-ai nice-hands slider 实测无效（peace_sign 左手 10→8）
- [x] 有效候选确认：Benevolent/Perfect Hands v2（SDXL te1/te2 完整），实测 peace_sign 左手 10→16、右手 17→19 [priority:: high] [verify:: outputs/hand_pipe/lora_test/peace_sign_perfect_hands_v2_0p6_42.png]
- [x] diffusers 0.39 load_lora_weights 崩溃排查（rank 推断前缀 bug）→ 手动注入方案 `load_kohya_lora_manual`（64 层 Linear，delta=alpha/rank×up@down）
- [ ] perfect_hands_v2 接入管线 step1（pipe_server.py 生成时手动注入，权重 0.6）[priority:: high] [verify:: POST /pipeline 重跑 4 场景]
- [ ] LoRA + ControlNet 联合效果验证（step1 注入 LoRA → step2 CN 骨骼约束是否保持手部增益）[priority:: medium]
- [ ] LoRA 权重扫描（0.4/0.6/0.8 三档，peace_sign 场景为主评估）[priority:: low]
- [ ] 其他 SDXL 手部 LoRA 候选补充（HF 新发布 / civitai 不可达时定期复查）[priority:: low]
- [ ] HandRefiner 精炼节点（需 SD1.5+ControlNet 专用管线，当前 inpaint 替代已达标）[priority:: low] [ref:: docs/sdxl/hand_pipe.md §6.4]
- [ ] 批量场景回归（run_pipeline.py --scenes 0,1,2,3 全量重跑，多 seed 采样）[priority:: low]

## 四、服务化与集成（待办）

- [ ] 10335 纳入 `scripts/start_services.sh` 一键启动（内存 15GB 限制需评估串行加载）[priority:: medium]
- [ ] 与 10331 基础服务互通测试（管线产出图可直接超分/重绘）[priority:: low]
- [ ] 机器重启后服务恢复流程验证（模型缓存热加载提速）[priority:: low]
- [ ] CHANGELOG.md 登记管线能力 [priority:: low]

---

## 状态汇总

| 状态 | 计数 | 说明 |
|------|------|------|
| `- [x]` 已完成 | 18 | 组件测试 + 管线搭建 + 客观验收 + LoRA 调研 |
| `- [/]` 执行中 | 0 | — |
| `- [ ]` 待执行 | 14 | 人工复核 + 打磨/服务化 |
| `- [!]` 失败/阻塞 | 0 | — |
| **总计** | **32** | — |

> 最后更新：2026-08-26
