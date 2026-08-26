# NICELOG 亮点实践记录

> 记录**为什么这么做是对的**；CHANGELOG.md 只记录**改了什么**。下次做相似功能直接复用正确姿势。
> 与 PITFAILLOG.md 成对使用——「怎么做对了」进本表，「怎么踩坑了」进 PITFAILLOG。

## 一、推理管线（SDXL / ControlNet / 超分 / LoRA）

### 1. 无人工看图时的质量验收：DWPose 客观量化（21 点完整度 + 5 指几何单调性）
- **实践**：用 DWPose 检测生成图的手部 21 点（conf>0.3 计数）+ 手指几何检查（每指链 根→中→尖 距手腕距离递增，≥4 指完整才判 OK），量化对比「基线 vs 提示词 vs ControlNet vs LoRA」各方案
- **价值**：AI 无法直接看图时，把"手指崩坏"变成可测指标；实测能区分方案优劣（基线 peace_sign 左手 10/21 conf 0.49 → LoRA 后 16/19），且发现超分输出黑图/噪声（DWPose 检出 0 人）
- **落地**：`inference/sdxl/hand_pipe/acceptance.py`（finger_metrics + 拼图对比）
- **复用**：任何"生图质量对比"场景照抄；超分/放大类输出必须过一遍检测器（shape 对不代表内容对）

### 2. 超分权重双格式适配器：官方/ComfyUI 命名统一映射 + 加载后校验
- **实践**：`RealESRGANUpscaler._adapt_keys` 把官方（`body.0.rdb1.conv1.weight`）与 ComfyUI（`model.1.sub.0.RDB1.conv1.0.weight`）统一映射到自实现 RRDBNet（rdb1/2/3 小写）；加载后校验 `missing>10 或 unexpected>10 直接 raise`
- **价值**：两个 67MB 模型（4x-UltraSharp / RealESRGAN_x4plus）一次适配全部可用；校验机制杜绝 strict=False 静默失败（曾致黑图/噪声骗过 mean 检查）
- **落地**：`inference/sdxl/hand_pipe/rrdbnet.py`
- **复用**：任何 torch 预训练权重加载照抄「适配器 + 匹配数校验」；strict=False 一律配计数断言

### 3. 绕开 diffusers LoRA 系统的手动注入：delta = alpha/rank × up@down
- **实践**：`load_kohya_lora_manual` 解析 kohya 单文件（`_convert_non_diffusers_lora_to_diffusers` 转 lora_linear_layer 格式），对每个模块 `mod.weight.add_(scale × alpha/rank × up@down)`，64 层 Linear 注入生效
- **价值**：diffusers 0.39 的 load_lora_weights 对 kohya 文件 rank 推断崩溃（前缀 bug），手动注入不依赖库实现、可控可验证；LoRA 数学本质就一行
- **落地**：`inference/sdxl/hand_pipe/test_lora.py`（load_kohya_lora_manual）
- **复用**：任何 diffusers LoRA 加载异常时的兜底方案；接入管线时把注入放 step1 生成前

### 4. P40 显存组合拳：最小常驻集 + empty_cache + expandable_segments
- **实践**：只常驻管线核心（base 8bit + openpose CN + inpaint 8bit ≈ 14.3GB），depth/canny 懒加载；每次推理后 `torch.cuda.empty_cache()`；启动加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **价值**：24GB P40 上静态 19.5→14.3GB，超分环节从 OOM 到全通过；4 场景端到端 328-358s 稳定
- **落地**：`pipe_server.py`（load_cn 懒加载 + _gen 尾部 empty_cache）+ `start_pipe.sh`
- **复用**：多模型服务先算静态总和留 ≥6GB 余量；推理型服务每次调用后释放缓存是固定套路

### 5. 手脚修复管线正确顺序：txt2img → 骨骼约束 → 局部重绘 → 最后超分
- **实践**：`/pipeline` 编排 step1 txt2img → step2 ControlNet OpenPose（手部 21 点锁定）→ step3 DWPose 检测手部 bbox → inpaint 重绘 → step4 全局超分
- **价值**：先修手脚再放大（先放大后修会固化畸形）；ControlNet 是核心环节（step2 后 4 场景双手 21/21 点 conf 0.97），inpaint 精修 conf 至 0.98
- **落地**：`pipe_server.py`（/pipeline endpoint）+ `docs/sdxl/hand_pipe.md` §4.6
- **复用**：任何生图质量管线照抄此顺序；「检测→修复→放大」三段式是通用套路

### 6. LoRA 调研的格式判定法：先验 key 命名再下载大文件
- **实践**：候选 LoRA 先 `safetensors` 读 key 判定格式——`lora_te1/te2_` = SDXL、`lora_te_` = SD1.5、`diffusion_model.double_blocks` = FLUX；5 候选 2 个白下载（GoodHands 镜像实为 SD1.5、Muapi 实为 FLUX）
- **价值**：672MB 的 Muapi 下载后才发现是 FLUX 格式，白费带宽；key 判定 10 秒出结论，避免反复试错
- **落地**：`TODO-hand-pipe.md` §3.3（候选评估记录）
- **复用**：下载任何 LoRA/微调权重前先看 key 前缀判定底座模型；SDXL 认 `lora_unet_`/`lora_te1_/te2_` 前缀

## 二、验证方法论

### 7. 端到端管线验收包：客观数据 + 对比拼图 + README 说明打包交付
- **实践**：验收材料按 `1_review/（5 列拼图）/ 2_upscale/（修复后大图）/ 3_lora/（方案对比）/ README.md（验收要点+数据）` 分类打包 zip，附 DWPose 客观指标
- **价值**：评审人 10 分钟看完所有对比（同 seed 同 prompt 多方案并排），客观数据与主观图互相印证；"评审-反馈"闭环可复用
- **落地**：`outputs/hand_pipe_acceptance_20260826.zip`（生成脚本见 acceptance.py 拼图逻辑）
- **复用**：任何需要外部评审的功能交付照抄「拼图并排 + 客观指标 + README 要点」三件套

> 最后更新：2026-08-26
