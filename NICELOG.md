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

## 三、模型部署与复现（System One 开源复现）

### 8. 上游依赖新版 API 时的 sitecustomize no-op shim（不改上游源码）
- **实践**：NanoJev 上游记录 `torch==2.14.0`，评估脚本 `from torch._native import triton_utils; deregister_op_overrides()`；P40 只能用 torch 2.7.1（无该模块）。写 `sitecustomize.py` 在 import 期检测缺失后注册 no-op stub（torch 2.7 本就没有需要关闭的 native Triton overrides），经 `PYTHONPATH` 自动生效，上游源码 0 行改动
- **价值**：上游 release 的脚本 sha256 校验值保持原样（复现证据可信）；一处 shim 同时修复 maze/snake/native-qwen 三组脚本；未来 torch 自带该模块时 shim 自动让位
- **落地**：`inference/nanojev/sitecustomize.py` + `start_server.sh` / `run_bench_*.sh` 的 PYTHONPATH 导出
- **复用**：凡「上游代码调用新版框架 API、本机只能用旧版」的场景，优先 shim 而非改源码；no-op 前提是该 API 在旧版语义上无需动作

### 9. P40 部署决策：默认 fp32，bf16 仅作对照（实测 1.5x 差异）
- **实践**：torch 2.7 的 `torch.cuda.is_bf16_supported()` 在 P40（sm_61）返回 True（模拟路径），服务默认选 bf16；实测 fp32 单次 4 题决策 136.7ms vs bf16 200.5ms，且 fp32 的 50×50 maze 结果与官方 A100-bf16 记录完全一致（244/36）
- **价值**：避免"API 说支持就默认 bf16"的隐性 1.5× 性能损失；同精度口径还让复现数字对齐
- **落地**：`inference/nanojev/start_server.sh` 默认 `--precision fp32`；`docs/systemone/README.md`
- **复用**：Pascal 及更老卡上先跑 dtype 对照再定默认值；`is_bf16_supported()` 只作参考，不作决策依据

### 10. 上游测试套件即部署验收：von 23 passed 作为 drop-in 证据
- **实践**：von 部署后用其自带 `tests/`（FastAPI TestClient + OpenJev 契约兼容 + primitives/fanout/patterns/presets）做验收：CUDA fp32 下 **23 passed / 1 deselected**（trio 为可选依赖）；其中 `test_fanout` 的阈值断言（noul>0.5、score>1.0）直接暴露了 P40 模拟 bf16 的精度回退（CPU 过 / CUDA 挂）
- **价值**：无需自造验收集，上游测试即「协议对等 + 最小质量门禁」；阈值型断言是可复现的 dtype 问题探针（对照 CPU/CUDA 一次定位）
- **落地**：`inference/nanojev/patches/von-p40-fp32.patch` + `start_von.sh` 默认 `VON_DTYPE=fp32`；结果记录在 `inference/nanojev/README.md` §5.6 与 `docs/systemone/README.md`
- **复用**：部署任何开源模型服务，先跑上游测试套件再自评；对「两路 softmax/阈值」模型固定用 fp32 并保留一组 dtype 对照

## 四、AI 修图产线（#179）

### 11. 灰度老照片上色：大强度重绘取色 + LAB 色度迁移保结构
- **实践**：colorize 模式让 SDXL img2img 以 strength 0.8 重绘生成色彩（提示词用场景/油画描述 + 负面词压 black and white/gray/monochrome），随后 `chroma_transfer`：最终图 = 修复后原图的 L 通道 + AI 输出的 a/b 色度（cv2 LAB）
- **价值**：解决两个矛盾目标——低强度上色无效（cf 1-11，模型保持灰度）、高强上色才出色（cf 19.5-57.7）；色度迁移后脸部/构图零漂移（亮度完全沿用原图），上色再激进也不毁图
- **落地**：`inference/photo/retouch.py`（chroma_transfer）；实测 hist01 cf 0→19.5、油画对拍 18.7→57.7（GT 48.1），PP 参见 `docs/photo-retouch-v0.md` §6
- **复用**：任何"给灰度/单色内容加色彩"的场景（老照片/线稿/地图）通用——结构通道与色彩通道解耦，AI 只负责猜色

### 12. 无视觉模型时的修图验收：合成退化对拍集 + 无参指标 + 代理评分 + 总览拼图
- **实践**：高清 CC0 GT → `degrade.py` 合成"老照片"输入（噪声/模糊/划痕/灰度/降采样/JPEG）→ 批量修复 → 对拍集算 PSNR/SSIM（native + 统一 1024 口径），全量算无参指标（Laplacian 清晰度/中值残差噪声/色彩度）+ 分档代理评分，最后拼 11 案例 Owner 总览图
- **价值**：本地无 VLM 也能给出可复现的质量底线与方案对比（如 AI 修复 vs 纯超分基线的保真差异：SSIM 0.40 vs 0.52）；"合成退化"让无 GT 的老照片场景也有了可算指标的主体，评审人只需看图
- **落地**：`inference/photo/{degrade.py,evaluate.py}`；报告 `docs/photo-retouch-v0.md` §5/§6；产物 `outputs/photo-retouch/{metrics.json,owner_overview.png}`
- **复用**：任何图像生成/修复任务，先造"退化-复原对"再上指标；无参指标用于没有 GT 的样例，代理评分只做趋势不做定论（须标注"待人工校准"）

## 六、上色 / 修图（#224/#225）

### 1. 上色 = DDColor 色度 + work 亮度（D2）：保结构、消串色
- **实践**：上色出色用 **DDColor-L**（结构保真），再 `chroma_transfer(DDColor出色, work)` 只取**色度 a/b**、保留**原图亮度 L**；**不套 skin_fix**
- **价值**：皮肤蓝晕 0 · 色度最足最自然 · 结构对齐最好（#225 四路对比：D2 edge_align 0.243 / L_shift 0.076；对比 SDXL 蓝块 / 越权）
- **落地**：`inference/photo/retouch.py --colorizer ddcolor`
- **复用**：通用 img2img 上色易串位时，换结构保真专用模型 + 只迁色度

### 2. 四选/海报图按**焦点**裁剪（`cover_bias`）避免人物截头
- **实践**：`gen_service` 的缩略图用 `cover_bias(im, w, h, fy=…)` 按焦点裁剪（修复/高清看上 12–20%，电商 55%），替代居中 `cover`
- **价值**：全身像是居中 cover 会**截头**；焦点裁剪保证人物头部完整
- **落地**：`inference/photo/marketing.py`
- **复用**：任何"竖构图缩略进方框"场景按主体位置设 fy
