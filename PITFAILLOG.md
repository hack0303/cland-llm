# PITFAILLOG 踩坑记录

> 记录**为什么**踩坑；CHANGELOG.md 只记录**改了什么**。下次遇到同类症状直接命中根因。

## [工具链] pkill/pgrep -f 会匹配到自己的命令行，导致自杀

- **日期**：2026-08-09 · **模块**：shell 运维（本次部署反复踩 5+ 次）
- **症状**：执行 `pkill -f "hf download"` / `pkill -f "sdxl_convert"` 后命令无输出、进程没杀成，bash 会话直接消失
- **根因**：`pkill -f` / `pgrep -f` 匹配**完整命令行**，而 bash -c 执行的那条命令本身包含目标字符串 → 匹配到自己 → `kill -9` 自杀
- **修复**：先 `pgrep -f xxx` 拿 PID 列表，再用 `for pid in ...; do kill -9 $pid; done` 单独 kill；或匹配更独特的子串
- **预防**：**禁止**在 pkill/pgrep 参数里写会被自身命令行包含的完整字符串；杀进程一律两步走

## [环境] torch cu128 构建不支持 P40 (sm_61)，`cuda.is_available()` 是假象

- **日期**：2026-08-09 · **模块**：conda 环境（gemma_env）
- **症状**：gemma_env 里 `torch.cuda.is_available()` 返回 True，但实际跑 CUDA 计算报 `no kernel image is available for execution on the device`
- **根因**：torch 2.10 (cu128) 构建只含 sm_70~sm_120 内核；P40 是 sm_61（Pascal）。`is_available()` 只探测驱动不验证内核。gemma 之前能跑是因为 llama.cpp 自带 sm_50+ 内核，不走 torch
- **修复**：P40 上用 torch 2.7.1+cu118（base）或 2.6.0+cu118（triposg_env），cu118 构建保留 sm_61 内核
- **预防**：新装 torch 后必须跑一次真实 matmul（`torch.randn(2048,2048,device='cuda') @ x`）验证，不能只看 `is_available()`

## [性能] bitsandbytes 8bit 在 P40 上无加速（71s vs FP16 73s），显存也不降

- **日期**：2026-08-09 · **模块**：SDXL 推理（inference/sdxl）
- **症状**：SDXL `load_in_8bit=True` 后出图 70.9s、峰值显存 10.45GB，与 FP16 模式几乎完全相同
- **根因**：bitsandbytes 的 8bit matmul 内核在 Pascal (sm_61) 上没有 Tensor Core 路径，实际先反量化再以 FP16 计算，等于 FP16 + 额外反量化开销；且 VAE 被 offload 到 CPU 反而增加瓶颈
- **修复**：放弃 8bit，保持 FP16（torch_dtype=float16）；P40 的 INT8 优势（47 TOPS）需要 TensorRT/ONNX 级别的算子融合才吃得到，bitsandbytes 路线走不通
- **预防**：在 P40 上做量化前先小样本实测对比（同 seed 同参数跑一遍计时），不要默认"8bit 一定更快"

## [编译] diso CUDA 扩展：gcc-13 + CUDA 11.8 + glibc 2.39 三方不兼容

- **日期**：2026-08-09 · **模块**：TripoSG 部署（diso-0.1.4）
- **症状**：`pip install diso` 编译报错连环出现：`unsupported GNU version` → `__builtin_dynamic_object_size undefined` → `_Float32/_Float64/_Float128 undefined` → `bits/wordsize.h not found` → `new: No such file or directory` → `bits/c++config.h not found`
- **根因**：系统 Ubuntu 24.04 只有 gcc-13（nvcc 11.8 只支持 ≤11）；glibc 2.39 头文件用了 gcc 12+ 内建函数与 C23 `_Float*` 类型；nvcc 11.8 的预处理器/前端全都不认识。hack 编译参数（`-allow-unsupported-compiler`/`-D_Float*`/补 `-I`）只能逐个击破、按下葫芦浮起瓢
- **修复**：换 **conda gcc-11** 作 host 编译器（`__GNUC__=11` 走 glibc 的 typedef 分支，无需任何宏 hack），`TORCH_CUDA_ARCH_LIST=6.1`，并把 5 个系统 include 路径与 4 个链接路径写死进 diso 的 setup.py
- **预防**：编译 CUDA 扩展前先检查 `gcc --version` 与 `nvcc --version` 的兼容矩阵（CUDA 11.8 ≤ gcc 11 / CUDA 12.4 ≤ gcc 12）；不兼容时直接装 conda-forge 的 gcc 11 工具链，不要试图用编译参数硬闯

## [编译] setup.py 的 `extra_link_args` 必须传给 Extension 对象，不是 `setup()`

- **日期**：2026-08-09 · **模块**：diso 编译
- **症状**：`setup(extra_link_args=[...])` 后链接命令里完全看不到这些参数，报 `cannot find crti.o / -lm / -lgcc`
- **根因**：`extra_link_args` 是 `CUDAExtension(...)` 构造函数的参数；`setup()` 顶层没有这个标准参数，传了会被静默忽略（链接命令用的是 Extension 自带的参数）
- **修复**：改到 `CUDAExtension("diso._C", sources, ..., extra_link_args=[...])` 调用里
- **预防**：给 torch extension 加链接参数时先 grep 生成的链接命令行（`pip install -v`）确认参数真的出现，而不是只看报错

## [部署] TripoSG 官方脚本每次 `snapshot_download` 重复下载 7.5GB 权重

- **日期**：2026-08-09 · **模块**：TripoSG 部署
- **症状**：官方 `inference_triposg.py` 把权重下载到 `pretrained_weights/TripoSG`，即使本地已有完整模型
- **根因**：脚本硬编码 `snapshot_download(repo_id="VAST-AI/TripoSG", local_dir="pretrained_weights/TripoSG")`，无本地复用判断
- **修复**：`ln -sfn /mnt/data/ai_workspace/models/TripoSG pretrained_weights/TripoSG`，让脚本的 local_dir 指向符号链接，跳过下载
- **预防**：部署带自动下载的官方脚本时，先检查 `local_dir` 参数，用 symlink 或预下载占位

## [部署] FastAPI `async def` 里跑同步阻塞代码会卡死整个事件循环

- **日期**：2026-08-09 · **模块**：TripoSG server.py
- **症状**：请求推理期间 `/health` 也无响应，表现为服务"假死"
- **根因**：`async def generate()` 在事件循环内执行，内部 20 分钟的同步 CUDA 推理阻塞了事件循环，所有请求排队
- **修复**：改为普通 `def generate()`，FastAPI 自动丢线程池执行，事件循环不被阻塞
- **预防**：CPU/GPU 密集同步任务一律用 `def`（非 `async def`）；只有真正的 IO 异步才用 async

## [网络] pip/conda 官方源在部分网络下会无限卡死，需换国内镜像

- **日期**：2026-08-09 · **模块**：环境搭建
- **症状**：`pip install torch` 从 pytorch 官方源跑 15 分钟无进展（无网络活动）；conda-forge repodata 拉取 10 分钟卡住
- **根因**：pypi/pytorch/conda-forge 官方源在该网络环境不可达或极慢，pip/conda 重试机制不报错、无限等待
- **修复**：pip 用 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；torch/torchvision wheel 从阿里云镜像 `https://mirrors.aliyun.com/pytorch-wheels/cu118/` 直接 curl 下载（注意 HTML 里 `+` 转义为 `&#43;`）；conda 用 `--offline` + 直接指定 pkgs 缓存里的 `.conda` 文件路径
- **预防**：大包下载前先 curl 测速（`-w "%{speed_download}"`）；pip 超时无进展（`ss -tnp` 无网络连接）立即杀掉换镜像

## [工具链] bash 工具 timeout 会杀死整个进程组，nohup 不防 SIGTERM

- **日期**：2026-08-15 · **模块**：图生视频部署（ComfyUI 服务）
- **症状**：ComfyUI 采样跑到 80%（16/20 步）时进程消失，日志无任何报错；同机 `pgrep` 找不到进程，显存已释放
- **根因**：后台 `nohup python main.py &` 与前台 `python generate.py` 同属一个 bash -c 进程组；bash 工具 900s 超时杀进程组 → ComfyUI 收到 SIGTERM。`nohup` 只忽略 SIGHUP，**不防 SIGTERM**
- **修复**：用 `setsid env ... nohup python main.py ... < /dev/null &` 启动常驻服务，脱离进程组
- **预防**：常驻服务一律 setsid 启动；长任务前台执行时 bash 工具 timeout 给足余量

## [兼容] AnimateDiff + IPAdapter 在 P40 上报 query/key dtype 不一致（fp32 × fp16）

- **日期**：2026-08-15 · **模块**：图生视频（ComfyUI + AnimateDiff-Evolved + IPAdapter_plus）
- **症状**：KSampler 报 `Expected query, key, and value to have the same dtype, but got query.dtype: float key.dtype: c10::Half`
- **根因**：AnimateDiff-Evolved 的 `_diffusion_model_groupnormed_wrapper` 把 groupnorm 输出转 fp32（数值稳定），下游 cross-attention 的 query 变 fp32；IPAdapter_plus 注入的 ip_k/ip_v 是 fp16 → SDPA 拒绝混合精度。P40 无 Tensor Core 走 PyTorch 标准路径，严格校验 dtype；消费卡（flash-attn 路径）同样会炸
- **修复**：`ComfyUI_IPAdapter_plus/CrossAttentionPatch.py` 在 `optimized_attention` 前对齐 dtype：`if q.dtype != ip_k.dtype: ip_k = ip_k.to(q.dtype); ip_v = ip_v.to(q.dtype)`
- **预防**：第三方 patch 类节点（cross-attention 注入）在 P40 上都要检查 dtype 对齐；改代码后必须重启 ComfyUI 才生效（模块常驻内存）

## [工具] ffmpeg 直接解码动画 webp 失败（exit 69），PIL 可正常逐帧提取

- **日期**：2026-08-15 · **模块**：图生视频输出管线
- **症状**：`ffmpeg -i out.webp -c:v libx264 out.mp4` 报 `Invalid data found when processing input` / `error code: -22`
- **根因**：ComfyUI SaveAnimatedWEBP 产出的动画 webp 带 alpha/特殊块，ffmpeg 内置 webp 解码器（libwebp 解静态）对多帧动画支持不佳
- **修复**：PIL `Image.open` → 逐帧 `seek(i)+save(png)` → `ffmpeg -framerate N -i f%03d.png` 拼接 mp4（generate.py 已内置）
- **预防**：动画 webp → mp4 一律走 PIL 提取帧；或 ComfyUI 直接输出 PNG 序列

## [工具链] pip 默认源（pypi.org）挂起无响应，换清华镜像秒装

- **日期**：2026-08-15 · **模块**：环境（base，opencv/scikit-image 安装）
- **症状**：`pip install opencv-python-headless` 长时间无输出直到超时（300s+）；期间误以为进程卡在业务代码
- **根因**：本机访问 pypi.org 不稳定（直连挂起）；清华镜像可达
- **修复**：`pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple`
- **预防**：本机 pip 一律加 `-i` 清华源；pip 挂起时先怀疑网络而非业务代码（用 `pip install -v` 或直接换源验证）

## [能力边界] 文生图"一图多视图/多格排版"不可靠（三视图/表情/动作表）

- **日期**：2026-08-15 · **模块**：角色设定工作流（char_sheet.py）
- **症状**：SDXL prompt 要求三视图（character sheet, three views）输出排版失控：背景非白（实测 90% 非白像素）、视图重复/布局混乱；表情 prompt（close-up face）输出全身比例，表情不体现
- **根因**：
  - 文生图模型擅长"画一张好看的画"，不擅长"画一张信息准确的图纸"——三视图是排版/布局任务，模型把"三视图"当风格词而非布局指令；训练数据里带场景的设计稿图占多数，白底纯排版图稀缺
  - 多视图一致性需要 3D 概念（同物体转 90°），文生图只有文本→像素统计映射，做不到
  - 表情失败叠加角色设计因素：戴头盔/面具的角色（宇航员）脸部被遮挡，表情天然无法表达
  - 表情/动作图独立文生图无锚定 → 与定妆照长相不一致
- **修复**：三视图拆三张单视图（front/side/back 各一张）→ PIL 拼版；表情独立特写模板（close-up portrait, face filling the frame）；动作表按生产标准扩至 14 动作 + 6 表情；文档明确"头盔角色表情受限"
- **预防**：涉及"排版/图纸/多视图"类需求，先拆成单张生成再程序化拼版；角色资产先确认面部可见性；跨图一致性默认不可靠，需要时用 IPAdapter 锚定或人工挑选

## [工具] ffmpeg 音频链顺序：先 atrim 再 adelay（反序语音全丢）

- **日期**：2026-08-15 · **模块**：compose.py 音频混音
- **症状**：多段配音混音后只听得到第 1 句，其余全静音
- **根因**：`adelay=2300,atrim=0:1.8` 先插入 2.3s 静音再截取前 1.8s——截掉的 1.8s 全是插入的静音，语音内容（2.3s 之后）被剪没
- **修复**：顺序反转为 `atrim=0:{limit},adelay={ms}|{ms}`（先截原音频，再延迟到目标位置）
- **预防**：ffmpeg filter 链里时间偏移类 filter（adelay/atrim/setpts）严格按"先内容后位置"排序；合成后用 volumedetect 分段验证每段能量

## [音频] loudnorm 双坑：采样率 192kHz + PTS 爆炸（无声）

- **日期**：2026-08-16 · **模块**：V1/V2 音频合成（compose.py）
- **症状**：合成后成片无声/播放异常；ffmpeg 显示 Audio 96000Hz、位速 27Mbps 异常；提取音频流时长 488 万小时
- **根因**：① loudnorm 内部 192kHz 处理，输出采样率被抬到 192kHz（源 16kHz）；② loudnorm 输出流的 PTS 异常（缓冲/时间基 bug），多路 amix 后某路 PTS 爆炸 → mux 时长 488 万小时
- **修复**：loudnorm 后链 `aresample=16000,asetpts=PTS-STARTPTS`；**amix 后也要 `asetpts=PTS-STARTPTS`**（amix 输出的时间基异常）
- **预防**：loudnorm 必须搭配 aresample+asetpts；合成后验证三件套：采样率 16kHz / 位速 <200k / 音频解码时长 ≈ 视频时长

## [ComfyUI] 工作流模板提示词残留（case001 的猫引导每镜头动画）

- **日期**：2026-08-16 · **模块**：workflow_i2v.json 节点 5/6
- **症状**：动画效果不对（角色往猫方向生成、运动语义错）——排查发现正向提示词还是 case001 的 "a cute cat waving its paw, subtle motion"
- **根因**：workflow JSON 是模板，节点 5/6 的 text 是静态字段；跑 Lumo 时只动态改了 image/seed/帧数，**没改提示词** → 模板残留
- **修复**：pipe_anim 动态写入节点 5（分镜 motion_prompt）+ 节点 6（全约束负向）
- **预防**：workflow 模板的"内容字段"（提示词）必须由调用方显式覆盖；排查效果问题时先检查提交的工作流内容（POST /prompt 的 body 可回看）

## [ComfyUI] AnimateDiff Motion LoRA 用法三坑（目录/参数名/依赖环）

- **日期**：2026-08-16 · **模块**：Motion LoRA 接入
- **症状**：① LoRA 选项列表空（放错目录）；② 400 报 required_input_missing: name；③ Dependency cycle detected
- **根因**：
  - Motion LoRA 目录是 `models/animatediff_motion_lora/`（不是 loras/）
  - ADE_AnimateDiffLoRALoader 参数是 `name`（不是 lora_name），且**不需要 model 输入**（纯输出 MOTION_LORA 对象）
  - 若把 loader 的 model 输入接回 ADE loader 的 model → 依赖环
- **修复**：目录放对 + name 参数 + 节点 15 只接 name/strength，输出接 ADE loader 的 motion_lora 输入
- **预防**：新节点先查 object_info（参数名/输入输出/目录映射），再改 workflow；400 错误信息有 node_errors 明细

## [ComfyUI] prompt 缓存命中：同 seed 提交秒回 completed 但输出文件不存在

- **日期**：2026-08-16 · **模块**：pipe_anim
- **症状**：I2V 提交 5 秒返回 completed，history 有输出记录，但 output 目录找不到文件
- **根因**：ComfyUI 对相同工作流（同 seed/同输入）返回缓存结果；旧记录的输出文件已被清理 → history 有、磁盘无
- **修复**：换 seed（pipe_anim seed 500+）；规避缓存
- **预防**：批量任务用递增 seed；取输出后必须验证文件存在（FileNotFoundError 兜底）

## [路径] ComfyUI 输出可能带 subfolder，硬拼 output/ 找不到

- **日期**：2026-08-16 · **模块**：pipe_anim
- **症状**：FileNotFoundError: output/anim_s301_00001_.webp（history 记录存在但路径错）
- **根因**：history 输出 image 对象有 filename + subfolder 字段；部分节点输出带 subfolder 子目录，硬拼 `output/{filename}` 漏了 subfolder
- **修复**：`os.path.join(output_dir, img.get("subfolder",""), img["filename"])`
- **预防**：所有 ComfyUI 输出路径统一从 history 的 filename+subfolder 拼接

## [流程] storyboard output 字段残留导致"已存在跳过"误判

- **日期**：2026-08-16 · **模块**：run_story Phase 0 配音
- **症状**：清空 audio 目录后重跑，配音全跳过（Phase 0 完成但 audio 空）
- **根因**：判断条件用 `"voice" not in clip["output"]`（字段残留判断），storyboard.json 上一轮回填的 output 字段还在 → 误判已存在
- **修复**：改为**文件存在判断**（os.path.exists(audio/sceneXXX_voice.wav)）
- **预防**：幂等判断一律以"产物文件存在"为准，不要以元数据字段为准（字段可残留/可伪造）
