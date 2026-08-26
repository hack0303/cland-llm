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

## [模型] Gemma-4 的 DSL 与 OpenAI API 是两套协议，适配要拆"渲染/解析"两层看

- **日期**：2026-08-23 · **模块**：gemma4 推理服务（inference/gemma/gemma4.py）
- **症状**：gemma4 工具调用"指令不兼容 OpenAI API"；曾误以为是 ollama 适配问题，实际自建 llama-cpp-python 服务也绕不开
- **根因**：Gemma-4 原生 DSL（`<|turn>` 分轮、`<|channel>thought` 思考、`<|tool_call>call:name{args}` 工具调用、`<|tool_response>` 工具返回）与 OpenAI JSON 协议（messages/tool_calls/reasoning_content）是两套东西，引擎必须做双向翻译。实测 llama-cpp-python 0.3.20：**渲染层 ✅ 自动用 GGUF 内嵌官方模板**（Jinja2ChatFormatter，手写的 GEMMA_4_JINJA 可以删）；**解析层 ❌ 不把 DSL 转成 tool_calls JSON**（finish_reason 仍是 stop）——这层必须自己写
- **修复**：删手写 Jinja 模板，保留手写转换层（正则提取 thought/tool_call + 清洗标签 + 有 tool_calls 时 content 置 None）；工具调用时加 stop 防"自导自演"
- **预防**：换模型先分清"模板渲染"和"协议解析"两层是否被引擎覆盖；GGUF 里带 tokenizer.chat_template 的模型优先让引擎渲染，别急着手写模板

## [模型] llama-cpp-python 流式工具调用：stop 参数失效 + DSL 跨 token 泄漏

- **日期**：2026-08-23 · **模块**：gemma4.py 流式 SSE
- **症状**：stream=true 时原始 `<|tool_call>` 标签一个 token 一个 token 漏给客户端，模型继续"自导自演"输出假的 `<|tool_response>` 和结果文本；非流式下同样的 stop 参数是有效的
- **根因**：① llama-cpp-python 流式模式不执行 stop 序列（物理截断只能自己做）；② DSL 标签被切碎成多个 token 到达，逐 token 透传必然泄漏；③ 工具参数里的转义标签 `<|"|>` 会让"尾部暂缓"判断误判（rfind("<") 找到的是转义标签的 `<`）
- **修复**：流式状态机——buffer 累积，检测到 `<|tool_call` 出现即**暂停下发一切文本**，攒到完整闭合标签后一次性转成 OpenAI tool_calls delta 并 return 物理截断；thought 块完整闭合后才转 reasoning_content 下发
- **预防**：流式处理任何带 DSL 的模型都先想"标签会不会跨 token"，一律缓冲到完整闭合再动作；不要依赖 stop 参数做工具截断

## [模型] ollama 报 `unknown model architecture: 'gemma4'` ≠ P40 不兼容（诊断顺序）

- **日期**：2026-08-23 · **模块**：ollama 0.18.2（gemma4 加载）
- **症状**：ollama 拉 gemma4 报 `error loading model: unknown model architecture: 'gemma4'`，一度怀疑"新版 ollama 不兼容 P40"
- **根因**：**版本太老**——ollama 0.18.2（3 月版）内核没有 gemma4 架构支持；与 GPU 无关。实测 ollama 0.18.2 的 cuda_v12 runner（CUDA 12.8）在 535 驱动上 `cudaGetDeviceCount=2` 正常识别 2 张 P40，libggml-cuda.so 含 sm_61 cubin；只有 cuda_v13（CUDA 13）需要 r580+ 驱动。llama.cpp b10488（新版 ollama 内核）源码也保留 61-virtual
- **修复**：gemma4 走自建 llama-cpp-python 0.3.20（内核认识 gemma4 + sm_61）；ollama 需升级到支持 gemma4 的新版（v0.32.15+）
- **预防**：模型加载失败先看日志里的架构报错，别甩锅给 GPU/驱动；验证"引擎 vs 驱动"兼容性的顺序：dlopen runner → cudaGetDeviceCount → 真跑一个模型

## [模型] langchain Chroma.add_texts 忽略传入的 embeddings，索引维度从没生效过

- **日期**：2026-08-23 · **模块**：embedding 漏斗检索（PythonTest example_embeddings_03/04）
- **症状**：03 脚本"256 维索引 + metadata 存全量"方案，实测 Chroma 集合里存的却是 1024/768 维全量向量；04 脚本用 add_texts 传 256 维也报 `Embedding dimension 256 does not match collection dimensionality 1024`
- **根因**：langchain-community 0.0.38 的 `add_texts` 签名只有 `(texts, metadatas, ids, **kwargs)`——**没有 embeddings 参数**！传进去被 `**kwargs` 静默吞掉，函数内部自己调 `embedding_function.embed_documents()` 重新生成全量维度。Matryoshka 截断索引方案因此从未真正生效（存的是全量，只是"看起来"没问题）
- **修复**：弃用 langchain 包装器，直接走 **chromadb 原生 API**（`PersistentClient` + `collection.add(embeddings=...)` / `collection.query(query_embeddings=...)`），不指定 collection 的 embedding_function，维度完全由传入向量决定
- **预防**：任何"手动控制向量维度/自定义 embeddings"的场景，先查库版本签名（`inspect.signature`）确认参数真的存在，再写业务代码；传参前先 `col.get(include=['embeddings'])` 验证存储维度

## [diffusers] 0.39 弃用 `load_in_8bit=`：pipeline 层要 PipelineQuantizationConfig，模型层要模型级 Config

- **日期**：2026-08-26 · **模块**：手脚修复管线（inference/sdxl/hand_pipe）
- **症状**：`from_pretrained(..., load_in_8bit=True)` 打出 "Keyword arguments not expected ... will be ignored" 警告，8bit 未生效，fp16 全量加载 → 多模型常驻直接 OOM 崩机
- **根因**：diffusers 0.39 起 pipeline 级 8bit 必须 `PipelineQuantizationConfig(quant_backend="bitsandbytes_8bit", quant_kwargs={"load_in_8bit": True}, components_to_quantize=[...])`；而模型层（ControlNetModel.from_pretrained）只认模型级 QuantizationConfigMixin，传 PipelineQuantizationConfig 报 `no attribute quant_method`
- **修复**：pipeline 用 PipelineQuantizationConfig；ControlNet 单独加载时不量化（fp16），或由 pipeline 的 components_to_quantize 统一处理
- **预防**：升级 diffusers 后先查 CHANGELOG/源码确认量化 API；加载后看 `torch.cuda.max_memory_allocated` 判断 8bit 是否真生效（8bit unet 约 fp16 一半）

## [diffusers] `controlnet=[单模型]` 报 "pipeline is not a Module subclass"；0.39 必须显式 MultiControlNetModel

- **日期**：2026-08-26 · **模块**：管线 cn_pipe 加载
- **症状**：`StableDiffusionXLControlNetPipeline.from_pretrained(controlnet=[cn])` 在构造器 `MultiControlNetModel([cn])` 处抛 `TypeError: ...StableDiffusionXLControlNetPipeline is not a Module subclass`（ModuleList 里混入 pipeline 对象）；换单实例 `controlnet=cn` 又在 check_inputs 走 `assert False`
- **根因**：0.39 量化路径下 from_pretrained 对 list 参数处理有 bug（list 元素被替换/污染）；单实例时 8bit 量化包装导致 controlnet 类型既不是 ControlNetModel 也不是 MultiControlNetModel，check_inputs 的 else 分支直接 assert False
- **修复**：先 `ControlNetModel.from_pretrained` 加载模型，再显式 `MultiControlNetModel([cn])` 传实例；调用时 image 传 `[cond_img]`、controlnet_conditioning_scale 传 `[strength]`（Multi 约定）
- **预防**：0.39 下 ControlNet 相关一律走「显式 MultiControlNetModel + list 传参」模板，别信老版本的单实例写法

## [超分] RRDBNet 权重加载静默失败：strict=False + 命名不匹配 → 黑图/噪声，mean 值还骗过了验收

- **日期**：2026-08-26 · **模块**：rrdbnet.py
- **症状**：4x-UltraSharp 超分输出全黑（mean 4.5）；RealESRGAN_x4plus 输出 mean 176.7 看似正常实为随机噪声；端到端 step4 被 DWPose 检出 0 人
- **根因**：三层问题叠加——① 我的 RRDBNet 实现是简化 5C 块，真实架构是 RDB1/2/3 三子块堆叠（RRDB）；② 4x-UltraSharp 是 ComfyUI 命名（model.0 / 大写 RDB / 紧凑层号 3,6,8,10），x4plus 是官方小写 rdb 命名；③ `load_state_dict(strict=False)` 全不匹配时**静默丢弃**，模型以随机权重运行
- **修复**：重写为正确 RRDB 架构（rdb1/2/3 小写，匹配官方）+ `_adapt_keys` 双格式适配（官方直通 / ComfyUI 去 model 前缀+大写转小写+去 .0 序号）+ 加载后校验（missing>10 或 unexpected>10 直接 raise）
- **预防**：加载预训练权重后**必须校验匹配数**（missing/unexpected 计数），并跑真实输入看输出分布（mean/std 与输入同量级）；strict=False 是静默失败重灾区

## [检测] DWPose yolox onnx 不是"resize 就能跑"：标准 grid+stride decode + SimCC 双输出

- **日期**：2026-08-26 · **模块**：dwpose.py
- **症状**：直接 resize 640 + 简单阈值 → 检不出人（conf 全 ~0.0003）；关键点模型输出 shape (1,133,576)/(1,133,768) 不是 heatmap，argmax 后坐标全在边缘（335124 这种天文数字）
- **根因**：yolox_l.onnx（yzd-v/DWPose）输出是**未解码**的 head 输出，必须 `(tx+grid)*stride` 解码 + `exp(tw)*stride`；dw-ll_ucoco_384.onnx 是 RTMPose SimCC 表示（simcc_x/simcc_y 双输出），置信度 = (max_x+max_y)/2，位置 ÷simcc_split_ratio=2，且预处理是 bbox→center/scale(×1.25)→仿射变换（无黑边）+ ImageNet 归一化（RGB, float32）
- **修复**：按 sd-webui-controlnet 的 cv_ox_det.py/cv_ox_pose.py 标准实现重写（grid+stride decode + NMS、SimCC 解码、top_down_affine）
- **预防**：onnx 模型先打印输入输出 shape 并对照官方推理代码，不要凭经验猜后处理；检测类模型输出 conf 全低时先怀疑 decode/预处理而非阈值

## [环境] 15GB 内存被 100 个 torch compile_worker 吃光 → 整机 OOM 重启（服务全部丢失）

- **日期**：2026-08-26 · **模块**：机器运维
- **症状**：服务加载途中整机重启（/tmp 日志清空、GPU 全空）；事后 `free -g` 显示 15GB 全满，`pgrep -fc compile_worker` 100 个
- **根因**：torch 2.7 每次 import 触发 inductor compile_worker 池（默认 32 个/进程 × 3 个服务进程 ≈ 100 个），每个 ~150MB，15GB 内存直接耗尽触发 OOM killer
- **修复**：启动命令加 `TORCHINDUCTOR_COMPILE_THREADS=1`；15GB 机器上**串行**启动多服务（同时加载权重峰值内存超限）
- **预防**：小内存机器跑 torch 服务先设 compile 线程数；内存不足时优先查子进程数而非只盯主进程

## [显存] 多 ControlNet 常驻导致推理峰值 OOM：最小常驻集 + empty_cache + expandable_segments

- **日期**：2026-08-26 · **模块**：pipe_server.py
- **症状**：base 8bit + 3×CN fp16 + inpaint 8bit 常驻 19.5GB，一次推理后 22.4GB，`/upscale` 需 4GB → OOM
- **根因**：P40 24GB 显存，静态模型 + 推理峰值缓存（cuda cache 不自动释放）叠加超限；depth/canny 非主线环节也常驻浪费 5GB
- **修复**：只常驻 openpose（管线核心），depth/canny 懒加载重建 cn_pipe；每次推理后 `torch.cuda.empty_cache()`；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 防碎片。静态 19.5→14.3GB，4 场景端到端含超分全通过
- **预防**：多模型服务先算静态总和，留 ≥6GB 推理余量；非主线组件一律懒加载

## [diffusers] load_lora_weights 对 kohya 单文件报 rank 推断 IndexError（前缀 bug）→ 手动注入

- **日期**：2026-08-26 · **模块**：LoRA 接入（test_lora.py）
- **症状**：`pipe.load_lora_weights(perfect_hands_v2.safetensors)` 抛 `IndexError: list index out of range`（get_peft_kwargs 里 rank_dict 为空）；手动转换后 `lora_linear_layer` 格式又加载出 0 权重（警告 No LoRA keys found）
- **根因**：0.39 的 `_load_lora_into_text_encoder` 用裸模块名拼 `{name}.lora_B.weight` 推断 rank，但转换后的 key 带 `text_encoder.` 前缀 → 永不匹配 → rank_dict 空；而 unet 转换输出 `lora.down/up` 与 peft 的 `lora_A/B` 又不对应
- **修复**：放弃 diffusers LoRA 系统，实现 `load_kohya_lora_manual`：`_convert_non_diffusers_lora_to_diffusers` 转出 lora_linear_layer.down/up + alpha → `delta = alpha/rank × up@down` 直接 add 到模块权重（64 层 Linear，scale 0.6 生效）
- **预防**：diffusers LoRA 加载崩/0 权重时，先验证转换链（kohya→lora_linear_layer→peft 的 key 前后缀），不行就手动注入——LoRA 数学就是 delta=alpha/rank×up@down，不依赖库实现
