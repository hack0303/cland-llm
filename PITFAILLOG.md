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
