# TripoSG 图生 3D 服务

基于 VAST-AI/TripoSG（1.5B rectified flow transformer，图生 3D）的常驻 API 服务。
代码见 `inference/triposg/`，部署于 **GPU 1 / 端口 10332**。

## 环境（triposg_env，python 3.10）

```bash
conda activate triposg_env
# torch 2.6.0+cu118（阿里云镜像 wheel，非 pip 官方源）
# 依赖：diffusers transformers torchvision 0.21.0 einops opencv-python trimesh
#       omegaconf scikit-image numpy==1.22.3 peft jaxtyping typeguard pymeshlab fastapi uvicorn
```

## 启动

```bash
cd /mnt/data/ai_workspace/cland-llm/inference/triposg
CUDA_VISIBLE_DEVICES=1 nohup python3 server.py --port 10332 > /tmp/triposg_server.log 2>&1 &
```

首次加载 ~90s（模型 + RMBG 背景移除），常驻显存仅 **4.15GB**。

## 调用

```bash
# 健康检查
curl http://127.0.0.1:10332/health

# 图生 3D（上传图片 → GLB）
curl -X POST http://127.0.0.1:10332/generate \
  -F "image=@/path/to/img.png" \
  -F "steps=50" -F "seed=42" -F "guidance_scale=7.0" -F "faces=-1"
```

| 参数 | 默认 | 说明 |
|---|---|---|
| image | 必填 | 输入图片（自动去背景） |
| steps | 50 | 采样步数，30~50 |
| seed | 42 | 随机种子 |
| guidance_scale | 7.0 | CFG 强度 |
| faces | -1 | >0 时简化面数（quadric edge collapse） |

响应：`{"glb": "...", "seconds": 1520, "vertices": 667518, "faces": 1335036, ...}`

输出目录：`/mnt/data/ai_workspace/outputs3d/`（`triposg_<ts>_<seed>.glb`）

## 实测性能（单张 P40）

| 阶段 | 耗时 |
|---|---|
| 模型加载 | ~90s（一次性） |
| 50 步扩散推理 | **~7.5 分钟**（9s/步） |
| SDF→mesh 提取（505³） | **~18 分钟**（瓶颈，diso DiffDMC） |
| 总计 | ~26 分钟/图 |

显存峰值 ~9.6GB（推理阶段），常驻 4.15GB。

## 关键部署记录（踩坑）

### diso 编译（无预编译 wheel，必须现场编译）

系统环境：Ubuntu 24.04（gcc-13）+ CUDA 11.8（nvcc 只支持 gcc≤11）+ glibc 2.39。
解决方案（`/tmp/diso-0.1.4` 已 patch，编译参数写死进 setup.py）：

```bash
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc   # conda gcc-11（conda install --offline 本地 .conda 文件）
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export NVCC_FLAGS="-O3"
export TORCH_CUDA_ARCH_LIST="6.1"                        # P40 sm_61
pip install --no-build-isolation /tmp/diso-0.1.4/
```

setup.py patch 内容：
- `extra_compile_args["cxx"]` 加 `-I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 -I/usr/include/c++/13/x86_64-linux-gnu -I/usr/include/c++/13/backward -I/usr/include/x86_64-linux-gnu`（conda gcc 找不到系统 C++ 头）
- `extra_link_args`（**必须传给 CUDAExtension 对象，不是 setup()**）：`-B/usr/lib/gcc/x86_64-linux-gnu/13 -B/usr/lib/x86_64-linux-gnu -L/usr/lib/gcc/x86_64-linux-gnu/13 -L/usr/lib/x86_64-linux-gnu`（conda ld 找不到 crt/libm/libgcc）

### 其他

- conda gcc 安装：`conda install -y --offline ./gcc_linux-64-*.conda ./gxx_linux-64-*.conda ...`（直接指定 pkgs 缓存里的 .conda 文件，绕过网络）
- conda gcc 缺 sysroot：`ln -sfn / $CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot`（曾尝试，最终未用）
- 权重：`models/TripoSG`（7.5G）+ `inference/triposg/pretrained_weights/RMBG-1.4`（804M）
- 官方脚本会 `snapshot_download` 权重到 `pretrained_weights/`——用 symlink 指向本地模型避免重复下载

## 端口/GPU 分配

| 端口 | 服务 | GPU | 显存 |
|---|---|---|---|
| 10331 | SDXL 生图 | 0 | 13GB |
| 10332 | TripoSG 图生 3D | 1 | 4.2GB 常驻 / 9.6GB 峰值 |
| 10303 | vllm（gemma，未常驻） | 双卡 | - |
