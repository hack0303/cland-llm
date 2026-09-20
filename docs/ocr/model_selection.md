# OCR 模型选型（C-Land 本机）

> 2026-08-31 定稿。部署详情见 `inference/ocr/server.py` 与技能 `cland-llm-ocr`。

## 一、结论

| 方案 | 状态 | 说明 |
|---|---|---|
| **PaddleOCR 2.7.3（PP-OCRv4 中英）** | ✅ **生产（10337）** | GPU 推理 ~0.1s，中文精度高，服务化完成 |
| EasyOCR | 📌 备选 | 本机已有 CLI（`~/miniconda3/bin/easyocr`），模型小，未服务化 |
| RapidOCR | 📌 备选 | onnxruntime 轻量，免 paddle 依赖，未部署 |
| paddleocr 3.x（paddlex 3.7） | ❌ 不可用 | 需 paddle 3.x + 新版 cudnn，P40 链路未验证 |

## 二、为什么选 PaddleOCR 2.7.3

1. **中文识别精度最高**：PP-OCRv4 中英模型（det+rec+cls 共 ~18MB）在票据/截图场景实测
   `订单号：20260830-001` 0.9985 / 长中文行 0.9991，远优于 EasyOCR 中文表现
2. **轻量**：GPU 显存仅 **240MiB**（P40 24GB 余量极大），RAM <1GB，可与其他服务共存
3. **P40 兼容**：paddlepaddle-gpu 2.6.1（cu118）自带 sm_61 内核，实测 `run_check` 通过
4. **方向矫正内置**：`use_angle_cls=True`，倒置/旋转文字自动矫正

## 三、版本铁律（踩坑结论）

| 依赖 | 版本 | 原因 |
|---|---|---|
| paddlepaddle-gpu | **2.6.1**（cu118 源） | paddle 3.x 的 P40/cuDNN 链路未验证 |
| paddleocr | **2.7.3** | 3.x 依赖 paddlex 3.7 → 调 `AnalysisConfig.set_optimization_level`（paddle 3 API），2.6 无此方法 |
| numpy | **1.26.4**（<2） | numpy 2.x 与 paddle 2.6 的 `__array__` 协议不兼容 |
| cudnn | nvidia-cudnn-cu11 8.9.6.50 | paddle 2.6 找 **libcudnn.so（无版本号）**，需软链 `paddle/libs/libcudnn.so → libcudnn.so.8` |
| 环境 | `ocr_env`（python 3.10） | 独立环境，避免污染 audio_env/gemma_env |

## 四、部署要点

- 服务：FastAPI，端口 **10337**，端点 `/ocr`（文件上传）、`/ocr/base64`、`/health`
- 启动：`export LD_LIBRARY_PATH=.../paddle/libs`（cudnn 软链目录）后 `nohup python server.py`，~15s 就绪
- 模型：首次实例化自动下载到 `~/.paddleocr/`（无需手动下），离线可用
- 响应：`lines[] = {text, score, box(4 点坐标)}` + `elapsed`

## 五、端口分配总表（更新）

| 端口 | 服务 | GPU |
|---|---|---|
| 10303 | gemma4 文本 LLM（按需起） | 双卡 |
| 10331 | SDXL 生图 | 0 |
| 10332 | TripoSG 图生 3D | 1 |
| 10333/10334 | TTS/ASR（语音） | 1 |
| 10335/10336 | 音乐/音效 | 1 |
| **10337** | **PaddleOCR 文字识别** | 0 |
