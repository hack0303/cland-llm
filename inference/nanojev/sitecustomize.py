"""C-Land 兼容 shim：为 torch 2.7.1+cu118（P40 / sm_61）补齐上游脚本引用的 torch._native.triton_utils。

背景：NanoJev 记录依赖 torch==2.14.0，其中
`from torch._native import triton_utils; triton_utils.deregister_op_overrides()`
用于关闭 native Triton op overrides（走 ATen fallback）。
本机 P40 只能用 torch 2.7.1+cu118（torch≥2.9 已砍 sm_70 以下），该模块不存在。
torch 2.7.1 没有需要关闭的 native Triton overrides，故提供 no-op 等价实现。

加载方式：运行时 PYTHONPATH 包含本目录，Python 通过 sitecustomize 自动加载。
仅在 `torch._native.triton_utils` 确实缺失时注册 stub；未来 torch 若自带则不受影响。
"""
import importlib
import sys
import types

try:
    import torch  # noqa: F401
except Exception:  # torch 不可用时不做任何事
    torch = None

if torch is not None and "torch._native" not in sys.modules:
    try:
        importlib.import_module("torch._native.triton_utils")
    except Exception:
        _native = types.ModuleType("torch._native")
        _triton_utils = types.ModuleType("torch._native.triton_utils")

        def deregister_op_overrides():
            """torch 2.7 无 native Triton op overrides，no-op。"""
            return None

        _triton_utils.deregister_op_overrides = deregister_op_overrides
        _native.triton_utils = _triton_utils
        sys.modules["torch._native"] = _native
        sys.modules["torch._native.triton_utils"] = _triton_utils
        torch._native = _native
