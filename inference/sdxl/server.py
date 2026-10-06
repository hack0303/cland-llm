#!/usr/bin/env python3
# ⛔ 已停用（Owner 2026-10-06）：本机 15G 内存不足，SDXL 常驻会拖垮整机。
#    口径：模型服务「按需启」，扩容(#302)到位前不常驻/不乱拉。
#    恢复：扩容后由 Owner 放行 → mv server.py.real server.py
import sys
print("[SDXL-DISABLED] SDXL 已按 Owner 口径停用（资源治理）；扩容 #302 到位后再启。请勿拉起。", file=sys.stderr)
sys.exit(3)
