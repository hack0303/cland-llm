#!/usr/bin/env python3
# ⛔ SDXL(:10331) 停用 —— 拒启守卫（stub）
# 背景：本机 15G 内存不足，SDXL 常驻会拖垮整机；口径=模型服务「按需启」，#302 扩容前不常驻。
# 出处（校正后）：由 atlas 于 2026-10-06 21:02 公告（msgId chat-1791291767502）+ R905/R912/R927 裁定；
#   Owner 是否认可待确认（已列入日终待决清单）—— 不把未证实的归属写死。
# 恢复：等 #302 扩容到位后，Owner 放行 → mv server.py.real server.py
# 说明：真身 = server.py.real（fp16 修复 d7d32fc）；本 stub 仅为阻止误拉起。
import sys
print("[SDXL-DISABLED] SDXL 已停用（资源治理 · 出处见文件头注释）；扩容 #302 到位且放行后再启。请勿拉起。", file=sys.stderr)
sys.exit(3)
