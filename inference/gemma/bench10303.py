#!/usr/bin/env python3
"""gemma4 (10303) 吞吐/并发实测脚本 —— 复测基线用。

跑法: 先确保 10303 已就绪, 然后: python3 bench10303.py

A. 短请求校准 tok/s
B. batch=10 画像模拟 (10 人一条 prompt -> 10 份画像) -> 推算 7300 人全量时长
C. 并发 1/2/4 路总吞吐 (验证单序列串行: 预期聚合吞吐不随 N 涨)

2026-09-06 实测结论 (单卡 GPU1, 详见 skill cland-llm-text §7):
  decode 稳态 ~43-48 tok/s; batch=10 画像 ~10s/请求; 7300 人全量 ≈ 2.0-2.6h;
  并发 1/2/4 无收益 (严格串行, 服务端日志可证: ENTER 恒在上一 DONE 后)。
"""
import json, time, urllib.request, threading, sys

URL = "http://127.0.0.1:10303/v1/chat/completions"

def call(prompt, max_tokens=512, timeout=900):
    body = json.dumps({"model": "gemma4",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.4}).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    dt = time.time() - t0
    u = resp["usage"]
    return dt, u["prompt_tokens"], u["completion_tokens"]

def show(tag, dt, p, c):
    print(f"[{tag}] wall={dt:6.1f}s  prompt={p:5d}t  completion={c:5d}t  "
          f"decode={c/dt:6.1f} tok/s  总={ (p+c)/dt:6.1f} tok/s", flush=True)

# ---------- A. 短请求校准 ----------
print("== A. 短请求校准 ==", flush=True)
dt, p, c = call("用一句话介绍你自己。", max_tokens=128, timeout=300)
show("A", dt, p, c)

# ---------- B. batch=10 画像模拟 ----------
print("\n== B. batch=10 画像（10 人一条 prompt）==", flush=True)
records = []
for i in range(1, 11):
    records.append(
        f"用户{i}: 男{i%2==0 and '女' or '男'}，{20+i}岁，{['程序员','设计师','学生','销售','教师'][i%5]}，"
        f"最近浏览:{['数码产品','运动装备','书籍','美妆','家居'][i%5]}，近30天下单{i*3}单均价{50+i*10}元，"
        f"常用时段{['晚上','中午','凌晨','下午','早晨'][i%5]}")
prompt = ("你是电商用户画像分析师。以下是 10 位用户的行为记录：\n" +
          "\n".join(records) +
          "\n\n请对每位用户输出一份用户画像，包含：身份特征、兴趣偏好、消费习惯、一句话运营建议。"
          "严格按 '用户N：' 编号逐条输出，每人 40-60 字，不要输出其他内容。")
dt, p, c = call(prompt, max_tokens=1800, timeout=900)
show("B", dt, p, c)
if c > 0:
    per_req = dt
    n_req = 7300 / 10
    print(f"→ 推算 7300 人（batch=10 → {int(n_req)} 个请求）：{per_req*n_req/3600:.1f} 小时（串行单请求耗时不变）", flush=True)

# ---------- C. 并发 1/2/4 路 ----------
print("\n== C. 并发 1/2/4 路总吞吐 ==", flush=True)
PROMPT_C = "请写一篇主题为「城市公园的早晨」的中文短文，约 500 字，内容充实、语句通顺，不要提前结束。"
for N in (1, 2, 4):
    out = [None] * N
    def worker(i):
        try:
            out[i] = call(PROMPT_C, max_tokens=800, timeout=1200)
        except Exception as e:
            out[i] = ("ERR", str(e))
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    ok = [o for o in out if not (isinstance(o, tuple) and o[0] == "ERR")]
    errs = [o for o in out if isinstance(o, tuple) and o[0] == "ERR"]
    tot_p = sum(o[1] for o in ok); tot_c = sum(o[2] for o in ok)
    dts = [o[0] for o in ok]
    print(f"N={N}: 总墙钟={wall:6.1f}s  成功={len(ok)}/{N}  总输出={tot_c}t  "
          f"聚合吞吐={tot_c/wall:6.1f} tok/s  单请求耗时={['%.1f'%d for d in dts]}", flush=True)
    if errs: print(f"    错误: {errs}", flush=True)

print("\nDONE", flush=True)
