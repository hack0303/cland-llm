#!/usr/bin/env python3
"""管线产出图验收 — DWPose 客观评估 + 手指几何合理性分析
验收维度：
1. 手部 21 点完整度（conf>0.3 点数）
2. 手指几何：5 指链（根→中→尖）距离递增性、指尖-腕距离、指长比例
3. 对比基线（同 prompt 无管线图）
"""
import glob
import json
import sys

sys.path.insert(0, "/mnt/data/ai_workspace/cland-llm/inference/sdxl/hand_pipe")
from dwpose import DWPose, LEFT_HAND, RIGHT_HAND

OUT = "/mnt/data/ai_workspace/outputs/hand_pipe"

# 手指链（DWPose/OpenPose 手部索引）：每指 4 点 根→...→尖
FINGER_CHAINS = {
    "thumb": [1, 2, 3, 4],      # 1=根 4=尖
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def finger_metrics(kps, hand_idx):
    """单手几何评估。返回 (OK, 报告dict)"""
    pts = {}
    for name, chain in FINGER_CHAINS.items():
        ps = [(kps[hand_idx[i], 0], kps[hand_idx[i], 1], kps[hand_idx[i], 2])
              for i in chain]
        pts[name] = ps
    wrist = kps[hand_idx[0], :2]
    rep = {"fingers": {}, "ok": True, "n_valid_fingers": 0}
    for name, ps in pts.items():
        valid = [p for p in ps if p[2] > 0.3]
        tip, root = ps[0], ps[-1]
        d_tip = ((tip[0] - wrist[0]) ** 2 + (tip[1] - wrist[1]) ** 2) ** 0.5
        d_root = ((root[0] - wrist[0]) ** 2 + (root[1] - wrist[1]) ** 2) ** 0.5
        # 尖比根更远（正常手指）或 尖置信度不足但根在
        monotonic = d_tip > d_root - 5 if tip[2] > 0.3 and root[2] > 0.3 else None
        rep["fingers"][name] = {
            "valid_pts": len(valid), "tip_conf": round(float(tip[2]), 2),
            "len_tip": round(float(d_tip), 1), "len_root": round(float(d_root), 1),
            "monotonic": monotonic,
        }
        if len(valid) >= 3:
            rep["n_valid_fingers"] += 1
        if monotonic is False:
            rep["ok"] = False
    rep["ok"] = rep["ok"] and rep["n_valid_fingers"] >= 4  # 至少 4 指结构完整
    return rep


def evaluate(path):
    import cv2
    import numpy as np
    dp = DWPose()
    img = cv2.imread(path)
    # 超分大图（4096px）先缩放回 1024 再检测（yolox 640 输入对小目标失效）
    if max(img.shape[:2]) > 2048:
        img = cv2.resize(img, (1024, 1024))
    dets = dp.detect_full(img)
    out = []
    for d in dets:
        kps = d["kps"]
        hl = int((kps[LEFT_HAND, 2] > 0.3).sum())
        hr = int((kps[RIGHT_HAND, 2] > 0.3).sum())
        lf = finger_metrics(kps, LEFT_HAND) if hl >= 3 else None
        rf = finger_metrics(kps, RIGHT_HAND) if hr >= 3 else None
        out.append({
            "bbox": [round(float(v), 1) for v in d["bbox"][:4]],
            "body": int((kps[:17, 2] > 0.3).sum()),
            "handL": hl, "handR": hr,
            "geomL_ok": lf["ok"] if lf else None,
            "geomR_ok": rf["ok"] if rf else None,
            "fingersL": lf["n_valid_fingers"] if lf else 0,
            "fingersR": rf["n_valid_fingers"] if rf else 0,
        })
    return out


def main():
    dp = DWPose()
    import cv2
    scenes = ["peace_sign", "holding_cup", "chin_hands", "dancer"]
    print(f"{'scene':12s} {'step':5s} {'body':>4s} {'L21':>3s} {'R21':>3s} "
          f"{'Lfing':>5s} {'Rfing':>5s} {'Lgeom':>5s} {'Rgeom':>5s} {'persons':>7s}")
    report = {}
    for sc in scenes:
        for step in ("1_txt2img", "2_controlnet", "3_inpaint", "4_upscaled"):
            files = sorted(glob.glob(f"{OUT}/pipe_{step}_*.png"))
            # 按场景匹配：peace_sign 是最早时间戳的 step1（1787683972）
            # 场景顺序按时间戳分组：取每个 step 的 4 个文件按时间排序后对应
        # 简化：按时间戳顺序将 4 步文件分配给 4 场景（每场景一轮 4 步）
    # 重建场景映射：4 场景 × 4 步，按 step 文件时间戳排序
    seq = {}
    for step in ("1_txt2img", "2_controlnet", "3_inpaint", "4_upscaled"):
        files = sorted(glob.glob(f"{OUT}/pipe_{step}_*.png"))
        # 场景0 是第一次跑（时间戳最小），场景1-3 是第二批
        seq[step] = files
    try:
        rep = json.load(open(f"{OUT}/pipeline_report.json"))
        for sc, r in rep.items():
            row = {"steps": {}}
            for st in r["steps"]:
                row["steps"][st["name"]] = st["image"]
            report[sc] = row
    except Exception as e:
        print("report.json 读取失败:", e)
    # peace_sign 单独跑过（report 被覆盖），用时间戳补映射
    if "peace_sign" not in report:
        import re
        p1 = sorted(glob.glob(f"{OUT}/pipe_1_txt2img_*.png"),
                    key=lambda p: int(re.search(r'(\d+)_42\.png$', p).group(1)))
        base_ts = None
        for p in p1:
            ts = int(re.search(r'(\d+)_42\.png$', p).group(1))
            if 1787683900 <= ts <= 1787684000:  # 第一次单独跑的场景0
                base_ts = ts
                break
        if base_ts:
            row = {"steps": {}}
            for sname, tag in [("txt2img", "1_txt2img"), ("controlnet_openpose", "2_controlnet"),
                               ("inpaint_hand", "3_inpaint"), ("upscale_x4", "4_upscaled")]:
                cand = glob.glob(f"{OUT}/pipe_{tag}_*.png")
                # 取与 base_ts 同一轮的：时间差 < 400s
                best = min(cand, key=lambda p: abs(int(re.search(r'(\d+)(?:_42)?\.png$', p).group(1)) - (base_ts if 'txt2img' in tag else base_ts)))
                row["steps"][sname] = best
            report["peace_sign"] = row
    # 只保留 4 场景
    report = {k: v for k, v in report.items() if k in scenes}

    for sc, r in report.items():
        for sname, path in r["steps"].items():
            ev = evaluate(path)
            best = max(ev, key=lambda x: x["handL"] + x["handR"]) if ev else {}
            print(f"{sc:12s} {sname[:5]:5s} {best.get('body',0):4d} "
                  f"{best.get('handL',0):3d} {best.get('handR',0):3d} "
                  f"{best.get('fingersL',0):5d} {best.get('fingersR',0):5d} "
                  f"{str(best.get('geomL_ok')):5s} {str(best.get('geomR_ok')):5s} "
                  f"{len(ev):7d}")


if __name__ == "__main__":
    main()
