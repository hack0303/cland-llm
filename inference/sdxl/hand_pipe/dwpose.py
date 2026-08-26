#!/usr/bin/env python3
"""DWPose 全身姿态检测（133 点 COCO-WholeBody）+ 骨架绘制 + 手部区域提取
- yolox_l.onnx: 人体检测 (NMS)
- dw-ll_ucoco_384.onnx: 133 点关键点 (body 17 + face 68 + hands 21x2)
纯 onnxruntime CPU 推理，P40 不占用 GPU。
"""
import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = "/mnt/data/ai_workspace/models/dwpose"
BODY_CONNS = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
              (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
FACE_CONNS = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),
              (10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),
              (18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),
              (26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),
              (34,35),(35,36),(36,37),(37,38),(38,39),(39,40),(40,41),(41,42),
              (42,43),(43,44),(44,45),(45,46),(46,47),(47,48),(48,49),(49,50),
              (50,51),(51,52),(52,53),(53,54),(54,55),(55,56),(56,57),(57,58),
              (58,59),(59,60),(60,61),(61,62),(62,63),(63,64),(64,65),(65,66),(66,67)]
HAND_CONNS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
              (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),
              (18,19),(19,20)]
# 关键点索引: 0-16 body, 17-22 左手(handL), 23-24 脸边缘?, 25-90 face, 91-96 右手? 
# COCO-WholeBody: 0-16 body, 17-22 left hand, 23-90 face, 91-96 right hand? 实际:
# 0-16 body(17), 17-22 left hand(6), 23-90 face(68), 91-96 right hand(6)?? 错误。
# 官方: left_hand 17-22 是 6 个腕点, face 23-90, right_hand 91-96 腕点,
# 手部 21 点: 左手 92-112, 右手 113-133 (索引 0-based: 91-111, 112-132)
LEFT_HAND = list(range(91, 112))   # 21 点
RIGHT_HAND = list(range(112, 133)) # 21 点


class DWPose:
    def __init__(self, model_dir=MODEL_DIR):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.det = ort.InferenceSession(f"{model_dir}/yolox_l.onnx", so,
                                        providers=["CPUExecutionProvider"])
        self.pose = ort.InferenceSession(f"{model_dir}/dw-ll_ucoco_384.onnx", so,
                                         providers=["CPUExecutionProvider"])
        self.det_in = self.det.get_inputs()[0]
        self.pose_in = self.pose.get_inputs()[0]

    # ---------- 人体检测 (yolox, 标准 grid+stride decode) ----------
    @staticmethod
    def _yolox_decode(outputs, img_size=(640, 640)):
        grids, strides = [], []
        for stride in (8, 16, 32):
            hsize, wsize = img_size[0] // stride, img_size[1] // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            strides.append(np.full((*grid.shape[:2], 1), stride))
        grids = np.concatenate(grids, 1)
        strides = np.concatenate(strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides
        return outputs

    @staticmethod
    def _nms(boxes, scores, thr):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
            order = order[np.where(ovr <= thr)[0] + 1]
        return keep

    def detect_person(self, img_bgr):
        h, w = img_bgr.shape[:2]
        size = self.det_in.shape[2]  # 640
        # letterbox 预处理
        r = min(size / h, size / w)
        resized = cv2.resize(img_bgr, (int(w * r), int(h * r)))
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[:int(h * r), :int(w * r)] = resized
        inp = canvas.astype(np.float32).transpose(2, 0, 1)[None]
        out = self.det.run(None, {self.det_in.name: inp})[0]
        pred = self._yolox_decode(out)[0]  # (8400,85)
        boxes = pred[:, :4]
        scores = pred[:, 4:5] * pred[:, 5:]  # obj * class80
        x1y1x2y2 = np.ones_like(boxes)
        x1y1x2y2[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        x1y1x2y2[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        x1y1x2y2[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        x1y1x2y2[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        x1y1x2y2 /= r  # 还原原图尺度
        # 只取 person 类 (class 0)
        cls_scores = scores[:, 0]
        valid = cls_scores > 0.3
        if valid.sum() == 0:
            return []
        vb = x1y1x2y2[valid]
        vs = cls_scores[valid]
        keep = self._nms(vb, vs, 0.45)
        res = []
        for i in keep:
            x1, y1, x2, y2 = vb[i]
            res.append((x1, y1, x2, y2, float(vs[i])))
        return res

    # ---------- 关键点 (RTMPose SimCC, 标准 top-down 仿射) ----------
    @staticmethod
    def _rotate_point(pt, angle_rad):
        x, y = pt
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        return np.array([x * cs - y * sn, x * sn + y * cs], dtype=np.float32)

    @staticmethod
    def _get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    @classmethod
    def _get_affine(cls, center, scale, output_size, rot=0.0):
        """mmpose get_warp_matrix（无 200px 参考系因子，带宽高比修正）"""
        dst_w, dst_h = output_size
        # 宽高比修正
        w, h = scale
        if w > h * (dst_w / dst_h):
            scale = np.array([w, w * dst_h / dst_w])
        else:
            scale = np.array([h * dst_w / dst_h, h])
        src_w = scale[0]
        rot_rad = np.deg2rad(rot)
        src_dir = cls._rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])
        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        src[2:, :] = cls._get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = cls._get_3rd_point(dst[0, :], dst[1, :])
        return cv2.getAffineTransform(np.float32(src), np.float32(dst))

    def keypoints(self, img_bgr, person_xyxy):
        x1, y1, x2, y2 = person_xyxy
        pw, ph = 288, 384  # 模型输入 (w, h)
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        scale = np.array([x2 - x1, y2 - y1], dtype=np.float32) * 1.25
        # 仿射变换（无黑边，标准 top-down）
        trans = self._get_affine(center, scale, (pw, ph))
        warped = cv2.warpAffine(img_bgr, trans, (pw, ph),
                                flags=cv2.INTER_LINEAR)
        # ImageNet 归一化 (RGB)
        img = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img = (img - mean) / std
        inp = img.transpose(2, 0, 1).astype(np.float32)[None]
        simcc_x, simcc_y = self.pose.run(None, {self.pose_in.name: inp})
        sx, sy = simcc_x[0], simcc_y[0]  # (133, Bx) (133, By)
        # 标准 SimCC 解码: argmax + 均值置信度 + split_ratio=2
        x_locs = np.argmax(sx, axis=1)
        y_locs = np.argmax(sy, axis=1)
        vals = (sx.max(axis=1) + sy.max(axis=1)) * 0.5
        kps = np.stack([x_locs, y_locs], axis=-1).astype(np.float32) / 2.0
        # 映射回原图: kps / input_size * scale + center - scale/2
        kps = kps / np.array([pw, ph]) * scale + center - scale / 2
        kps = np.concatenate([kps, vals[:, None]], axis=1)
        return kps

    def detect_full(self, img_bgr):
        """返回 [{bbox, kps(133,3), hands:[l,r]}]"""
        persons = self.detect_person(img_bgr)
        out = []
        for b in persons:
            kps = self.keypoints(img_bgr, b[:4])
            if kps is None:
                continue
            lh = kps[LEFT_HAND]
            rh = kps[RIGHT_HAND]
            out.append({"bbox": b, "kps": kps, "left_hand": lh, "right_hand": rh})
        return out

    # ---------- 绘制 ----------
    @staticmethod
    def draw_skeleton(img_bgr, kps, scale=1.0):
        canvas = np.zeros_like(img_bgr)
        h, w = img_bgr.shape[:2]
        def pt(i):
            x, y, c = kps[i]
            if c < 0.1:
                return None
            return int(x), int(y)
        # 手部 (红/蓝)
        for hand, color in ((LEFT_HAND, (0, 0, 255)), (RIGHT_HAND, (255, 0, 0))):
            for a, b in HAND_CONNS:
                pa, pb = pt(hand[a]), pt(hand[b])
                if pa and pb:
                    cv2.line(canvas, pa, pb, color, 2)
            for i in hand:
                p = pt(i)
                if p:
                    cv2.circle(canvas, p, 3, color, -1)
        # 身体 (绿)
        for a, b in BODY_CONNS:
            pa, pb = pt(a), pt(b)
            if pa and pb:
                cv2.line(canvas, pa, pb, (0, 255, 0), 3)
        for i in range(17):
            p = pt(i)
            if p:
                cv2.circle(canvas, p, 4, (0, 255, 0), -1)
        # 脸 (白, 稀疏)
        for a, b in FACE_CONNS[::4]:
            pa, pb = pt(23 + a), pt(23 + b)
            if pa and pb:
                cv2.line(canvas, pa, pb, (200, 200, 200), 1)
        return canvas

    @staticmethod
    def hand_boxes(kps, img_shape, expand=0.35, min_side=96):
        """从关键点提取手部 bbox（用于 inpaint mask），返回 [(x1,y1,x2,y2)]"""
        h, w = img_shape
        boxes = []
        for hand in (LEFT_HAND, RIGHT_HAND):
            pts = [(kps[i, 0], kps[i, 1]) for i in hand if kps[i, 2] > 0.2]
            if len(pts) < 3:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            bw, bh = x2 - x1, y2 - y1
            if bw < 8 or bh < 8:  # 太小视为无效
                continue
            ex, ey = bw * expand, bh * expand
            x1, y1 = max(0, x1 - ex), max(0, y1 - ey)
            x2, y2 = min(w, x2 + ex), min(h, y2 + ey)
            if (x2 - x1) < min_side or (y2 - y1) < min_side:  # 保底尺寸
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                half = max(min_side / 2, (x2 - x1) / 2, (y2 - y1) / 2)
                x1, y1 = max(0, cx - half), max(0, cy - half)
                x2, y2 = min(w, cx + half), min(h, cy + half)
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
        return boxes


if __name__ == "__main__":
    import sys
    dp = DWPose()
    img = cv2.imread(sys.argv[1])
    res = dp.detect_full(img)
    print(f"{len(res)} person(s)")
    for r in res:
        print("bbox:", [round(v, 1) for v in r["bbox"]])
        print("hands:", dp.hand_boxes(r["kps"], img.shape[:2]))
        canvas = dp.draw_skeleton(img, r["kps"])
        out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/skeleton.png"
        cv2.imwrite(out, canvas)
        print("skeleton ->", out)
