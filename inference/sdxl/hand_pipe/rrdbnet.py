#!/usr/bin/env python3
"""Real-ESRGAN 超分（RRDBNet 正确架构，纯 torch 实现）
RRDB = ResidualDenseBlock_5C ×3 堆叠（RDB1/RDB2/RDB3），与官方/ComfyUI 权重格式一致
兼容权重命名：官方 Real-ESRGAN (0.weight) / ComfyUI (model.0.weight)
"""
import torch
import torch.nn as nn
import numpy as np
import cv2


class ResidualDenseBlock_5C(nn.Module):
    """5-conv 稠密残差块（RDB1/2/3 的基本单元）"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block：RDB1/2/3 堆叠 + 0.2 残差"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock_5C(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch)
                                    for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(
            nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(
            nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def pixel_unshuffle(x, scale):
    b, c, hh, ww = x.shape
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and ww % scale == 0
    return x.reshape(b, c, hh // scale, scale, ww // scale, scale) \
            .permute(0, 1, 3, 5, 2, 4) \
            .reshape(b, out_channel, hh // scale, ww // scale)


class RealESRGANUpscaler:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        params = ckpt.get("params", ckpt.get("params_ema", ckpt))
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                             num_block=23, num_grow_ch=32, scale=4)
        state = self._adapt_keys(params)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        n_miss_body = len([m for m in missing if m.startswith(("conv_", "body"))])
        if n_miss_body > 10 or len(unexpected) > 10:
            raise RuntimeError(
                f"权重加载失败: missing={len(missing)} unexpected={len(unexpected)} "
                f"(模型格式不匹配，检查 pth 是否为 RRDBNet x4)")
        self.model.to(device).eval()

    @staticmethod
    def _adapt_keys(sd):
        """适配权重命名 → 本实现命名（官方小写 rdb）
        官方 Real-ESRGAN: conv_first / body.0.rdb1.conv1.weight / conv_body / conv_up1 ... （直通）
        ComfyUI: model.0.weight; model.1.sub.{i}.RDB{j}.conv{k}.0.weight → body.{i}.rdb{j}.conv{k}.weight;
                 model.1.sub.23.weight → conv_body; model.2→conv_up1 ... 
        """
        out = {}
        for k, v in sd.items():
            parts = k.split(".")
            if parts[0] == "model":
                parts = parts[1:]
            top = parts[0]
            if top == "0":
                nk = "conv_first." + ".".join(parts[1:])
            elif top == "1" and len(parts) > 1 and parts[1] == "sub":
                if len(parts) > 3 and parts[3].startswith("RDB"):
                    # 1.sub.{i}.RDB{j}.conv{k}.0.weight → body.{i}.rdb{j}.conv{k}.weight
                    i = int(parts[2])
                    j = parts[3].lower()          # RDB1 → rdb1
                    nk = f"body.{i}.{j}.{parts[4]}.{parts[6]}"
                else:
                    # 1.sub.{i}.weight → body 序列尾部 conv（i==23 为 conv_body）
                    nk = "conv_body." + ".".join(parts[3:])
            elif top == "3":
                nk = "conv_up1." + ".".join(parts[1:])
            elif top == "6":
                nk = "conv_up2." + ".".join(parts[1:])
            elif top == "8":
                nk = "conv_hr." + ".".join(parts[1:])
            elif top == "10":
                nk = "conv_last." + ".".join(parts[1:])
            else:
                nk = k  # 官方格式直通（conv_first/body.0.rdb1.../conv_body/conv_up1...）
            out[nk] = v
        return out

    def upscale(self, img_bgr, out_scale=4):
        """img_bgr: HxWx3 uint8 BGR → 放大图"""
        img = img_bgr.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))[None].to(self.device)
        with torch.no_grad():
            out = self.model(img)
        out = out.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255.0
        return out.astype(np.uint8)


if __name__ == "__main__":
    import sys
    up = RealESRGANUpscaler(sys.argv[1] if len(sys.argv) > 1
                            else "/mnt/data/ai_workspace/models/upscale/4x-UltraSharp.pth")
    img = cv2.imread(sys.argv[2])
    out = up.upscale(img)
    dst = sys.argv[3] if len(sys.argv) > 3 else "/tmp/upscaled.png"
    cv2.imwrite(dst, out)
    print(f"{img.shape} -> {out.shape} -> {dst} (mean={out.mean():.1f})")
