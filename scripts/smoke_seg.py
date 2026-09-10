#!/usr/bin/env python3
"""T3 SEG smoke: instantiate UNet/UNet++ from notebook, forward dummy batch, load local weights if available."""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── UNet (exact notebook code, lines 176-250) ──
class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        self.enc1 = self.double_conv(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = self.double_conv(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec6 = self.double_conv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec7 = self.double_conv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec8 = self.double_conv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec9 = self.double_conv(128, 64)
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c1 = self.enc1(x);  p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        c4 = self.enc4(p3); p4 = self.pool4(c4)
        c5 = self.dropout(self.bottleneck(p4))
        u6 = torch.cat([self.up6(c5), c4], 1); c6 = self.dec6(u6)
        u7 = torch.cat([self.up7(c6), c3], 1); c7 = self.dec7(u7)
        u8 = torch.cat([self.up8(c7), c2], 1); c8 = self.dec8(u8)
        u9 = torch.cat([self.up9(c8), c1], 1); c9 = self.dec9(u9)
        return torch.sigmoid(self.final(c9))

# ── UNetPlusPlus (exact notebook code, lines 401-495) ──
class UNetPlusPlus(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        nb = [64, 128, 256, 512, 1024]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.5)
        self.conv0_0 = self.double_conv(input_channels, nb[0])
        self.conv1_0 = self.double_conv(nb[0], nb[1])
        self.conv2_0 = self.double_conv(nb[1], nb[2])
        self.conv3_0 = self.double_conv(nb[2], nb[3])
        self.conv4_0 = self.double_conv(nb[3], nb[4])
        self.conv0_1 = self.double_conv(nb[0]+nb[1], nb[0])
        self.conv1_1 = self.double_conv(nb[1]+nb[2], nb[1])
        self.conv2_1 = self.double_conv(nb[2]+nb[3], nb[2])
        self.conv3_1 = self.double_conv(nb[3]+nb[4], nb[3])
        self.conv0_2 = self.double_conv(nb[0]*2+nb[1], nb[0])
        self.conv1_2 = self.double_conv(nb[1]*2+nb[2], nb[1])
        self.conv2_2 = self.double_conv(nb[2]*2+nb[3], nb[2])
        self.conv0_3 = self.double_conv(nb[0]*3+nb[1], nb[0])
        self.conv1_3 = self.double_conv(nb[1]*3+nb[2], nb[1])
        self.conv0_4 = self.double_conv(nb[0]*4+nb[1], nb[0])
        if deep_supervision:
            self.final1 = nn.Conv2d(nb[0], output_channels, 1)
            self.final2 = nn.Conv2d(nb[0], output_channels, 1)
            self.final3 = nn.Conv2d(nb[0], output_channels, 1)
            self.final4 = nn.Conv2d(nb[0], output_channels, 1)
        else:
            self.final = nn.Conv2d(nb[0], output_channels, 1)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x4_0 = self.dropout(self.conv4_0(self.pool(x3_0)))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        if self.deep_supervision:
            return (torch.sigmoid(self.final1(x0_1)) + torch.sigmoid(self.final2(x0_2)) +
                    torch.sigmoid(self.final3(x0_3)) + torch.sigmoid(self.final4(x0_4))) / 4
        return torch.sigmoid(self.final(x0_4))

# ── Smoke tests ──
B, H, W = 2, 224, 224
dummy = torch.randn(B, 3, H, W).to(device)

print("\n=== UNet forward ===")
unet = UNet(3, 1).to(device)
unet.eval()
with torch.no_grad():
    out_unet = unet(dummy)
print(f"Output shape: {out_unet.shape}")
assert out_unet.shape == (B, 1, H, W), f"FAIL: expected (2,1,224,224), got {out_unet.shape}"
print(f"Value range: [{out_unet.min().item():.4f}, {out_unet.max().item():.4f}] (sigmoid)")
print(f"Mean pixel: {out_unet.mean().item():.4f}")
n_params_unet = sum(p.numel() for p in unet.parameters())
print(f"Parameters: {n_params_unet:,}")

print("\n=== UNetPlusPlus forward ===")
unetpp = UNetPlusPlus(3, 1, deep_supervision=False).to(device)
unetpp.eval()
with torch.no_grad():
    out_pp = unetpp(dummy)
print(f"Output shape: {out_pp.shape}")
assert out_pp.shape == (B, 1, H, W), f"FAIL: expected (2,1,224,224), got {out_pp.shape}"
n_params_pp = sum(p.numel() for p in unetpp.parameters())
print(f"Parameters: {n_params_pp:,}")

# Deep supervision mode
unetpp_ds = UNetPlusPlus(3, 1, deep_supervision=True).to(device)
unetpp_ds.eval()
with torch.no_grad():
    out_ds = unetpp_ds(dummy)
print(f"Deep supervision output shape: {out_ds.shape}")

# ── Load local weights if available ──
weight_path = os.path.expanduser("~/Downloads/segmentation/seg_model_unet.pth")
if os.path.exists(weight_path):
    print(f"\n=== Loading local weights: {weight_path} ({os.path.getsize(weight_path)/(1024*1024):.1f} MB) ===")
    unet_loaded = UNet(3, 1).to(device)
    state = torch.load(weight_path, map_location=device)
    unet_loaded.load_state_dict(state)
    unet_loaded.eval()
    with torch.no_grad():
        out_loaded = unet_loaded(dummy)
    print(f"Loaded UNet forward shape: {out_loaded.shape}")
    assert out_loaded.shape == (B, 1, H, W)
    print(f"Loaded model prediction range: [{out_loaded.min().item():.4f}, {out_loaded.max().item():.4f}]")
    print(f"Loaded model mean prediction: {out_loaded.mean().item():.4f}")
    # Threshold test
    binary = (out_loaded > 0.5).float()
    road_pct = binary.mean().item() * 100
    print(f"Binary threshold(0.5) road pixels: {road_pct:.2f}%")
    print("WEIGHT_LOAD_OK")
else:
    print(f"\n=== Local weights not found at {weight_path}, skipping weight load ===")

# ── IoU calculation smoke (from notebook calculate_iou_score) ──
import numpy as np
gt_mask = np.random.randint(0, 2, (H, W)).astype(np.uint8)
pred_mask = (out_unet[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
intersection = np.logical_and(gt_mask, pred_mask).sum()
union = np.logical_or(gt_mask, pred_mask).sum()
iou = intersection / (union + 1e-7)
print(f"\n=== IoU smoke (random GT vs UNet pred) ===")
print(f"IoU: {iou:.6f}")
print(f"Intersection pixels: {int(intersection)}, Union pixels: {int(union)}")

print("\n=== ALL SMOKE TESTS PASSED ===")
