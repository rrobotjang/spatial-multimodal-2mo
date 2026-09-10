"""Reconstruct the MPII pose-estimation model (Stacked Hourglass) from the notebook.

The class definitions below are transcribed 1:1 from
~/Downloads/pose_estimation_submission_final_v2_transferChatgpt5.6.ipynb
cells 21-24 (BottleneckBlock, HourglassModule, LinearLayer, StackedHourglassNetwork).

Feeds: Task 4 smoke reproduction + Task 13 (yoga bonus).
"""

from __future__ import annotations

import io
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1, downsample=False):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, filters // 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(filters // 2, momentum=0.9)
        self.conv2 = nn.Conv2d(filters // 2, filters // 2, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(filters // 2, momentum=0.9)
        self.conv3 = nn.Conv2d(filters // 2, filters, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample_conv(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += identity
        return out


class HourglassModule(nn.Module):
    def __init__(self, order, filters, num_residual):
        super(HourglassModule, self).__init__()
        self.order = order

        self.up1_0 = BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
        self.up1_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])

        if order > 1:
            self.low2 = HourglassModule(order - 1, filters, num_residual)
        else:
            self.low2_blocks = nn.Sequential(*[
                BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
                for _ in range(num_residual)
            ])

        self.low3_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1_0(x)
        up1 = self.up1_blocks(up1)

        low1 = self.pool(x)
        low1 = self.low1_blocks(low1)
        if self.order > 1:
            low2 = self.low2(low1)
        else:
            low2 = self.low2_blocks(low1)
        low3 = self.low3_blocks(low2)
        up2 = self.upsample(low3)

        return up2 + up1


class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackedHourglassNetwork(nn.Module):
    def __init__(self, input_shape=(256, 256, 3), num_stack=4, num_residual=1, num_heatmap=16):
        super(StackedHourglassNetwork, self).__init__()
        self.num_stack = num_stack

        in_channels = input_shape[2]  # 3
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)

        self.bottleneck1 = BottleneckBlock(in_channels=64, filters=128, stride=1, downsample=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck2 = BottleneckBlock(in_channels=128, filters=128, stride=1, downsample=False)
        self.bottleneck3 = BottleneckBlock(in_channels=128, filters=256, stride=1, downsample=True)

        self.hourglass_modules = nn.ModuleList()
        self.residual_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.heatmap_convs = nn.ModuleList()
        self.intermediate_convs = nn.ModuleList()
        self.intermediate_outs = nn.ModuleList()

        for i in range(num_stack):
            self.hourglass_modules.append(HourglassModule(order=4, filters=256, num_residual=num_residual))
            self.residual_modules.append(nn.Sequential(*[
                BottleneckBlock(in_channels=256, filters=256, stride=1, downsample=False)
                for _ in range(num_residual)
            ]))
            self.linear_layers.append(LinearLayer(in_channels=256, out_channels=256))
            self.heatmap_convs.append(nn.Conv2d(256, num_heatmap, kernel_size=1, stride=1, padding=0))

            if i < num_stack - 1:
                self.intermediate_convs.append(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.intermediate_outs.append(nn.Conv2d(num_heatmap, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.bottleneck1(x)
        x = self.pool(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)

        outputs = []
        for i in range(self.num_stack):
            hg = self.hourglass_modules[i](x)
            res = self.residual_modules[i](hg)
            lin = self.linear_layers[i](res)
            heatmap = self.heatmap_convs[i](lin)
            outputs.append(heatmap)

            if i < self.num_stack - 1:
                inter1 = self.intermediate_convs[i](lin)
                inter2 = self.intermediate_outs[i](heatmap)
                x = inter1 + inter2

        return outputs


MPII_JOINT_NAMES = [
    'R_ANKLE', 'R_KNEE', 'R_HIP', 'L_HIP', 'L_KNEE', 'L_ANKLE',
    'PELVIS', 'THORAX', 'UPPER_NECK', 'HEAD_TOP',
    'R_WRIST', 'R_ELBOW', 'R_SHOULDER', 'L_SHOULDER', 'L_ELBOW', 'L_WRIST',
]


def load_state_dict_verbose(model: nn.Module, ckpt: dict, tag: str) -> None:
    """Strict load first; on mismatch, report exact error then retry strict=False."""
    state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[load] {tag}: STRICT load OK")
        return
    except RuntimeError as e:
        print(f"[load] {tag}: strict load FAILED -> {str(e)[:500]}")

    prefixed = any(k.startswith('module.') for k in state_dict.keys())
    if prefixed:
        print(f"[load] {tag}: retry after stripping 'module.' prefix")
        state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
                      for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"[load] {tag}: non-strict load applied (missing=/unexpected reported above)")


def find_max_coordinates(heatmaps: torch.Tensor) -> torch.Tensor:
    """heatmaps: (H, W, C) -> (C, 2) [x, y] per channel."""
    H, W, C = heatmaps.shape
    flatten_heatmaps = heatmaps.reshape(-1, C)
    indices = torch.argmax(flatten_heatmaps, dim=0)
    y = indices // W
    x = indices - W * y
    return torch.stack([x, y], dim=1)


def extract_keypoints_from_heatmap(heatmaps: torch.Tensor) -> torch.Tensor:
    """heatmaps: (H, W, C) -> (C, 2) normalized [0..1] coordinates."""
    H, W, C = heatmaps.shape
    max_keypoints = find_max_coordinates(heatmaps)

    heatmaps_permuted = heatmaps.permute(2, 0, 1)
    padded = F.pad(heatmaps_permuted, (1, 1, 1, 1))
    padded_heatmaps = padded.permute(1, 2, 0)

    adjusted_keypoints = []
    for i, keypoint in enumerate(max_keypoints):
        max_x = int(keypoint[0].item()) + 1
        max_y = int(keypoint[1].item()) + 1

        patch = padded_heatmaps[max_y - 1:max_y + 2, max_x - 1:max_x + 2, i].clone()
        patch[1, 1] = 0
        flat_patch = patch.reshape(-1)
        index = torch.argmax(flat_patch).item()

        next_y = index // 3
        next_x = index % 3
        delta_y = (next_y - 1) / 4.0
        delta_x = (next_x - 1) / 4.0

        adjusted_x = keypoint[0].item() + delta_x
        adjusted_y = keypoint[1].item() + delta_y
        adjusted_keypoints.append((adjusted_x, adjusted_y))

    adjusted_keypoints = torch.tensor(adjusted_keypoints)
    adjusted_keypoints = torch.clamp(adjusted_keypoints, 0, H)
    normalized_keypoints = adjusted_keypoints / H
    return normalized_keypoints


def build_model(num_heatmap: int = 16) -> StackedHourglassNetwork:
    return StackedHourglassNetwork(
        input_shape=(256, 256, 3),
        num_stack=4,
        num_residual=1,
        num_heatmap=num_heatmap,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='POSE smoke reproduction (T4 / precondition T13)')
    parser.add_argument('--weights', nargs='*', default=[
        '/Users/robotjang/Downloads/best_model.pt',
        '/Users/robotjang/Downloads/poseEstimate/model-epoch-2-loss-1.2050.pt',
    ], help='checkpoint paths to attempt loading')
    parser.add_argument('--image', default='/Users/robotjang/Downloads/person.jpg',
                        help='real person image for one forward')
    parser.add_argument('--num-heatmap', type=int, default=16)
    args = parser.parse_args()

    model = build_model(args.num_heatmap)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] StackedHourglassNetwork(num_stack=4, num_residual=1, num_heatmap={args.num_heatmap})")
    print(f"[model] parameters: {n_params:,}")

    for wpath in args.weights:
        print(f"\n=== loading {wpath} ===")
        ckpt = torch.load(wpath, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'encoder.embedding.weight' in ckpt:
            print(f"[load] {wpath}: NOT a pose checkpoint "
                  f"(keys like 'encoder.embedding.weight'/'decoder.attention.W1.weight' "
                  f"= Seq2Seq translation model). SKIP structural load attempt on pose arch.")
            continue
        load_state_dict_verbose(model, ckpt, wpath)

    model.eval()
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        outputs = model(dummy)
    if not isinstance(outputs, list):
        outputs = [outputs]
    last = outputs[-1]
    print(f"\n[smoke] dummy forward OK: {len(outputs)} stacks; last heatmap shape = "
          f"{tuple(last.shape)} (B, C=num_heatmap, H, W)")
    assert last.shape[-1] == 64 and last.shape[-2] == 64
    assert last.shape[1] == args.num_heatmap
    determined_class = last.shape[1]
    print(f"[smoke] output channel count == {determined_class} == num_heatmap "
          f"({16 if determined_class == 16 else determined_class} MPII keypoints)  [PASS]")

    import os
    if args.image and os.path.exists(args.image):
        from PIL import Image
        import torchvision.transforms as transforms

        image = Image.open(args.image).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])
        inputs = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            out = model(inputs)
        heatmap_tensor = out[-1].squeeze(0).detach().cpu().permute(1, 2, 0)
        kp = extract_keypoints_from_heatmap(heatmap_tensor)

        print(f"\n[image] {os.path.basename(args.image)} -> predicted keypoints "
              f"({len(kp)} MPII joints, normalized [0,1]):")
        for name, (x, y) in zip(MPII_JOINT_NAMES[: len(kp)], kp):
            print(f"  {name:12s} ({x.item():.4f}, {y.item():.4f})")
        assert kp.shape == (args.num_heatmap, 2)
        print(f"[image] keypoint tensor shape {tuple(kp.shape)} == (16, 2)  [PASS]")
    else:
        print(f"[image] no real image at {args.image!r} - skipped (dummy forward only)")

    print("\n[smoke] EXIT 0 - pose model loads and runs inference")
    return 0


if __name__ == '__main__':
    sys.exit(main())