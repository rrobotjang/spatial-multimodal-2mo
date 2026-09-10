"""
Smoke test for DET knowledge extraction (Task 2).
RetinaNet-ResNet50-FPN: instantiate, forward dummy, verify output shape.
"""

import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

KMP_DUPLICATE_LIB_OK = True

NUM_CLASSES = 8
KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
CLASS_TO_ID = {name: i for i, name in enumerate(KITTI_CLASSES)}

print(f"PyTorch     : {torch.__version__}")
print(f"Torchvision : {torchvision.__version__}")

# 1. Instantiate RetinaNet-ResNet50-FPN with COCO pretrained weights
model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)

# 2. Swap classification head for KITTI 8 classes
old_head = model.head.classification_head
model.head.classification_head = RetinaNetClassificationHead(
    in_channels=model.backbone.out_channels,
    num_anchors=old_head.num_anchors,
    num_classes=NUM_CLASSES,
    prior_probability=0.01,
)

print(f"Model: {model.__class__.__name__}")
print(f"KITTI classes: {NUM_CLASSES}")

# 3. Forward dummy tensor in eval mode
model.eval()
dummy = torch.randn(1, 3, 375, 1242)  # typical KITTI resolution
with torch.no_grad():
    outputs = model(dummy)

output = outputs[0]
print(f"Output keys: {list(output.keys())}")
print(f"boxes shape : {output['boxes'].shape}")
print(f"labels shape: {output['labels'].shape}")
print(f"scores shape: {output['scores'].shape}")

assert output['boxes'].ndim == 2 and output['boxes'].shape[1] == 4, "boxes must be [N, 4]"
assert output['labels'].ndim == 1, "labels must be [N]"
assert output['scores'].ndim == 1, "scores must be [N]"

# 4. Verify classification head output channels
head = model.head.classification_head
print(f"Head num_classes: {head.num_classes}")
assert head.num_classes == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {head.num_classes}"

# 5. Test KITTIRetinaDataset with fabricated mini-dataset (no real KITTI needed)
from PIL import Image
import numpy as np
import tempfile, os

tmpdir = tempfile.mkdtemp()
# Create 2 dummy KITTI-format images
for i in range(2):
    img = Image.fromarray(np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8))
    img.save(os.path.join(tmpdir, f"{i:06d}.png"))

# Simulate KITTI-style annotations (torchvision Kitti format)
# Each annotation dict: {"type": str, "bbox": (x1,y1,x2,y2), ...}

class FakeKITTIDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, n=2):
        self.img_dir = img_dir
        self.n = n
        # Fabricated annotations: 1 car per image
        self.annotations = [
            [{"type": "Car", "bbox": (100, 200, 400, 350), "truncated": 0, "occluded": 0}],
            [{"type": "Pedestrian", "bbox": (500, 150, 550, 350), "truncated": 0, "occluded": 0}],
        ]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(os.path.join(self.img_dir, f"{idx:06d}.png")).convert("RGB")
        return img, self.annotations[idx]


class KITTIRetinaDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices=None, train=False):
        self.base = base_dataset
        self.indices = list(range(len(base_dataset))) if indices is None else list(indices)
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, raw_target = self.base[real_idx]
        image = torchvision.transforms.functional.to_tensor(image)

        boxes, labels = [], []
        for obj in raw_target:
            name = obj["type"]
            if name not in CLASS_TO_ID:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_ID[name])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([real_idx])}
        return image, target


fake_base = FakeKITTIDataset(tmpdir)
dataset = KITTIRetinaDataset(fake_base, [0, 1])
print(f"Dataset len: {len(dataset)}")

img, tgt = dataset[0]
print(f"Image shape: {img.shape}")
print(f"Target boxes: {tgt['boxes'].shape}")
print(f"Target labels: {tgt['labels']}")

# 6. Forward dataset sample through model (eval mode, no loss)
model.eval()
with torch.no_grad():
    out = model([img])[0]
print(f"Forward on dataset sample OK — {len(out['scores'])} detections")

# 7. Compute a dummy training step (1 batch, 1 iteration) to verify loss computation
model.train()
fake_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda b: tuple(zip(*b)))
images, targets = next(iter(fake_loader))
images = [i for i in images]
targets = [{k: v for k, v in t.items()} for t in targets]
loss_dict = model(images, targets)
loss = sum(loss_dict.values())
print(f"Training loss: {loss.item():.4f}")
print(f"Loss components: {', '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())}")

assert loss.item() > 0, "Loss must be positive"

print("\n=== SMOKE PASSED ===")
print(f"RetinaNet-ResNet50-FPN | {NUM_CLASSES} KITTI classes | loss={loss.item():.4f}")
