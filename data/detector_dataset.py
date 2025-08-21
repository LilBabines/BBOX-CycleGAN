import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def clean_yolo_labels(label_dir):
    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)
        cleaned_lines = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                if bw <= 0 or bh <= 0:
                    print(f"[Skipped] {file}: {line.strip()}")
                    continue
                cleaned_lines.append(line.strip())
        with open(path, 'w') as f:
            for line in cleaned_lines:
                f.write(line + "\n")


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, albu_transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.albu_transforms = albu_transforms
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.tif', '.tiff'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]

        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, os.path.splitext(image_filename)[0] + '.txt')
        # print(f"Loading image: {image_path}")
        # print(f"Loading label: {label_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        boxes = []
        labels = []
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    xmin = (xc - bw / 2) * w
                    ymin = (yc - bh / 2) * h
                    xmax = (xc + bw / 2) * w
                    ymax = (yc + bh / 2) * h
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(cls))

        if self.albu_transforms is not None:
            transformed = self.albu_transforms(image=np.array(image), bboxes=boxes)
            image = transformed['image']
            boxes = transformed['bboxes']
            # labels = transformed['labels']
        # Tensors, même si vides
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor([])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # if self.transforms:
        #     image = self.transforms(image)

        return image, target