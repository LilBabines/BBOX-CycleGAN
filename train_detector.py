import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
from data.detector_dataset import YOLODataset  # assume this is in dataset.py
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
import numpy as np
import math
import csv


def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def metrics(target_boxes, output_boxes, output_scores, iou_threshold=0.5):
    
    iou_matrix = box_iou(target_boxes, output_boxes)
    gt_idx, pred_idx = linear_sum_assignment(-iou_matrix.numpy())

    for i, j in zip(gt_idx, pred_idx):
        if iou_matrix[i, j] >= iou_threshold:
            tp += 1
        else:
            fn += 1
            fp += 1
    for i in range(len(target_boxes)):
        if i not in gt_idx:
            fn  += 1
    for j in range(len(output_boxes)):
        if j not in pred_idx:
            fp += 1
    return tp, fp, fn





def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),  # bruit blanc gaussien
        A.MotionBlur(blur_limit=3, p=0.1),  # optionnel pour simuler les flous radar
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        # label_fields=['labels']
    ))

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0)),
        ToTensorV2(),

    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        # label_fields=['labels']
    ))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    num_classes = 2  # 1 class (foreground) + 1 background
    epochs = 20
    batch_size = 8
    lr = 0.005

    best_val_loss = math.inf
    runs_dir = "runs/detector"
    best_ckpt_path = os.path.join(runs_dir, "best_model.pt")
    csv_path = os.path.join(runs_dir, "training_log.csv")
    os.makedirs(runs_dir, exist_ok=True)

    # Prépare le CSV (écrase s'il existe)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_loss"])


    # Dataset paths
    root = "datasets/yolo_clean"
    train_dataset = YOLODataset(
        images_dir=os.path.join(root, "images/train"),
        labels_dir=os.path.join(root, "labels/train"),
        albu_transforms=get_train_transforms()
    )
    val_dataset = YOLODataset(
        images_dir=os.path.join(root, "images/val"),
        labels_dir=os.path.join(root, "labels/val"),
        albu_transforms=get_val_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

   
    for epoch in range(epochs):

        model.train()
        total_loss = 0.0
        for images, targets in tqdm(train_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        avg_train_loss = total_loss / max(1, len(train_loader))
        
        print(f"[Epoch {epoch+1}] Loss: {avg_train_loss:.4f}")

        # Validation loop
        total_val_loss = 0.0
        nb_batches = 0
        # model.eval()
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                
                loss_dict = model(images, targets)

                # Calculate metrics
                
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses.item()
                nb_batches += 1


        avg_val_loss = total_val_loss / nb_batches if nb_batches > 0 else 0.0
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")

        # ---- Log CSV ----
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}"])

        # ---- Save best checkpoint only ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"✅ Nouveau meilleur modèle sauvegardé à: {best_ckpt_path} (val_loss={best_val_loss:.6f})")

    print(f"Entraînement terminé. Meilleure val_loss: {best_val_loss:.6f}")
    print(f"Chemin du meilleur modèle: {best_ckpt_path}")
    print(f"CSV des pertes: {csv_path}")





if __name__ == "__main__":

    # Clean YOLO labels before training

    # from data.detector_dataset import clean_yolo_labels
    # clean_yolo_labels("datasets/yolo_clean/labels/train")
    # clean_yolo_labels("datasets/yolo_clean/labels/val")

    main()