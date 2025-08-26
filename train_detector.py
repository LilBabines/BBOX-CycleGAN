import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
from data.detector_dataset import YOLODataset  # assume this is in dataset.py
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
import numpy as np
import math
import csv

from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_png


def collate_fn(batch):
    return tuple(zip(*batch))



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


@torch.inference_mode()
def save_sample_predictions(
    loader,
    model,
    device,
    epoch:int,
    split:str,
    runs_dir:str,
    max_images:int = 8,
    score_thresh:float = 0.1):
    """
    Sauvegarde jusqu'à `max_images` images du `loader` avec:
      - GT (vert, label 'gt')
      - Prédictions (rouge, label 'pred:<score>')
    """
    model.eval()
    out_dir = os.path.join(runs_dir, "preds", f"epoch_{epoch:03d}", split)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for images, targets in loader:
        # Envoie sur device
        images = [img.to(device) for img in images]
        outputs = model(images)

        for img, tgt, out in zip(images, targets, outputs):
            if saved >= max_images:
                break

            # img: CxHxW en [0,1] (vu ton Normalize std=255)
            img_u8 = (img.detach().clamp(0, 1).cpu() * 255).to(torch.uint8)
            if img_u8.shape[0] == 1:
                img_u8 = img_u8.repeat(3, 1, 1)   # draw_bounding_boxes attend 3 canaux

            # GT boxes
            gt_boxes = tgt.get("boxes", torch.empty((0,4))).detach().cpu()
            gt_labels = ["gt"] * (gt_boxes.shape[0] if gt_boxes is not None else 0)

            # Prédictions filtrées par score
            scores = out.get("scores", torch.empty(0)).detach().cpu()
            keep = scores >= score_thresh
            pred_boxes = out.get("boxes", torch.empty((0,4))).detach().cpu()[keep]
            pred_labels = [f"pred:{s:.2f}" for s in scores[keep].tolist()]

            # Dessin : d’abord GT (vert), puis prédictions (rouge)
            canvas = img_u8
            if gt_boxes.numel() > 0:
                canvas = draw_bounding_boxes(
                    canvas, gt_boxes.round().to(torch.int64),
                    labels=gt_labels, colors="green", width=2
                )
            if pred_boxes.numel() > 0:
                canvas = draw_bounding_boxes(
                    canvas, pred_boxes.round().to(torch.int64),
                    labels=pred_labels, colors="red", width=2
                )

            # Sauvegarde PNG
            save_path = os.path.join(out_dir, f"{split}_{saved:03d}.png")
            write_png(canvas, save_path)
            saved += 1

        if saved >= max_images:
            break

    print(f"[Epoch {epoch}] {split}: {saved} prédictions sauvegardées -> {out_dir}")


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0,0.2),per_channel=False),  # bruit blanc gaussien
        A.MotionBlur(blur_limit=3, p=0.1),  # optionnel pour simuler les flous radar
        A.Affine( scale=(0.9,1.1), rotate=90, p=0.5),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1,1,1)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        # label_fields=['labels']
    ))

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1,1,1)),
        ToTensorV2(),

    ]
    # , bbox_params=A.BboxParams(
    #     format='pascal_voc',
    #     # label_fields=['labels'])
    )
    
def get_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    num_classes = 2  # 1 class (foreground) + 1 background
    epochs = 20
    batch_size = 8
    lr = 0.02

    best_val_loss = math.inf
    runs_dir = "runs/train_detector"
    best_ckpt_path = os.path.join(runs_dir, "best_model.pt")
    csv_path = os.path.join(runs_dir, "training_log.csv")
    os.makedirs(runs_dir, exist_ok=True)

    # Prépare le CSV (écrase s'il existe)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_loss"])


    # Dataset paths
    root = "datasets/yolo_clean_less_empty"
    train_dataset = YOLODataset(
        images_dir=os.path.join(root, "images/train"),
        labels_dir=os.path.join(root, "labels/train"),
        albu_transforms=get_train_transforms(),
        remove_empty_pourcent=00
    )
    val_dataset = YOLODataset(
        images_dir=os.path.join(root, "images/val"),
        labels_dir=os.path.join(root, "labels/val"),
        albu_transforms=get_val_transforms(),
        remove_empty_pourcent=00
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

        save_sample_predictions(train_loader, model, device, epoch, "train", runs_dir,
                            max_images=8, score_thresh=0.01)
        save_sample_predictions(val_loader, model, device, epoch, "val", runs_dir,
                                max_images=8, score_thresh=0.01)


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
    # pass
    main()