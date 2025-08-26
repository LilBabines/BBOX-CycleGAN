import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
import os
import random
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from types import SimpleNamespace
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

yolo = YOLO("runs/yolo_less_empty2/weights/best.pt")
net = yolo.model                 
net.eval()


m = net.model[-1]  
print("nc:", m.nc, "reg_max:", getattr(m, "reg_max", 1), "stride:", m.stride)
print("has args:", hasattr(net, "args"))  


transform = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor()
])

for p in net.parameters():
    p.requires_grad = False      
def load_img_batch(
    dir_images: str,
    k: int = 8,
    img_size: int = 800,
    class_id: int = 0, ):
    """
    Retourne:
      x:        Tensor [B,3,img_size,img_size]
      batch_y:  dict avec
                - "batch_idx": LongTensor [N] (index image pour chaque box)
                - "cls":       FloatTensor [N,1] (id de classe, ici 0)
                - "bboxes":    FloatTensor [N,4] (xywh normalisés)
    Hypothèses:
      - Les labels sont en format YOLO (cls xc yc w h) normalisés [0,1]
      - Dossier des labels = dir_images avec 'images' remplacé par 'labels'
      - Fichiers labels ont la même racine + extension .txt
    """

    # Sélectionne k fichiers image existants (évite les non-images)
    all_files = [f for f in os.listdir(dir_images)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"))]
    if len(all_files) < k:
        raise ValueError(f"Pas assez d'images dans {dir_images} (trouvé {len(all_files)}, demandé {k}).")
    files = random.sample(all_files, k=k)

    # Transform torch différentiable (resize + ToTensor)
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # [0,1], shape [3,H,W]
    ])

    images = []
    all_boxes = []   # liste de tensors (ni,4)
    all_cls = []     # liste de tensors (ni,1)
    all_idx = []     # liste de tensors (ni,) indices image

    for i, fname in enumerate(files):
        img_path = Path(dir_images) / fname

        # Construit le chemin du .txt: remplace 'images' -> 'labels' dans le dossier
        # et met l'extension .txt
        labels_dir = Path(str(dir_images).replace("images", "labels"))
        stem = os.path.splitext(fname)[0]     # "image.tif" -> "image"
        txt_path = labels_dir / f"{stem}.txt"

        # ---- image -> tensor
        img = Image.open(img_path).convert("RGB")
        images.append(transform(img))

        # ---- labels -> concat aplatie (N,4) et (N,1)
        if txt_path.exists() and os.path.getsize(txt_path) > 0:
            boxes = []
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # ignore lignes corrompues
                    cls_str, xc, yc, bw, bh = parts
                    # Avec nc=1, on force la classe à 0 (ou vérifie)
                    # Si tu veux respecter la classe du fichier:
                    # cls_id = int(float(cls_str))
                    cls_id = class_id
                    boxes.append([float(xc), float(yc), float(bw), float(bh)])

            if boxes:
                boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)   # (ni,4)
                cls = torch.full((boxes.shape[0], 1), float(class_id))            # (ni,1) float
                idx = torch.full((boxes.shape[0],), i, dtype=torch.long)          # (ni,)
            else:
                boxes = torch.empty(0, 4, dtype=torch.float32)
                cls = torch.empty(0, 1, dtype=torch.float32)
                idx = torch.empty(0, dtype=torch.long)
        else:
            boxes = torch.empty(0, 4, dtype=torch.float32)
            cls = torch.empty(0, 1, dtype=torch.float32)
            idx = torch.empty(0, dtype=torch.long)

        all_boxes.append(boxes)
        all_cls.append(cls)
        all_idx.append(idx)

    # Empile les images -> [B,3,H,W]
    x = torch.stack(images, dim=0)

    # Concatène toutes les annotations sur N (= somme des ni)
    if all_boxes:
        batch_bbox = torch.cat(all_boxes, dim=0) if all_boxes[0].numel() else torch.empty(0,4, dtype=torch.float32)
        batch_labels = torch.cat(all_cls, dim=0) if all_cls[0].numel() else torch.empty(0,1, dtype=torch.float32)
        batch_idx = torch.cat(all_idx, dim=0) if all_idx[0].numel() else torch.empty(0, dtype=torch.long)
    else:
        batch_bbox = torch.empty(0, 4, dtype=torch.float32)
        batch_labels = torch.empty(0, 1, dtype=torch.float32)
        batch_idx = torch.empty(0, dtype=torch.long)

    # Sanity prints
    print("bbox shape : ", batch_bbox.shape)    # (N,4)
    print("labels shape : ", batch_labels.shape)  # (N,1)
    print("idx shape : ", batch_idx.shape)        # (N,)

    batch_y = {
        "batch_idx": batch_idx,   # LongTensor (N,)
        "cls": batch_labels,      # FloatTensor (N,1)
        "bboxes": batch_bbox,     # FloatTensor (N,4) en xywh normalisés
    }
    return x, batch_y






# ⚠️ Adapter net.args si c'est un dict
h = net.args
if isinstance(h, dict):
    # Valeurs usuelles par défaut si absentes dans le dict
    box = h.get("box", 7.5)
    cls = h.get("cls", 0.5)
    dfl = h.get("dfl", 1.5)

    # Conserve tout le contenu de h + ajoute des attributs .box/.cls/.dfl
    net.args = SimpleNamespace(**h)
    net.args.box = box
    net.args.cls = cls
    net.args.dfl = dfl

crit = v8DetectionLoss(model=yolo.model)   

x,batch_y = load_img_batch("datasets/yolo_clean_less_empty/images/val")

preds = net(x)

loss_vec, _ = crit(preds,batch_y)
print(loss_vec)

loss = loss_vec.sum() / x.shape[0]  # moyenne par image
print(loss)
# loss.backward()


# feats = cartes de features (P3,P4,P5) renvoyées par le head
feats = preds[1] if isinstance(preds, tuple) else preds

# reshape prédictions comme dans v8DetectionLoss
no = net.model[-1].nc + net.model[-1].reg_max * 4
pred_distri, pred_scores = torch.cat(
    [xi.view(x.shape[0], no, -1) for xi in feats], 2
).split((net.model[-1].reg_max * 4, net.model[-1].nc), 1)

pred_scores = pred_scores.permute(0, 2, 1).sigmoid()  # (B,anchors,nc)
pred_distri = pred_distri.permute(0, 2, 1)            # (B,anchors,reg_max*4)

# ancrages (points d’ancrage et stride)
from ultralytics.utils.tal import make_anchors
anchor_points, stride_tensor = make_anchors(feats, net.model[-1].stride, 0.5)

# décoder boxes (distribution focal loss -> xyxy)
from ultralytics.utils.loss import dist2bbox
proj = torch.arange(net.model[-1].reg_max, device=x.device, dtype=torch.float)
b, a, c = pred_distri.shape
pred_dist = pred_distri.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_distri.dtype))
pred_bboxes = dist2bbox(pred_dist, anchor_points, xywh=False) * stride_tensor  # (B,anchors,4)

# sélectionner les meilleures prédictions (exemple : top 5 par image)
topk = 5
scores, idxs = pred_scores.max(-1)   # max score de classe par anchor
for i in range(x.shape[0]):
    img = x[i].permute(1,2,0).cpu().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # topk indices
    conf, inds = scores[i].topk(topk)
    for j, ind in enumerate(inds):
        if conf[j] < 0.3:  # seuil de confiance
            continue
        box = pred_bboxes[i, ind].cpu().numpy()  # [x1,y1,x2,y2]
        w, h = box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle(
            (box[0], box[1]), w, h,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1]-2, f"{conf[j]:.2f}", color='red', fontsize=8)
    plt.show()