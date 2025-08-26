import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch

class BBOXUnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'

        self.dir_ALabel = os.path.join(opt.dataroot, 'bbox_labels')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range    
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("L")  
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        bbox_target = self.load_bbox(A_path,A_img.size)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path, "A_label":bbox_target}


    def load_bbox(self, A_path, img_size):

        stem = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(label_dir, stem + ".txt")

        boxes_xywh = []
        cls_list   = []
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    # Fichier YOLO: cls, xc, yc, bw, bh  (déjà normalisés 0..1)
                    _, xc, yc, bw, bh = parts
                    # Vérif de validité
                    if float(bw) <= 0 or float(bh) <= 0:
                        continue
                    # Dans ton cas (nc=1), on force la classe à 0
                    boxes_xywh.append([float(xc), float(yc), float(bw), float(bh)])
                    cls_list.append(0.0)

        bboxes = torch.tensor(boxes_xywh, dtype=torch.float32).reshape(-1, 4)          # (n,4) xywh norm
        cls    = torch.tensor(cls_list, dtype=torch.float32).reshape(-1, 1)            # (n,1) float

        return {  "cls": cls, "bboxes": bboxes   }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
