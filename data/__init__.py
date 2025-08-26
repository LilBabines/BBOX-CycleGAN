"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""

import importlib
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

def detection_aware_collate_yolo(batch):
    """
    batch: liste de samples renvoyés par BBOXUnalignedDataset.__getitem__
           chaque sample = {"A": Tensor, "B": Tensor, "A_paths": str, "B_paths": str, "A_label": dict}

    Retour:
      out = {
        "A": Tensor [B,3,H,W],
        "B": Tensor [B,1,H,W] (dans ton code B est en niveau de gris),
        "A_paths": [str]*B,
        "B_paths": [str]*B,
        "A_label": [dict]*B,    # on conserve la liste de dicts par image
        "yolo_batch": {         # <-- pour v8DetectionLoss
           "batch_idx": LongTensor (N,),
           "cls":       FloatTensor (N,1),
           "bboxes":    FloatTensor (N,4)  # xywh norm [0,1]
        },
        "img_hw": (H, W)        # utile pour affichage/debug
      }
    """
    B = len(batch)
    # 1) Empiler A et B
    A = torch.stack([sample['A'] for sample in batch], dim=0)
    B_img = torch.stack([sample['B'] for sample in batch], dim=0)

    A_paths = [sample["A_paths"] for sample in batch]
    B_paths = [sample["B_paths"] for sample in batch]

    # On suppose que toutes les images ont la même taille après transform
    _, _, H, W = A.shape

    # 2) Conserver la liste de dicts A_label telle quelle
    A_label_list = [sample["A_label"] for sample in batch]

    # 3) Construire le batch YOLO aplati
    idx_list, cls_list, box_list = [], [], []
    for i, t in enumerate(A_label_list):
        # On tolère plusieurs formats, on normalise:
        #  - t["bboxes"]: (n,4) xywh norm
        #  - t["cls"]: (n,1) float  OU (n,) int/float
        b = t.get("bboxes", None)
        c = t.get("cls", None)
        if b is None or b.numel() == 0:
            continue

        b = torch.as_tensor(b, dtype=torch.float32).view(-1, 4)  # (n,4)
        # cls -> (n,1) float
        if c is None:
            # nc=1 par défaut -> classe 0
            c = torch.zeros((b.shape[0], 1), dtype=torch.float32)
        else:
            c = torch.as_tensor(c)
            if c.ndim == 1:
                c = c.view(-1, 1)
            c = c.to(torch.float32)

        bi = torch.full((b.shape[0],), i, dtype=torch.long)       # (n,)

        box_list.append(b)
        cls_list.append(c)
        idx_list.append(bi)

    if len(box_list) == 0:  # aucun objet dans le batch
        batch_idx = torch.empty(0, dtype=torch.long)
        cls       = torch.empty(0, 1, dtype=torch.float32)
        bboxes    = torch.empty(0, 4, dtype=torch.float32)
    else:
        batch_idx = torch.cat(idx_list, dim=0)         # (N,)
        cls       = torch.cat(cls_list, dim=0)         # (N,1)
        bboxes    = torch.cat(box_list, dim=0)         # (N,4) xywh norm

    out = {
        "A": A,
        "B": B_img,
        "A_paths": A_paths,
        "B_paths": B_paths,
        "A_label": {
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
        },
        "img_hw": (H, W),
    }
    return out

class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        # Use DistributedSampler for DDP training
        if "LOCAL_RANK" in os.environ:
            print(f'create DDP sampler on rank {int(os.environ["LOCAL_RANK"])}')
            self.sampler = DistributedSampler(self.dataset, shuffle=not opt.serial_batches)
            shuffle = False  # DistributedSampler handles shuffling
        else:
            self.sampler = None
            shuffle = not opt.serial_batches

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=shuffle, sampler=self.sampler, num_workers=int(opt.num_threads),collate_fn=detection_aware_collate)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def set_epoch(self, epoch):
        """Set epoch for DistributedSampler to ensure proper shuffling"""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
