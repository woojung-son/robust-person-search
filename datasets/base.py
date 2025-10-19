import torch
from PIL import Image
import numpy as np
import random

class BaseDataset:
    """
    Base class of person search dataset.
    """

    def __init__(self, root, corruption, transforms, split):
        self.root = root
        self.corruption = corruption
        self.corrupt_prob = 2.0
        self.transforms = transforms
        self.split = split
        assert self.split in ("train", "gallery", "query", "full")
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """
        For each image, load its annotation that is a dictionary with the following keys:
            img_name (str): image name
            img_path (str): image path
            boxes (np.array[N, 4]): ground-truth boxes in (x1, y1, x2, y2) format
            pids (np.array[N]): person IDs corresponding to these boxes
            cam_id (int): camera ID (only for PRW dataset)
        """
        raise NotImplementedError

    def __getitem__(self, index):
        anno = self.annotations[index]
        img = Image.open(anno["img_path"]).convert("RGB")
        boxes = torch.as_tensor(anno["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(anno["pids"], dtype=torch.int64)
        if "domain" in anno:
            target = {"img_name": anno["img_name"], "boxes": boxes, "labels": labels, "domain": anno["domain"]}
        else:
            target = {"img_name": anno["img_name"], "boxes": boxes, "labels": labels}

        if self.corruption is not None:
            img = self.corruption(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.annotations)
