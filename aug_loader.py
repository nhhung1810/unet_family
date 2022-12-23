import fnmatch
import json
import albumentations as A
import os
from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm

from loader import scan_dir, make_data_path, show


class AugButterFly(Dataset):

    def __init__(
        self,
        metadata_path: str,
        group: str = 'train',
        is_debugging=True,
    ) -> None:
        super().__init__()
        self.metadata_path = metadata_path
        self.group = group

        with open(self.metadata_path, 'r') as out:
            self.metadata = json.load(out)

        self.metadata = self.metadata[self.group]
        self.data = self.load_all()
        self.transform = A.Compose(
            [
                A.GridDropout(always_apply=is_debugging),
                A.PixelDropout(dropout_prob=0.1, p=0.25, always_apply=is_debugging),
                A.SafeRotate(p=0.5, always_apply=is_debugging),
                A.HorizontalFlip(p=0.5, always_apply=is_debugging),
                # A.RandomCrop(width=256, height=256, p=0.5, always_apply=is_debugging),
                A.ElasticTransform(always_apply=is_debugging, p=0.25),
                A.Resize(height=256, width=256, always_apply=True)
            ]
        )

    @staticmethod
    def _cv2Load(path) -> np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_all(self) -> list:
        _data = []
        for _, (image_path, mask_path) in tqdm(
            enumerate(self.metadata), desc="Loading all data...", total=len(self.metadata)
        ):

            imag = AugButterFly._cv2Load(image_path)
            mask = AugButterFly._cv2Load(mask_path)
            _data.append({'image': imag, 'mask': mask})
            pass

        return _data

    def __getitem__(self, index) -> Tensor:
        try:
            data = self.data[index]
            img: np.ndarray = data['image']
            mask: np.ndarray = data['mask']

            transformed = self.transform(image=img, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            # Transform to Torch Tensor
            transformed_image = Tensor(transformed_image).permute(2, 0, 1).float() / 255.
            # image = Tensor(img).permute(2, 0, 1).float() // 255.

            transformed_mask = Tensor(transformed_mask).permute(2, 0, 1).float()
            transformed_mask = torch.mean(transformed_mask, dim=0, keepdim=True)
            # mask = Tensor(mask).permute(2, 0, 1).float()
            transformed_mask[transformed_mask > 0.] = 1.
            # mask[mask > 0.] = 1.


            return {
                "trans_image": transformed_image,
                "trans_mask": transformed_mask,  # "image":image,
  # "mask":mask,
            }

        except Exception as msg:
            # Choose another image
            return self.__getitem__(index + 1)

    def __len__(self) -> int:
        return len(self.metadata)


if __name__ == "__main__":
    image_dir = "./data/leedsbutterfly/images"
    seg_dir = "./data/leedsbutterfly/segmentations"
    paths = list(make_data_path(image_dir, seg_dir))
    dataset = AugButterFly('metadata.json', group='test')
    data = dataset[0]
    _to_show = [
        Tensor(data['image']).type(torch.uint8),
        draw_segmentation_masks(
            Tensor(data['trans_image']).type(torch.uint8),
            masks=Tensor(data['trans_mask']) > 0,
            alpha=0.7
        ),
        draw_segmentation_masks(
            Tensor(data['image']).type(torch.uint8), masks=Tensor(data['mask']) > 0, alpha=0.7
        )
    ]
    show(_to_show)
    pass