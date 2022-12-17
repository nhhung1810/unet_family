import fnmatch
import json
import os
from typing import List, Tuple

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


def scan_dir(root, pattern):
    """Scan directory and find file that match pattern

    Args:
        root (str): path of directory to begin scanning
        pattern (str): pattern to filter for

    Yields:
        str: Full path to the file
    """
    for dirpath, _, files in os.walk(root):
        files = fnmatch.filter(files, pattern)
        if len(files) == 0:
            continue
        for filename in files:
            yield os.path.join(dirpath, filename)


def make_data_path(image_dir: str, seg_dir: str):
    image_list = list(scan_dir(image_dir, "*.png"))
    for image_path in image_list:
        base_name, ext = os.path.splitext(os.path.basename(image_path))
        mask_name = f"{base_name}_seg0{ext}"
        mask_path = os.path.join(seg_dir, mask_name)
        assert os.path.exists(mask_path), f"{mask_path} doesn't exist -> Check your data"
        yield (image_path, mask_path)


def split(data_path: List[Tuple[str, str]], save_metadata: str):
    train_split, test_split = train_test_split(data_path, test_size=0.1, random_state=1810)
    metadata = {'train': train_split, 'test': test_split}
    with open(save_metadata, 'w') as out:
        json.dump(metadata, out)
    pass


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


class ButterFly(Dataset):

    def __init__(self, metadata_path: str, group: str = 'train') -> None:
        super().__init__()
        self.metadata_path = metadata_path
        self.group = group
        with open(self.metadata_path, 'r') as out:
            self.metadata = json.load(out)

        self.metadata = self.metadata[self.group]
        self.data = self.load_all()

    def load_all(self) -> list:
        _data = []
        for _, (image_path, mask_path) in tqdm(
            enumerate(self.metadata),
            desc="Loading all data...",
            total=len(self.metadata)
        ):
            imag = read_image(image_path)
            mask = read_image(mask_path)
            imag = F.resize(imag, (256, 256)).float() / 255.0
            mask = F.resize(mask, (256, 256)).float()
            mask = torch.mean(mask, dim=0, keepdim=True)
            mask[mask > 0.] = 1.
            mask[mask == 0.] = 0.
            _data.append({'image': imag, 'mask': mask})
            pass
        return _data

    def __getitem__(self, index) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.metadata)


if __name__ == "__main__":
    image_dir = "./data/leedsbutterfly/images"
    seg_dir = "./data/leedsbutterfly/segmentations"
    paths = list(make_data_path(image_dir, seg_dir))
    dataset = ButterFly('metadata.json')
    for data in dataset:
        show(
            [
                draw_segmentation_masks(
                    data['image'], masks=data['mask'] > 0., alpha=0.7
                )
            ]
        )
        break

    pass