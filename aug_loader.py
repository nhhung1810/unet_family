import json
import os
from pathlib import Path
from sys import meta_path
import albumentations as A
import cv2
from matplotlib.pyplot import flag
import numpy as np
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm, trange

from loader import make_data_path, show


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
                A.SmallestMaxSize(max_size=256, always_apply=True),
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

    def __getitem__(self, index) -> dict:
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
            return {}

    def __len__(self) -> int:
        return len(self.metadata)


class CachedAugButterfly(Dataset):

    def __init__(self, cached_metadata_path: str) -> None:
        super().__init__()
        self.cached_metadata_path = cached_metadata_path
        if not os.path.exists(self.cached_metadata_path):
            raise Exception(f"Invalid path")

        with open(self.cached_metadata_path, 'r') as out:
            self.metadata = json.load(out)

        self.data = []
        for _path in tqdm(self.metadata, desc="Loading data..."):
            try:
                self.data.append(torch.load(_path))
            except Exception as msg:
                print(msg)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        return self.data[index]

    @staticmethod
    def cycle(iterable):
        while True:
            for item in iterable:
                yield item

    @staticmethod
    def build_cache(
        dataset: AugButterFly,
        multiply_factor: float,
        cache_dir: str = "./data/augment_leedsbutterfly"
    ) -> str:
        """Build cache of augmentation data to speedup training

        Args:
            dataset (AugButterFly): augmentation dataset
                multiply_factor (float): number of cached data. Augmentation can generate 
                very large number of example so we will set a limit for it.
            cache_dir (str, optional): _description_. Defaults to "./data/augment_leedsbutterfly".

        Returns:
            str: _description_
        """
        _range = trange(int(len(dataset) * multiply_factor), desc="Caching augment data...")
        metadata = []
        _len = len(dataset)
        for idx in _range:
            data = dataset[idx % _len]
            if len(data.keys()) == 0:
                continue
            path = os.path.join(cache_dir, f"cached-{idx}.pt")
            torch.save(data, path)
            metadata.append(path)
            pass

        with open(os.path.join(cache_dir, "metadata.json"), 'w') as out:
            json.dump(metadata, out)
        pass


def build_augment(group: str, factor: float):
    dataset = AugButterFly('metadata.json', group=group, is_debugging=False)
    directory = f"./data/augment_leedsbutterfly/{group}"
    Path(directory).mkdir(parents=True, exist_ok=True)
    CachedAugButterfly.build_cache(dataset, factor, directory)


if __name__ == "__main__":
    # image_dir = "./data/leedsbutterfly/images"
    # seg_dir = "./data/leedsbutterfly/segmentations"
    # paths = list(make_data_path(image_dir, seg_dir))
    # dataset = AugButterFly('metadata.json', group='test', is_debugging=False)
    # CachedAugButterfly.build_cache(dataset, 0.1)
    build_augment(group='train', factor=0.1)
    build_augment(group='test', factor=1)

    # dataset = CachedAugButterfly(cached_metadata_path="./data/augment_leedsbutterfly/metadata.json")
    # data = dataset[5]
    # _to_show = [
    #     Tensor(data['trans_image'] * 255).type(torch.uint8),
    #     draw_segmentation_masks(
    #         Tensor(data['trans_image'] * 255).type(torch.uint8),
    #         masks=Tensor(data['trans_mask']) > 0,
    #         alpha=0.7
    #     ),
    # ]
    # show(_to_show)
    pass