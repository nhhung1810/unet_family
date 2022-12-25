import json
import os
import os
from typing import List, Tuple
from cv2 import randShuffle

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks
import torch
from loss import DiceLoss
from nnet import UNet
from loader import ButterFly, draw_segmentation_masks


def load_model(path) -> UNet:
    try:
        model_path = os.path.join(path, 'model.pt')
        if not os.path.exists(model_path):
            raise Exception("Switch to state-dict loading")
        model: UNet = torch.load(model_path, map_location='cpu')
        return model
    except:
        model_path = os.path.join(path, "model-state-dict.pt")
        config_path = os.path.join(path, "1/config.json")
        with open(config_path, 'r') as out:
            model = UNet(init_features=json.load(out)['init_features'])
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model


def show_with_label(imgs: List[Tuple[torch.Tensor, str]]):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, (img, xlabel) in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=xlabel)

    plt.show()


def show_row(imgs: List[Tuple[torch.Tensor, str]], split: int = 5):
    if not isinstance(imgs, list):
        imgs = [imgs]

    _len = len(imgs)
    _n_row = _len // split
    _, axs = plt.subplots(ncols=split, nrows=_n_row, squeeze=False)
    for i, (img, xlabel) in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i // split, i % split].imshow(np.asarray(img))
        axs[i // split,
            i % split].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=xlabel)

    plt.show()


def make_visual_data(model: UNet, dataset: ButterFly, idx_list: List[int]):
    _to_show = []
    for idx in idx_list:
        info = dataset[idx]
        image = info['image'].unsqueeze(0)
        mask = info['mask']
        model.eval()

        with torch.no_grad():
            pred = model.forward(image)
            # Original Image
            _image = (image[0] * 255).type(torch.uint8)
            # Rescale Logit of prediction
            _rescale_pred = pred[0, 0].logit()
            _rescale_pred = (_rescale_pred - _rescale_pred.min()
                            ) / (_rescale_pred.max() - _rescale_pred.min()) * 255
            # Prediction with mean threshold
            _pred = (pred[0, 0] > pred.mean()).type(torch.uint8)
            # Overlay the prediction
            _overlay = draw_segmentation_masks(_image, masks=_pred.bool(), alpha=0.7)
            _to_show.extend(
                [
                    (_image, 'Orignal Image'), (mask[0] * 255, 'Ground truth mask'),
                    (_rescale_pred.type(torch.uint8), "Prediction's Logit"),
                    (_pred, 'Prediction with mean threshold'), (_overlay, 'Image with Prediction')
                ]
            )
        pass
    return _to_show
    # show_row(_to_show, split=5)


@torch.no_grad()
def _write_all_loss(model: UNet):
    dataset = ButterFly(metadata_path="./metadata.json", group='test')
    model.eval()
    dice_loss = DiceLoss()
    _loss = []
    for idx, data in enumerate(dataset):
        image = data['image'].unsqueeze(0)
        mask = data['mask'].unsqueeze(0)
        pred = model(image)
        loss = dice_loss.forward(pred, mask)
        _loss.append((idx, loss.item()))

    csv_str = "\n".join([f"{idx},{loss}" for idx, loss, in _loss])

    with open("loss_data.csv", 'w') as out:
        out.write("idx,loss\n")
        out.write(csv_str)
    pass


if __name__ == "__main__":
    model_path = {
        "baseline_2": "./model_store/baseline/baseline_unet2",
        # "baseline_4": "./model_store/baseline/baseline_unet4",
        # "baseline_8": "./model_store/baseline/baseline_unet8",
        # "baseline_16": "./model_store/baseline/baseline_unet16",
        # "baseline_32": "./model_store/baseline/baseline_unet32",
        # "aug_2": "./model_store/221223-augment/augment_unet2",
        "aug_2_25": "./model_store/221225-augment/augment_unet2",
        # "aug_4": "./model_store/221223-augment/augment_unet4",
        # "aug_8": "./model_store/221223-augment/augment_unet8",
        # "aug_16": "./model_store/221223-augment/augment_unet16",
        # "aug_32": "./model_store/221223-augment/augment_unet32",
    }
    # _write_all_loss(model=load_model(model_path['baseline_8']))

    dataset = ButterFly(metadata_path="./metadata.json", group='test')
    _to_show = []
    # idx = 81
    # for idx in [np.random.randint(low=0, high=len(dataset)) for _ in range(3)]:
    to_show = []
    for _, path in model_path.items():
        # path = model_path[16]
        model = load_model(path=path)
        # GOOD: 15, 70
        # BAD:  20
        _to_show = make_visual_data(model, dataset, [63, 59, 48, 9, 51, 22])
        to_show.extend(_to_show)
        # break
    show_row(to_show, split=5)
    pass