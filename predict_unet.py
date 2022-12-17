import os

import torch
from nnet import UNet
from loader import ButterFly, draw_segmentation_masks, show


def load_model() -> UNet:
    # path = "./model_store/baseline_unet32-221215-001859"
    path = "./model_store/baseline_unet8-221215-000512"
    path = "./model_store/baseline_unet16-221215-001108"
    model_path = os.path.join(path, 'model.pt')
    # config_path = os.path.join(path, '1/config.json')
    # with open(config_path, 'r') as out:
    #     config = json.load(out)

    # init_features = config['init_features']
    # model = UNet(init_features=init_features)
    model: UNet = torch.load(model_path, map_location='cpu')
    return model


if __name__ == "__main__":
    model = load_model()
    dataset = ButterFly(metadata_path="./metadata.json", group='test')
    info = dataset[0]
    image = info['image'].unsqueeze(0)
    mask = info['mask']
    model.eval()
    with torch.no_grad():
        pred = model.forward(image)
        _image = (image[0] * 255).type(torch.uint8)
        _rescale_pred = pred[0, 0].logit()
        _rescale_pred = (_rescale_pred - _rescale_pred.min()
                        ) / (_rescale_pred.max() - _rescale_pred.min()) * 255
        _pred = (pred[0, 0] > pred.mean()).type(torch.uint8)
        _overlay = draw_segmentation_masks(
            _image, masks=_pred.bool(), alpha=0.7
        )
        _to_show = [
            _image,
            mask[0] * 255,
            _rescale_pred.type(torch.uint8),
            _pred,
            _overlay
        ]
        show(_to_show)

        pass
    pass