import cv2
import numpy as np
import torch

from pytorch_modules.utils import device


@torch.no_grad()
def inference(model, imgs, img_size=(64, 64)):
    shapes = [img.shape for img in imgs]
    imgs = [
        cv2.resize(img, img_size)[:, :, ::-1].transpose(2, 0,
                                                        1).astype(np.float32)
        for img in imgs
    ]
    imgs = torch.FloatTensor(imgs).to(device)
    imgs -= torch.FloatTensor([123.675, 116.28,
                               103.53]).reshape(1, 3, 1, 1).to(imgs.device)
    imgs /= torch.FloatTensor([58.395, 57.12,
                               57.375]).reshape(1, 3, 1, 1).to(imgs.device)
    preds = model(imgs)
    kps = np.zeros((preds.shape[0], preds.shape[1], 2))
    for bi, pred in enumerate(preds):
        for ki, heat in enumerate(pred):
            index = heat.argmax()
            y = index // heat.shape[0]
            x = index % heat.shape[0]
            kps[bi, ki] = (x, y)
    kps[..., 0] /= float(img_size[0])
    kps[..., 1] /= float(img_size[1])
    return kps
