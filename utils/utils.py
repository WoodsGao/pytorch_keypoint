import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_modules.nn import FocalBCELoss


CE = nn.CrossEntropyLoss()
BCE = nn.BCEWithLogitsLoss()
MSE = nn.MSELoss()
focal = FocalBCELoss()


def compute_loss(outputs, targets, model):
    outputs = F.interpolate(outputs, (targets.size(2), targets.size(3)),
                            mode='bilinear',
                            align_corners=True)
    loss = MSE(outputs.sigmoid(), targets)
    return loss


def show_batch(inputs, targets):
    imgs = inputs.clone()[:8]
    segs = targets.clone()[:8].sigmoid() * 255
    imgs *= torch.FloatTensor([58.395, 57.12,
                               57.375]).reshape(1, 3, 1, 1).to(imgs.device)
    imgs += torch.FloatTensor([123.675, 116.28,
                               103.53]).reshape(1, 3, 1, 1).to(imgs.device)

    imgs = imgs.clamp(0, 255).permute(0, 2, 3,
                                      1).byte().cpu().numpy()[..., ::-1]
    imgs = np.ascontiguousarray(imgs)
    segs = segs.cpu().byte().numpy()
    segs = np.ascontiguousarray(segs)
    for i in range(len(imgs)):
        seg = segs[i]

        for heat in seg:
            # cv2.imshow('h', heat)
            # cv2.waitKey(0)
            index = heat.argmax()
            y = index // heat.shape[0]
            x = index % heat.shape[0]
            cv2.circle(imgs[i], (x, y), 2, (0, 0, 255), -1)

    imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3])
    segs = segs.transpose(0, 2, 1, 3).clip(0, 255).astype(np.uint8)
    segs = segs.reshape(-1, segs.shape[2] * segs.shape[3], 1)
    segs = np.ascontiguousarray(segs)
    segs = cv2.cvtColor(segs, cv2.COLOR_GRAY2BGR)
    imgs = np.concatenate((imgs, segs), 1)
    cv2.imwrite('batch.png', imgs)


def compute_metrics(tp, fn, fp):
    union = tp + fp + fn
    union[union <= 0] = 1
    miou = tp / union
    T = tp + fn
    P = tp + fp
    P[P <= 0] = 1
    P = tp / P
    R = tp + fn
    R[R <= 0] = 1
    R = tp / R
    F1 = (2 * tp + fp + fn)
    F1[F1 <= 0] = 1
    F1 = 2 * tp / F1
    return T, P, R, miou, F1
