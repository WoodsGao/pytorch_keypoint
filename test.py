import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from pytorch_modules.utils import device, Fetcher
from models import DeepLabV3Plus
from utils.datasets import CocoDataset
from utils.utils import compute_loss, show_batch, compute_metrics
from tqdm import tqdm
import argparse


@torch.no_grad()
def test(model, fetcher):
    model.eval()
    val_loss = 0
    classes = fetcher.loader.dataset.classes
    num_classes = len(classes)
    total_size = 0
    # true positive / intersection
    n = torch.zeros(num_classes)
    l2_sum = torch.zeros(num_classes)
    pbar = tqdm(fetcher)
    for idx, (inputs, targets) in enumerate(pbar):
        batch_idx = idx + 1
        outputs = model(inputs)
        loss = compute_loss(outputs, targets, model)
        val_loss += loss.item()
        normalize_size = (64, 64)
        targets = F.interpolate(targets,
                                normalize_size,
                                mode='bilinear',
                                align_corners=False).view(
                                    targets.size(0), targets.size(1),
                                    normalize_size[0] *
                                    normalize_size[1]).argmax(2)
        outputs = F.interpolate(outputs,
                                normalize_size,
                                mode='bilinear',
                                align_corners=False).view(
                                    outputs.size(0), outputs.size(1),
                                    normalize_size[0] *
                                    normalize_size[1]).argmax(2)
        y_dis = (targets // normalize_size[0] - outputs // normalize_size[0]) / float(normalize_size[1])
        x_dis = (targets % normalize_size[0] - outputs % normalize_size[0]) / float(normalize_size[0])
        l2 = y_dis**2 + x_dis**2
        l2 = torch.sqrt(l2)
        n += len(l2)
        l2_sum += l2.sum(0).cpu()
        pbar.set_description(
            'loss: %8g, NME: %8g' %
            (val_loss / batch_idx, l2_sum.sum() / max(1, n.sum())))
    if dist.is_initialized():
        n = n.to(device)
        l2_sum = l2_sum.to(device)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        dist.all_reduce(l2_sum, op=dist.ReduceOp.SUM)

    for c_i, c in enumerate(classes):
        print('cls: %8s, NME: %8g' % (c, l2_sum[c_i] / max(1, n[c_i])))
    return (l2_sum.sum() / max(1, n.sum())).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', type=str, default='data/coco.json')
    parser.add_argument('--img-size', type=str, default='512')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--num-workers', type=int, default=4)
    opt = parser.parse_args()

    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0])] * 2
    else:
        img_size = [int(x) for x in img_size]

    val_data = CocoDataset(opt.coco, img_size=img_size, augments=None)
    val_loader = DataLoader(
        val_data,
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
    )
    val_fetcher = Fetcher(val_loader, post_fetch_fn=val_data.post_fetch_fn)
    model = DeepLabV3Plus(len(val_data.classes))
    if opt.weights:
        state_dict = torch.load(opt.weights, map_location='cpu')
        model.load_state_dict(state_dict['model'])
    metrics = test(model, val_fetcher)
    print('metrics: %8g' % (metrics))
