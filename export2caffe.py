import argparse
import os
import os.path as osp
import torch
from utils.models import DeepLabV3Plus
from pytorch_modules.utils import fuse
from pytorch2caffe import pytorch2caffe


def export2caffe(weights, num_classes, img_size):
    model = DeepLabV3Plus(num_classes)
    weights = torch.load(weights, map_location='cpu')
    model.load_state_dict(weights['model'])
    model.eval()
    fuse(model)
    name = 'DeepLabV3Plus'
    dummy_input = torch.ones([1, 3, img_size[1], img_size[0]])
    pytorch2caffe.trans_net(model, dummy_input, name)
    pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str)
    parser.add_argument('--num-classes', type=int, default=21)
    parser.add_argument('--img-size', type=str, default='512')
    opt = parser.parse_args()

    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0])] * 2
    else:
        img_size = [int(x) for x in img_size]
    export2caffe(opt.weights, opt.num_classes, img_size)
