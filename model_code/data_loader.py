import numpy as np
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import os
import utils
from torch.utils.data import Dataset


def get_dataloader(batch_size, domain):
    datas = {
        'sku': "/unsullied/sharefs/wangyimu/data/dataset/mydataset/4evaluation/sku/unsupervised/",
        'shelf': "/unsullied/sharefs/wangyimu/data/dataset/mydataset/4evaluation/shelf/unsupervised/",
        'web': "/unsullied/sharefs/wangyimu/data/dataset/mydataset/4evaluation/Web/unsupervised/listbycls/",
    }
    img_size = (227, 227)

    transform = [
        transforms.Scale(img_size),
        transforms.ToTensor()
    ]

    data_loader = data.DataLoader(
        dataset=datasets.ImageFolder(
            datas[domain],
            transform=transforms.Compose(transform),
        ),
        num_workers=16,
        batch_size=batch_size,
        shuffle=True,
        # shuffle=False,
        drop_last=True
    )

    return data_loader


def get_dataloader_target(batch_size, domain, istrain):
    datas = {
        'train': {
            'sku': '/unsullied/sharefs/wangyimu/data/dataset/mydataset/4evaluation/sku/semi/train/',
            'shelf': '/unsullied/sharefs/wangyimu/data/dataset/mydataset/4evaluation/shelf/semi/train/',
            'web': '/unsullied/sharefs/wangyimu/data/dataset/mydataset/4evaluation/Web/semi/listbycls/train/'
        },
        'test': {
            'sku': '/unsullied/sharefs/wangyimu/data/dataset/mydataset/4evaluation/sku/semi/test/',
            'shelf': '/unsullied/sharefs/wangyimu/data/dataset/mydataset/4evaluation/shelf/semi/test/',
            'web': '/unsullied/sharefs/wangyimu/data/dataset/mydataset/4evaluation/Web/semi/listbycls/test/'
        },
    }
    img_size = (227, 227)

    transform = [
        transforms.Scale(img_size),
        transforms.ToTensor()
    ]

    data_loader = data.DataLoader(
        dataset=datasets.ImageFolder(
            datas[istrain][domain],
            transform=transforms.Compose(transform),
        ),
        num_workers=16,
        batch_size=batch_size,
        shuffle=True,
        # shuffle=False,
        drop_last=True
    )

    return data_loader
