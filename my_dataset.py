import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTENSIONS = [
    'jpg', 'JPG', 'jpeg', 'JPEG',
    'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP',
    'tif', 'TIF', 'tiff', 'TIFF',
]


def get_data(path):
    images_path = [os.path.join(path, img_path) for img_path in os.listdir(path)
                   if img_path.split('.')[-1] in IMG_EXTENSIONS]
    return images_path


# 这个只针对训练过程 测试过程还没有写
def get_transform(opt):
    trans = transforms.Compose([
        transforms.Resize(opt.load_size, transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(opt.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return trans


def get_test_transform(opt):

    trans = transforms.Compose([
        transforms.Resize(opt.load_size, transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(opt.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return trans


class SingleDataset(Dataset):

    def __init__(self, opt):
        super(SingleDataset, self).__init__()
        root_path = opt.data_path
        assert os.path.exists(root_path), f'{root_path} path is not exists...'

        self.img_list = get_data(root_path)
        self.img_len = len(self.img_list)
        self.transform = get_test_transform(opt)

    def __len__(self):
        return self.img_len

    def __getitem__(self, item):
        img_path = self.img_list[item % self.img_len]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'img': img, 'path': img_path}


class MyDataset(Dataset):

    def __init__(self, opt):
        super(MyDataset, self).__init__()

        root_A_path = os.path.join(opt.data_path, opt.phase+'A')
        root_B_path = os.path.join(opt.data_path, opt.phase+'B')
        assert os.path.exists(root_A_path), f'{root_A_path} path is not exists...'
        assert os.path.exists(root_B_path), f'{root_B_path} path is not exists...'

        self.img_A_list = get_data(root_A_path)
        self.img_B_list = get_data(root_B_path)

        self.A_size = len(self.img_A_list)
        self.B_size = len(self.img_B_list)

        self.transform_A = get_transform(opt)
        self.transform_B = get_transform(opt)

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, item):

        img_A_path = self.img_A_list[item % self.A_size]
        img_B_ind = random.randint(0, self.B_size-1)
        img_B_path = self.img_B_list[img_B_ind]

        img_A = Image.open(img_A_path).convert('RGB')
        img_B = Image.open(img_B_path).convert('RGB')

        if self.transform_A:
            img_A = self.transform_A(img_A)
        if self.transform_B:
            img_B = self.transform_B(img_B)
        return {'A': img_A, 'B': img_B, 'A_path': img_A_path, 'B_path': img_B_path}
