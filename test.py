import torch
import argparse
from torch.utils.data import DataLoader

from my_dataset import SingleDataset
from cycleGAN import TestModel


def test(opt):

    test_dataset = SingleDataset(opt)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    model = TestModel(opt)
    model.load_networks(opt)

    for i, data in enumerate(test_loader):

        model.set_input(data)
        model.test()
        model.save_result(opt)
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, opt.save_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/monet2photo/testB',
                        help="the path where the dataset is located")
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--phase', type=str, default='test', help='test phase')
    parser.add_argument('--load_size', type=int, default=256, help='scale image to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop image to this size')
    parser.add_argument('--gpu-ids', type=str, default='6', help="ie. 0, 1, 2")
    parser.add_argument('--in_c', type=int, default=3, help='input channel in G_A and output channel in G_B')
    parser.add_argument('--out_c', type=int, default=3, help='output channel in G_A and input channel in G_B')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--suffix', type=str, default='B', help='model suffix')
    parser.add_argument('--model_path', type=str, default='./checkpoints/monet2photo')
    parser.add_argument('--norm', type=str, default="Instance Norm",
                        help="Instance Norm or Batch Norm")
    parser.add_argument('--epoch', type=str, default='199',
                        help='which epoch to load? set to latest to use latest cached model')

    opt = parser.parse_args()

    # 打印和保存命令行参数
    message = ''
    message += '-----------------------------options---------------------------\n'

    for k, v in sorted(vars(opt).items()):
        comment = ''
        default_val = parser.get_default(k)

        if v != default_val:
            comment = '\t[default: %s]' % str(default_val)
        message += '{:>25}:{:<30}{}\n'.format(str(k), str(v), comment)

    message += '------------------------------end---------------------------------\n'

    test(opt)