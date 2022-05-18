import argparse
from cycleGAN import CycleGAN
from my_dataset import MyDataset
from torch.utils.data import DataLoader


def train(opt):

    model = CycleGAN(opt, is_train=True)
    model.set_up(opt)
    train_dataset = MyDataset(opt)
    train_img_len = len(train_dataset)
    print('the number of training images: {}'.format(train_img_len))

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    for epoch in range(opt.n_epoch+opt.n_epoch_decay):
        model.adjust_learning_rate()

        for i, data in enumerate(train_loader):
            model.set_input(data)
            model.optimizer_parameters()
        pass
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 训练参数
    parser.add_argument('--data_path', type=str, default='./datasets/monet2photo',
                        help="the path where the dataset is located")
    parser.add_argument('--phase', type=str, default='train', help='train phase')
    parser.add_argument('--load_size', type=int, default=286, help='scale image to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop image to this size')
    parser.add_argument('--gpu-ids', type=str, default='0', help="ie. 0, 1, 2")
    parser.add_argument('--pool-size', type=int, default=50, help='the size of image buffer that '
                                                                  'stores previously generated images')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epoch_decay', type=int, default=50, help='number of epochs to linearly decay learning '
                                                                      'rate to zero')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--verbose', action='store_true', help='if specified, print the model structure and parameters')

    # 模型参数
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 in adam')
    parser.add_argument('--batch-size', type=int, default=1, help='the batch size')
    parser.add_argument('--norm', type=str, default="Instance Norm",
                        help="Instance Norm or Batch Norm")
    parser.add_argument('--epoch', type=int, default=200, help="total epochs of the training")
    parser.add_argument('--in_c', type=int, default=3, help='input channel in G_A and output channel in G_B')
    parser.add_argument('--out_c', type=int, default=3, help='output channel in G_A and input channel in G_B')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')

    # 损失参数
    parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.')
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> A)')
    parser.add_argument('--gan-mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla | lsgan]')

    opt = parser.parse_args()

    train(opt)
