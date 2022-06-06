import time
import os
import argparse
from cycleGAN import CycleGAN
from my_dataset import MyDataset
from torch.utils.data import DataLoader
from visual import Visualizer


def train(opt):

    model = CycleGAN(opt, is_train=True)
    model.set_up(opt)
    train_dataset = MyDataset(opt)
    train_img_len = len(train_dataset)
    print('the number of training images: {}'.format(train_img_len))

    visualizer = Visualizer(opt)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    total_iter = 0  # the total number of training iterations

    for epoch in range(opt.n_epoch+opt.n_epoch_decay):

        epoch_start_time = time.time()
        iter_data_time = time.time()

        epoch_iter = 0
        print(f'Epoch: {epoch}')
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            data_load_time = 0
            if total_iter % opt.print_freq == 0:
                data_load_time = iter_start_time - iter_data_time

            total_iter += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimizer_parameters()

            if total_iter % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_iter % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, data_load_time)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / train_img_len, losses)

            iter_data_time = time.time()

        model.adjust_learning_rate()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epoch + opt.n_epoch_decay, time.time() - epoch_start_time))

        if (epoch+1) % opt.save_epoch_freq == 0:
            model.save_networks(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 训练参数
    parser.add_argument('--data_path', type=str, default='./datasets/monet2photo',
                        help="the path where the dataset is located")
    parser.add_argument('--phase', type=str, default='train', help='train phase')
    parser.add_argument('--load_size', type=int, default=286, help='scale image to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop image to this size')
    parser.add_argument('--gpu-ids', type=str, default='3', help="ie. 0, 1, 2")
    parser.add_argument('--pool-size', type=int, default=50, help='the size of image buffer that '
                                                                  'stores previously generated images')
    parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epoch_decay', type=int, default=100, help='number of epochs to linearly decay learning '
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

    # 效果显示参数
    parser.add_argument('--display-id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display-winsize', type=int, default=256, help='display window size for visdom')
    parser.add_argument('--name', type=str, default='monet2photo', help='name of experiment')
    parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single '
                                                                     'visdom web panel with certain number of images '
                                                                     'per row.')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--display_freq', type=int, default=400, help='frequency of display training results on visdom')

    # 模型以及结果保存
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epoch')

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
    # save to the disk
    expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

    train(opt)
