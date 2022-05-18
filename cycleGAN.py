import torch
import torch.nn as nn
import itertools
from functools import partial
from utils import ImagePool
import torch.optim as optimizer


class ResnetBlock(nn.Module):

    def __init__(self, c, norm_layer):
        super(ResnetBlock, self).__init__()

        conv_block = []

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(c, c, kernel_size=3, stride=1),
                       norm_layer(c),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(c, c, kernel_size=3, stride=1),
                       norm_layer(c)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):

        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):

    def __init__(self, in_c, out_c, norm_layer, ngf, blocks=9):
        super(ResnetGenerator, self).__init__()
        # self.conv1 = nn.Conv2d(3, 5, kernel_size=3)
        self.generator = []
        self.generator += [nn.ReflectionPad2d(3),
                           nn.Conv2d(in_c, ngf, kernel_size=7, stride=1),
                           norm_layer(ngf),
                           nn.ReLU(inplace=True)]

        down_layers = 2
        down_last_c = 0
        for i in range(down_layers):
            mul_pre = 2 ** i
            mul_cur = 2 ** (i+1)
            self.generator += [nn.Conv2d(ngf * mul_pre, ngf * mul_cur, kernel_size=3,
                                         stride=2, padding=1),
                               norm_layer(ngf * mul_cur),
                               nn.ReLU(inplace=True)]
            if i == down_layers - 1:
                down_last_c = mul_cur

        for i in range(blocks):
            self.generator += [ResnetBlock(down_last_c, norm_layer)]

        up_layers = 2
        for i in range(up_layers):
            pre_mul = 2 ** (2 - i)
            cur_mul = 2 ** (1 - i)
            self.generator += [nn.ConvTranspose2d(ngf * pre_mul, ngf * cur_mul, kernel_size=3,
                                                  stride=2, padding=1, output_padding=1),
                               norm_layer(ngf * cur_mul),
                               nn.ReLU(inplace=True)]

        self.generator += [nn.ReflectionPad2d(3),
                           nn.Conv2d(ngf, out_c, kernel_size=7, stride=1),
                           nn.Tanh()]

        self.model = nn.Sequential(*self.generator)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):

    def __init__(self, in_c, layers, norm_layer, ndf=64):
        super(PatchDiscriminator, self).__init__()

        model = []

        kernel_size = 4
        model += [nn.Conv2d(in_c, ndf, kernel_size=kernel_size, padding=1, stride=2),
                       nn.LeakyReLU(0.2, inplace=True)]

        for i in range(1, layers):
            pre_mul = 2 ** (i - 1)
            cur_mul = 2 ** i

            model += [nn.Conv2d(ndf*pre_mul, ndf*cur_mul, kernel_size=kernel_size,
                                     padding=1, stride=2),
                           norm_layer(ndf*cur_mul),
                           nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf*cur_mul, ndf*cur_mul*2, kernel_size=kernel_size,
                                 padding=1, stride=1),
                       norm_layer(ndf*cur_mul*2),
                       nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf*cur_mul*2, 1, kernel_size=kernel_size, padding=1, stride=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):  # 输入的x是(batch, 3, 256, 256)
        return self.model(x)


class CycleGAN(nn.Module):

    def __init__(self, opt, is_train=False):
        super(CycleGAN, self).__init__()

        # 目前只支持单一的GPU
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])
                                   if torch.cuda.is_available() else 'cpu')
        print(f'using device: {self.device}')
        self.is_train = is_train

        if opt.norm == "Instance Norm":
            self.norm = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif opt.norm == "Batch Norm":
            self.norm = partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        else:
            raise NotImplementedError(f'Normalization Layer {self.norm} is not recognized.')

        self.netG_A = define_G(opt.in_c, opt.out_c, self.norm, opt.ngf).to(self.device)
        self.netG_B = define_G(opt.out_c, opt.in_c, self.norm, opt.ngf).to(self.device)

        layer = 3
        if is_train:
            self.netD_A = define_D(opt.in_c, layer, self.norm, opt.ndf).to(self.device)
            self.netD_B = define_D(opt.out_c, layer, self.norm, opt.ndf).to(self.device)

        if is_train:
            if opt.lambda_identity > 0.0:
                assert (opt.in_c == opt.out_c), "in_c and out_c must be equal when using the Identity loss"

            if self.gan_mode == 'lsgan':
                self.criterionGAN = nn.MSELoss()
            elif self.gan_mode == 'vanilla':
                self.criterionGAN = nn.BCEWithLogitsLoss()
            else:
                raise NotImplementedError('gan loss must be vanilla or lsgan!')

            self.criterionCycle = nn.L1Loss()
            self.criterionIdentity = nn.L1Loss()

            self.fake_A_pool = ImagePool(opt.pool_size)  # 缓存图片的类
            self.fake_B_pool = ImagePool(opt.pool_size)

            self.optimizers = []
            self.optimizer_G = optimizer.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                              lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = optimizer.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                              lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.schedulers = [get_scheduler(optim, opt) for optim in self.optimizers]

    def set_up(self, opt):
        if self.is_train and opt.continue_train:
            # TODO: 设置中断后再训练的功能
            pass
        if opt.verbose:
            # TODO: 打印网络的结构和参数的总量
            pass

    def adjust_learning_rate(self):
        pre_lr = self.optimizers[0].param_groups[0]['lr']

        for scheduler in self.schedulers:
            scheduler.step()

        cur_lr = self.optimizers[0].param_groups[0]['lr']
        print('update the learning rate from %.7f to %.7f' % (pre_lr, cur_lr))

    def set_input(self, inp):

        self.real_A = inp['A'].to(self.device)
        self.real_B = inp['B'].to(self.device)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def set_requires_grad(self, nets, requires_grad):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def backward_G(self):
        # 计算self.real_A与self.rec_A，self.real_B与self.rec_B的cycle损失，
        # 计算self.fake_B与标签为1的grand_truth, self.fake_A与标签为1的ground_truth之间的GAN损失
        # 如果self.opt.lambda_identity大于0，计算self.fake_B与self.real_A, self.fake_A与self.real_B的identity损失
        pass

    def backward_D_A(self):
        pass

    def backward_D_B(self):
        pass

    def optimizer_parameters(self):

        # 生成fake_A fake_B
        self.forward()

        # 训练生成器G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # 训练判别器D
        self.set_requires_grad([self.netD_A, self.netD_B], True)  # 这里只需要设置D_A的梯度就好，G的梯度可以通过detach()截断
        self.optimizer_D.zero_grad()
        self.backward_D_A()  # 这里官方代码使用了两个backward，但是实际上只是用一个就可以了
        self.backward_D_B()
        self.optimizer_D.step()


def get_scheduler(optim, opt):

    def adjust_lr(epoch):
        lr = 1.0 - max(0.0, (epoch + 1 - opt.n_epoch) / opt.n_epoch_decay)
        return lr

    scheduler = optimizer.lr_scheduler.LambdaLR(optim, lr_lambda=adjust_lr)
    return scheduler


def define_G(in_c, out_c, norm_layer, ngf):

    model = ResnetGenerator(in_c, out_c, norm_layer, ngf)
    init_weight(model)
    return model


def define_D(in_c, layer, norm_layer, ndf):

    model = PatchDiscriminator(in_c, layer, norm_layer, ndf)
    init_weight(model)
    return model


# 权重初始化
def init_weight(net):

    def init_func(m):
        # print(m.__class__.__name__)
        module_name = m.__class__.__name__

        if hasattr(m, 'weight') and (module_name.find('Conv') != -1 or module_name.find('Linear') != -1):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias'):
                nn.init.zeros_(m.bias)
        elif module_name.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weiht, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    print('initialize the network')
    net.apply(init_func)
