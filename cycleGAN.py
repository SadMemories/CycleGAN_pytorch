import torch
import torch.nn as nn
import collections
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
            self.generator += [ResnetBlock(down_last_c * ngf, norm_layer)]

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

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.visual_A = ['real_A', 'fake_B', 'rec_A']
        self.visual_B = ['real_B', 'fake_A', 'rec_B']

        layer = 3
        if is_train:
            self.netD_A = define_D(opt.in_c, layer, self.norm, opt.ndf).to(self.device)
            self.netD_B = define_D(opt.out_c, layer, self.norm, opt.ndf).to(self.device)

        if is_train:
            self.lambda_identity = 0.0
            self.lambda_A = opt.lambda_A
            self.lambda_B = opt.lambda_B
            if opt.lambda_identity > 0.0:
                assert (opt.in_c == opt.out_c), "in_c and out_c must be equal when using the Identity loss"
                self.lambda_identity = opt.lambda_identity
                self.visual_A.append('idt_A')
                self.visual_B.append('idt_B')

            self.visual_names = self.visual_A + self.visual_B  # 在训练时显示的图片

            if opt.gan_mode == 'lsgan':
                self.criterionGAN = nn.MSELoss()
            elif opt.gan_mode == 'vanilla':
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

    def get_current_visuals(self):
        visuals = collections.OrderedDict()

        for name in self.visual_names:
            if isinstance(name, str):
                visuals[name] = getattr(self, name)
        return visuals

    def get_current_losses(self):
        losses = collections.OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses[name] = float(getattr(self, 'loss_'+name))
        return losses

    def backward_G(self):
        # 计算self.real_A与self.rec_A，self.real_B与self.rec_B的cycle损失，
        # 计算self.fake_B与标签为1的grand_truth, self.fake_A与标签为1的ground_truth之间的GAN损失
        # 如果self.opt.lambda_identity大于0，计算self.fake_B与self.real_A, self.fake_A与self.real_B的identity损失
        if self.lambda_identity > 0.0:
            self.idt_A = self.netG_A(self.real_A)
            self.idt_B = self.netG_B(self.real_B)

            self.loss_idt_A = self.criterionIdentity(self.real_A, self.idt_A) * self.lambda_identity * self.lambda_A
            self.loss_idt_B = self.criterionIdentity(self.real_B, self.idt_B) * self.lambda_identity * self.lambda_B
        else:
            self.loss_idt_A = 0.0
            self.loss_idt_B = 0.0

        self.loss_cycle_A = self.criterionCycle(self.real_A, self.rec_A) * self.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.real_B, self.rec_B) * self.lambda_B

        d_a = self.netD_A(self.fake_B)
        d_b = self.netD_B(self.fake_A)

        gt_label_a = torch.tensor(1.0).expand_as(d_a).to(self.device)
        gt_label_b = torch.tensor(1.0).expand_as(d_b).to(self.device)

        self.loss_G_A = self.criterionGAN(d_a, gt_label_a)
        self.loss_G_B = self.criterionGAN(d_b, gt_label_b)

        loss_g = self.loss_idt_A + self.loss_idt_B + self.loss_cycle_A \
                 + self.loss_cycle_B + self.loss_G_A + self.loss_G_B
        loss_g.backward()

    def backward_D_basic(self, D, fake_imgs, real_img):

        pred_true = D(real_img)
        gt_label_true = torch.tensor(1.0).expand_as(pred_true).to(self.device)
        loss_true = self.criterionGAN(pred_true, gt_label_true)

        pred_fake = D(fake_imgs.detach())
        gt_label_false = torch.tensor(0.0).expand_as(pred_fake).to(self.device)
        loss_false = self.criterionGAN(pred_fake, gt_label_false)

        loss_d = (loss_false + loss_true) * 0.5
        loss_d.backward()
        return loss_d

    def backward_D_A(self):
        # 更新判别器D_A
        fake_b_pool = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, fake_b_pool, self.real_B)

    def backward_D_B(self):
        # 更新判别器D_B
        fake_a_pool = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, fake_a_pool, self.real_A)

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
        lr = 1.0 - max(0.0, (epoch + 1 - opt.n_epoch)) / float(opt.n_epoch_decay)
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

    print(f'initialize the {net.__class__.__name__} network')
    net.apply(init_func)
