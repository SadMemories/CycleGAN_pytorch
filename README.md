# CycleGAN_pytorch
CycleGAN复现

5.24: 在原来模型的基础上，加了```torch.backends.cudnn.benchmark = True```这条命令，加入这条命令之后会令模型的训练速度加快，但是显存也会增加。
