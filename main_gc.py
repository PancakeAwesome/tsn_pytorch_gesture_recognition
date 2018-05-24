import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import TSN
from transforms import *
# parser是在opts.py中定义的关于读取命令行参数的对象
from opts import parser

best_prec1 = 0

def main():
    # main函数主要包含导入模型、数据准备、训练三个部分
    global args, best_prec1
    args = parser.parse_args()

    # 导入数据集
    # UCF101数据集包含13,320个视频剪辑，其中共101类动作。HMDB51数据集是来自各种来源的大量现实视频的集合，比如：电影和网络视频，数据集包含来自51个动作分类的6,766个视频剪辑。
    if args.dataset == 'ucf101':
        num_clsass = 101
    elif args.dataset == 'hmdb51':
        num_clsass = 51
    elif args.dataset == 'kinetics':
        num_clsass == 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    # part1:模型导入

    # 输入包含分
    # 类的类别数：num_class；
    # args.num_segments表示把一个video分成多少份，对应论文中的K，默认K=3；
    # 采用哪种输入：args.modality，比如RGB表示常规图像，Flow表示optical flow等；
    # 采用哪种模型：args.arch，比如resnet101，BNInception等；
    # 不同输入snippet的融合方式：args.consensus_type，比如avg等；
    # dropout参数：args.dropout。
    model = TSN(num_class, args.num_segments, args.modality, base_model = args.arch, consensus_type = args.consensus_type, dropout = args.dropout, partial_bn = not args.no_partialbn)
    # 交叉模式预训练技术：利用RGB模型初始化时间网络。
    # 根据不同输入类型改变网络第一层的结构
    # setattr(),getattr()

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    # 修改RGB模型第一个卷积层的权重来处理光流场的输入
    policies = model.get_optim_policies()

    # 设置多GPU训练模型
    model = torch.nn.DataParaller(model, device_ids = args.gpus).cuda()

    # 用来设置是否从断点处继续训练，比如原来训练模型训到一半停止了，希望继续从保存的最新epoch开始训练，因此args.resume要么是默认的None，要么就是你保存的模型文件（.pth）的路径
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            # 导入已训练好的模型
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # 完成导入模型的参数初始化model这个网络的过程
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # part2:数据导入
    if args.modality != 'RGBDiff':
        # 数据预处理
        # 如果是rgb or flow做数据归一化
        normalize = GroupNormalize(input_mean, input_std)
    else:
        # 如果是RGBDiff数据不做处理
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    # TSNDataSet类用来处理最原始的数据，返回的是torch.utils.data.Dataset类型
    # PyTorch中自定义的数据读取类都要继承torch.utils.data.Dataset这个基类
    # (self, root_path, list_file,
                 # num_segments=3, new_length=1, modality='RGB',
                 # image_tmpl='img_{:05d}.jpg', transform=None,
                 # force_grayscale=False, random_shift=True, test_mode=False)
    # torch.utils.data.DataLoader类是将batch size个数据和标签分别封装成一个Tensor，从而组成一个长度为2的list。对于torch.utils.data.DataLoader类而言，最重要的输入就是TSNDataSet类的初始化结果，其他如batch size和shuffle参数是常用的。通过这两个类读取和封装数据，后续再转为Variable就能作为模型的输入了
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # 导入测试数据
    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # part3：训练模型
    # 包括定义损失函数、优化函数、一些超参数设置等
     
    # 定义损失函数
    if args.loss_type == 'nll':
        # pytorch中CrossEntropyLoss是通过两个步骤计算出来的，第一步是计算log softmax，第二步是计算cross entropy（或者说是negative log likehood）
        # CrossEntropyLoss不需要在网络的最后一层添加softmax和log层，直接输出全连接层即可。而NLLLoss则需要在定义网络的时候在最后一层添加softmax和log层
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    # policies是网络第一层的信息
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    # 定义优化函数
    optimizer = torch.optim.SGD(policies, args.lr, momentum = args.momentum, weight_decay = args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # 调整学习率
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # 切换到训练模式
    # 注意：调用的model重写的train方法以达到冻结bn层参数的目的
    model.train()

    end = time.Time()
    for i, (input, target) in enumerate(train_loader):
        # 计算数据的导入时间
        # 当执行enumerate(train_loader)的时候，是先调用DataLoader类的__iter__方法，该方法里面再调用DataLoaderIter类的初始化操作__init__
        # 而当执行for循环操作时，调用DataLoaderIter类的__next__方法，在该方法中通过self.collate_fn接口读取self.dataset数据时就会调用TSNDataSet类的__getitem__方法，从而完成数据的迭代读取
        data_time.update(time.time() - end)

        # 读取到数据后就将数据从Tensor转换成Variable格式，然后执行模型的前向计算
        # 如果想在CUDA上进行计算，需要将操作对象放在GPU内存中。
        target = target.cuda(async = True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # 计算输出
        # output:batch size*class维度
        # 自动调用模型中自定义的forward函数
        output = model(input_var)
        loss = criterion(output, target_var)

        # 计算准确度和视频的总损失
        prec1, prec5 = accuracy(output.data, target, topk = (1, 5))
        losses.update(loss.data[0], input.size(0))
        prec1.update(prec1.data[0], input.size(0))
        prec5.update(prec5.data[0], input.size(0))

        # 对所有的参数的梯度缓冲区进行归零
        optimizer.zero_grad()

        # autograd.Variable 这是这个包中最核心的类。 
        # 它包装了一个Tensor，并且几乎支持所有的定义在其上的操作。一旦完成了你的运算，你可以调用 .backward()来自动计算出所有的梯度。
        # 反向传播，自动计算梯度
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        # 执行参数更新
        optimizer.step()

        # 计算消耗的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def validate(val_loader, model, criterion, iter, logger = None):
    # 验证模型操作
    # 没有optimizer.zero_grad()、loss.backward()、optimizer.step()等损失回传或梯度更新操作
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    # pytorch对模型验证
    # 将模型设置为evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async = True)
        # variable = Variable(tensor, requires_grad=True)
        # volatile=True相当于requires_grad=False
        input_var = torch.autograd.Variable(input, volatile = True)
        target_var = torch.autograd.Variable(target, volatile = True)

        # 计算输出
        output= model(input_var)
        loss = criterion(output, target_var)

        # 计算准确度和视频的损失
        prec1, prec5 = accuracy(output.data, target, topk = (1, 5))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # 计算消耗的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg

def accuracy(output, target, topk = (1, )):
    """Computes the precision@k for the specified values of k
    output是模型预测的结果，尺寸为batch size*num class；target是真实标签，长度为batch size
    """
    maxk = max(topk)
    # 读取batch size值
    batch_size = target.size(0)

    # 调用了PyTorch中Tensor的topk方法
    # 第一个输入maxk表示你要计算的是top maxk的结果；
    # 第二个输入1表示dim，即按行计算（dim=1）；
    # 第三个输入True完整的是largest=True，表示返回的是top maxk个最大值；
    # 第四个输入True完整的是sorted=True，表示返回排序的结果
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # target.view(1, -1).expand_as(pred)先将target的尺寸规范到1*batch size，然后将维度扩充为pred相同的维度
    # 调用eq方法计算两个Tensor矩阵相同元素情况，得到的correct是同等维度的ByteTensor矩阵，1值表示相等，0值表示不相等
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __main__ == 'main':
    main()