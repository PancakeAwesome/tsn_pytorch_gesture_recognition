from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

class TSN(nn.model):
    """"tsn模型类
    # 输入包含分
    # 类的类别数：num_class；
    # 修改网络时第一层卷积层snippet的卷积核的参数：new_length,rgb:1,diff:6,flow:5(采用当前帧以及之后4帧图像的两个方向的flow)
    # args.num_segments表示把一个video分成多少份，对应论文中的K，默认K=3；
    # 采用哪种输入：modality，比如RGB表示常规图像，Flow表示optical flow等；
    # 采用哪种模型：base_model，比如resnet101，BNInception等；
    # 不同输入snippet的融合方式：consensus_type，默认为avg等；
    # dropout参数：dropout
    """
    def __init__(self, num_class, num_segements, modality, base_model = 'resnet101', new_length = None, consensus_type = 'avg', before_softmax = True, dropout = 0.8, crop_num = 1, partial_bn = True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type# 各个snippet之间的融合方式:段共识函数,评估 g 的三种形式：（1）最大池化；（2）平均池化；（3）加权平均
        
        # 段共识函数在Softmax归一化之前
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
        Initializing TSN with base model: {}.
        TSN Configurations:
            input_modality:     {}
            num_segments:       {}
            new_length:         {}
            consensus_module:   {}
            dropout_ratio:      {}
                """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        # 导入模型
        # 论文中导入的是bninception
        # BNInception类，定义在tf_model_zoo文件夹下的bninception文件夹下的pytorch_load.py中
        self._prepare_base_model(base_model)

        # 导入模型
        feature_dim = self._prepare_tsn(num_class)
        # feature_dim是网络最后一层的输入feature map的channel数

        # 迁移学习的第一种方式：利用conv网络初始化参数
        # 交叉模式预训练技术：利用RGB模型初始化时间网络
        # 如果你的输入数据是optical flow或RGBDiff，那么还会对网络结构做修改，分别调用_construct_flow_model方法和_construct_diff_model方法来实现的，主要差别在第一个卷积层，因为该层的输入channel依据不同的输入类型而变化
        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            # 修改网络结构
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        else self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        # 段共识函数
        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        # 在用预训练模型初始化后，冻结所有Batch Normalization层的均值和方差参数，但第一个标准化层除外。由于光流的分布和RGB图像的分布不同，第一个卷积层的激活值将有不同的分布，于是，我们需要重新估计的均值和方差，称这种策略为部分BN
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            # 一张图片18个channel，每个通道3种差值
            # 并将原来的rgb 权重保存---concat在一起
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _prepare_tsn(self, num_class):
        # feature_dim是基础网络最后一层的输入feature map的channel数
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        # 接下来如果有dropout层，那么添加一个dropout层后连一个全连接层，否则就直接连一个全连接层
        if self.dropout == 0:
            # setattr是torch.nn.Module类的一个方法，用来为输入的某个属性赋值，一般可以用来修改网络结构
            # 输入包含3个值，分别是基础网络，要赋值的属性名，要赋的值
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p = self.dropout))
            # 添加一个全连接层
            self.new_fc = nn.Linear(feature_dim, num_class)

        # 最后对全连接层的参数（weight）做一个0均值且指定标准差的初始化操作，参数（bias）初始化为0
        std = 0.001
        if self.new_fc is None:
            # normals是torch的一个初始化方法
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            # 输入是光流场
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            # 输入是RGB Diff
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length 
        elif base_model == 'BNInception':
            import tf_model_zoo
            # 定义在tf_model_zoo文件夹下的bninception文件夹下的pytorch_load.py中
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        # 找到第一层卷积层
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        # params 包含weight(kernel)和bias
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        # e.g:(64,3,7,7) => (64,10,7,7)
        # new_length = 5（l=5,有6张图片）
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernel = params[0].data.mean(dim = 1, keepdim = True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        # new_conv.weight.data = new_kernels是赋值过程
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def get_optim_policies(self):
        # 定义模型中可学习的参数，用来作为参数更新函数的第一个参数
        # 修改RGB模型第一个卷积层的权重来处理光流场的输入
        # 返回第一层bn层参数
        # 用作之后的partial bn
        # 冻结
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.base_model.modules():
        # for m in self.modules():?
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    @property
    def crop_size(self):
        return self.input_size
    
    @property
    def scale_size(self):
        # 只读属性
        # resieze到256
        return self.input_size * 256 // 224

    def train(self, mode = True):
        """
        Override the default train() to freeze the BN parameters
        重写模型的train()方法
        在用预训练模型初始化后，冻结所有Batch Normalization层的均值和方差参数，但第一个标准化层除外。由于光流的分布和RGB图像的分布不同，第一个卷积层的激活值将有不同的分布，于是，我们需要重新估计均值和方差，称这种策略为部分BN
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown conv layer update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable    

    def _get_diff(self, input, keep_rgb=False):
        # 获得RGBDiff的输入数据
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            # 获得RGBDiff的输入数据
            input = self._get_diff(input)

        # 调用basemodel的forward方法
        # 得到前向输出结果
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        # 连接各段的输出
        output = self.consensus(base_out)
        # tensor去除维度大小为1的维度
        return output.squeeze(1)

    def get_augmentation(self):
        # 数据增强
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])