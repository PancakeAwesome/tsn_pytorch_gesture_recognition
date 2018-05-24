from .layer_factory import get_basic_layer, parse_expr
from torch import nn
import torch
import torch.utils.model_zoo as model_zoo
import yaml

class BNInception(nn.Module):
    """搭建bninception网络"""
    def __init__(self, model_path = 'tf_model_zoo/bninception/bn_inception.yaml', num_class = 101, weight_url = 'https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth'):
        super(BNInception, self).__init__()

        # 读进配置好的网络结构（.yml格式），返回的manifest是长度为3的字典，和.yml文件内容对应
        mainfest = yaml.load(open(model_path))

        # layers是layer的数组
        # layer包括数据流关系、名称和结构参数等信息
        layers = mainfest['layers']

        self._channel_dict = dict()

        self._op_list = list()
        # 构建网络
        for l in layers:
            # 获得某一层layer的数据流关系expr
            # 输出<=op操作<=输入
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat':
                # 封装了一层layer操作
                # module是layer操作函数
                id, out_name, module, out_channel, in_name = get_basic_layer(l, 3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]], conv_bias = True)

                self._channel_dict[out_name] = out_channel
                # setattr(self, id, module)是将得到的层写入self的指定属性中，就是搭建层的过程
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            else:
                # concat层
                # 将各种bn层的输出concat到一起
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel

        # 过提供的.pth文件的url地址来下载指定的.pth文件，在PyTorch中.pth文件就是模型的参数文件，如果你已经有合适的模型了且不想下载，那么可以通过torch.load(‘the/path/of/.pth’)导入
        # 在PyTorch中.pth文件就是模型的参数文件
        self.load_state_dict(torch.utils.model_zoo.load_url(weight_url))
        # 因此不想下载的话可以用checkpoint=torch.load('the/path/of/.pth')和self.load_state_dict(checkpoint)两行代替self.load_state_dict(torch.utils.model_zoo.load_url(weight_url))。
    
    def forward(self, input):
        # bninception的前向传播函数
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        def get_hook(name):

            def hook(m, grad_in, grad_out):
                print(name, grad_out[0].data.abs().mean())

            return hook
        for op in self._op_list:
            if op[1] != 'Concat' and op[1] != 'InnerProduct':
                data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
                # getattr(self, op[0]).register_backward_hook(get_hook(op[0]))
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].size())
                    raise
        return data_dict[self._op_list[-1][2]]

class InceptionV3(BNInception):
    def __init__(self, model_path='model_zoo/bninception/inceptionv3.yaml', num_classes=101,
                 weight_url='https://yjxiong.blob.core.windows.net/models/inceptionv3-cuhk-0e09b300b493bc74c.pth'):
        super(InceptionV3, self).__init__(model_path=model_path, weight_url=weight_url, num_classes=num_classes)

                
