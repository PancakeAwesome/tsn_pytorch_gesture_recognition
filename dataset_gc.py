import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    """
    一个视频的对象
    """
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class TSNDataSet(data.Dataset):
    """数据导入类
    自定义数据读取相关类的时候需要继承torch.utils.data.Dataset这个基类
    通过重写初始化函数__init__和__getitem__方法来读取数据
    """
    def __init__(self, root_path, list_file, num_segments = 3, new_length = 1, modality = 'RGB', image_tmpl = 'img_{:05d}.jpg', transform = None, force_grayscale = False, random_shift = True, test_mode = False):
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        # diff的通道：6
        # rgb的通道：3
        # flow的通道：5
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        # 数据读取类TSNDataSet返回两个值，第一个值是Tensor类型的数据，第二个值是int型的标签，而torch.utils.data.DataLoader类是将batch size个数据和标签分别封装成一个Tensor，从而组成一个长度为2的list
        self._parse_list()

    def __getitem__(self, index):
        """
        训练时枚举每个迭代时候会调用__getitem__这个魔法方法
        对于RGB输入，这个Tensor的尺寸是(3*self.num_segments,224,224)，其中3表示3通道彩色；对于Flow输入，这个Tensor的尺寸是(self.num_segments*2*self.new_length,224,224)，其中第一维默认是30(3*2*5)
        newlength 是论文temperal structural 的理念，每个snippet包含一个RGB图像和这张图片以及它后面的4帧图片的dx,dy光流
        在训练的时候：对于RGB输入，这个Tensor的尺寸是(3*self.num_segments,224,224)，其中3表示3通道彩色；对于Flow输入，这个Tensor的尺寸是(self.num_segments*2*self.new_length,224,224)，其中第一维默认是30(3*2*5)
        """
        # 魔法方法
        # 得到的record就是一帧图像的信息，index是随机的，这个和前面数据读取中的shuffle参数对应
        # Called to implement evaluation of self[key]
        record = self.video_list[index]

        if not self.test_mode:
            # training mode
            # 在训练的时候，self.test_mode是False，所以执行if语句，另外self.random_shift默认是True
            segment_indices = self._sample_indices(record) if self.random_shift else self _get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        # 通过indices获取真正的图片数据
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            # 对于RGB而言，self.new_length是1，这样images的长度就是indices的长度；对于Flow而言，self.new_length是5，这样images的长度就是indices的长度乘以(5*2)
            for i in range(self.new_length):
                # 对于RGB或RGBDiff数据，返回的seg_imgs是一个长度为1的列表
                # 对于Flow数据，返回的seg_imgs是一个长度为2的列表，然后将读取到的图像数据合并到images这个列表中
                seg_imgs = self._load_image(record.path, p)
                # 将图片放到图片列表中
                images.extend(seg_imgs)
                # 会使用包含这张图片以及之后的四张图片的flow作为一个snippet的光流输入
                if p < record.num_frames:
                    p += 1

        # 调用dataset的参数：transform方法
        # 做数据预处理
        # 一般为torch自带的方法
        # 将list类型的images封装成了Tensor
        process_data = self.transform(images)
        return process_data, record.label

    def _load_image(self, directory, idx):
        """
        采用PIL库的Image模块来读取图像数据
        """
        # 不同格式的输入数据
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            # idx是帧的indice
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx)).convert('RGB'))]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            # 返回dx和dy方向的flow
            return [x_img, y_img]

    def _get_test_indices(self, record):
        """获得测试集上的index
        将输入video按照相等帧数距离分成self.num_segments份，最终返回的offsets就是长度为self.num_segments的numpy array，表示从输入video中取哪些帧作为模型的输入
        """
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def _sample_indices(self, record):
        """
        获得训练时，数据集中的随机数据index，分为num_segments份
        :param record: VideoRecord
        :return: list
        假设average_duration:10,num_segments:3
        return [0+rand1, 10+rand2, 20+rand3]
        """
        # average_duration表示某个视频分成self.num_segments份的时候每一份包含多少帧图像
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        # 获得验证集上的index
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1
        
    def _parse_list(self):
        """
        self.list_file就是训练或测试的列表文件（.txt文件），里面包含三列内容，用空格键分隔，第一列是video名，第二列是video的帧数，第三列是video的标签
        self.video_list的内容就是一个长度为训练数据数量的列表，列表中的每个值都是VideoRecord对象，该对象包含一个列表和3个属性，列表长度为3，分别是帧路径、该视频包含多少帧、帧标签，同样这三者也是三个属性的值
        """
        # e.g:<ucf101_rgb_train_list>
        self.video_list = [VideoRecord(x.strip().split('') for x in open(self.list_file))]