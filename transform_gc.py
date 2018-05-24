import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch

class GroupScale(object):
    """
    图片缩放器
    resieze,双线性插值"""
    def __init__(self, size, interpolation = Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return (self.worker(img) for img in img_group)
        

class GroupOverSample(object):
    """进行重复采样的crop操作(随机采样和水平翻转)，最终一张图像得到10张crop的结果
    对输入的n张图像都做torchvision.transforms.Scale操作，也就是resize到指定尺寸
    Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, crop_size, scale_size = None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):
        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        # 返回的offsets是一个长度为5的列表，每个值都是一个tuple，其中前4个是四个点坐标，最后一个是中心点坐标，目的是以这5个点为左上角坐标时可以在原图的四个角和中心部分crop出指定尺寸的图
        # fill_fix_offset是类的静态方法
        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                # 按照crop_w*crop_h的大小去crop原图像，这里采用的是224*224
                crop = img.crop((ow, o_h, ow + crop_w, o_h + crop_h))
                normal_group.append(crop)
                # 对crop得到的图像做左右翻转。最后把未翻转的和翻转后的列表合并，这样一张输入图像就可以得到10张输出了（5张crop，5张crop加翻转）
                # 采用角裁剪（corner cropping）
                # 这就是论文中说的corner crop，而且是4个corner和1个center。
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)

        return oversample_group


class GroupMultiScanleCrop(object):
    """docstring for GroupMultiScanleCrop"""
    def __init__(self, input_size, scales = None, max_distort = 1, fix_crop = True, more_fix_crop = True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret
        

        