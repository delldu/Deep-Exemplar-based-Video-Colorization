# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:52:14 CST
# ***
# ************************************************************************************/
#

# The following code comes from
# https://github.com/zhangmozhe/video-colorization.git
# Thanks the authors
# Hi Guys, I love you !!!

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data

import pdb

REFERENCE_RESIZE = 512


def feature_normalize(feature_in):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + 1e-6
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm


def vgg_preprocess(tensor):
    # input is RGB tensor which ranges in [0,1.0]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    tensor_bgr_ml = tensor_bgr - torch.tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    del tensor_bgr

    return tensor_bgr_ml * 255


class VGG19ConvModel(nn.Module):
    """
    NOTE: no need to pre-process the input; input tensor should range in [0,1]
    """

    def __init__(self):
        super(VGG19ConvModel, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu_x = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        NOTE: input tensor should range in [0,1], is Lab
        """
        # x = lab2rgb(x)
        x = vgg_preprocess(x)
        out = {}

        out["r11"] = self.relu_x(self.conv1_1(x))
        out["r12"] = self.relu_x(self.conv1_2(out["r11"]))
        out["p1"] = self.pool1(out["r12"])
        out["r21"] = self.relu_x(self.conv2_1(out["p1"]))
        out["r22"] = self.relu_x(self.conv2_2(out["r21"]))
        out["p2"] = self.pool2(out["r22"])
        out["r31"] = self.relu_x(self.conv3_1(out["p2"]))
        out["r32"] = self.relu_x(self.conv3_2(out["r31"]))
        out["r33"] = self.relu_x(self.conv3_3(out["r32"]))
        out["r34"] = self.relu_x(self.conv3_4(out["r33"]))
        out["p3"] = self.pool3(out["r34"])
        out["r41"] = self.relu_x(self.conv4_1(out["p3"]))
        out["r42"] = self.relu_x(self.conv4_2(out["r41"]))
        out["r43"] = self.relu_x(self.conv4_3(out["r42"]))
        out["r44"] = self.relu_x(self.conv4_4(out["r43"]))
        out["p4"] = self.pool4(out["r44"])
        out["r51"] = self.relu_x(self.conv5_1(out["p4"]))
        out["r52"] = self.relu_x(self.conv5_2(out["r51"]))
        out["r53"] = self.relu_x(self.conv5_3(out["r52"]))
        out["r54"] = self.relu_x(self.conv5_4(out["r53"]))
        out["p5"] = self.pool5(out["r54"])

        # x --[1, 3, 320, 512]
        # r12 [1, 64, 320, 512]
        # r22 [1, 128, 160, 256]
        # r32 [1, 256, 80, 128]
        # r42 [1, 512, 40, 64]
        # r52 [1, 512, 20, 32]

        return out["r22"], out["r32"], out["r42"], out["r52"]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out


# Global Reference Warp Model
class GlobalWarpModel(nn.Module):
    def __init__(self):
        super(GlobalWarpModel, self).__init__()
        self.feature_channel = 64
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256

        # 44*44
        self.layer2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            # Down Sample !!!
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=2),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )
        self.layer3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )

        # 22*22->44*44
        self.layer4_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # 11*11->44*44
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        self.layer = nn.Sequential(
            ResidualBlock(
                self.feature_channel * 4,
                self.feature_channel * 4,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            ResidualBlock(
                self.feature_channel * 4,
                self.feature_channel * 4,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            ResidualBlock(
                self.feature_channel * 4,
                self.feature_channel * 4,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.theta = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # self.theta == Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))

        self.phi = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # self.phi -- Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        self.upsampling = nn.Upsample(scale_factor=4, mode="nearest")

    def forward(self, IB_lab, A_f2, A_f3, A_f4, A_f5, B_f2, B_f3, B_f4, B_f5):
        # IB_lab.size() -- [1, 3, 512, 512]
        A_f2 = feature_normalize(A_f2)
        A_f3 = feature_normalize(A_f3)
        A_f4 = feature_normalize(A_f4)
        A_f5 = feature_normalize(A_f5)

        B_f2 = feature_normalize(B_f2)
        B_f3 = feature_normalize(B_f3)
        B_f4 = feature_normalize(B_f4)
        B_f5 = feature_normalize(B_f5)

        batch_size = IB_lab.shape[0]
        channel = IB_lab.shape[1]
        feature_height = IB_lab.shape[2] // 4
        feature_width = IB_lab.shape[3] // 4

        # scale feature size to 44*44
        A_f2 = self.layer2_1(A_f2)
        A_f3 = self.layer3_1(A_f3)

        A_f4 = self.layer4_1(A_f4)
        A_f5 = self.layer5_1(A_f5)

        B_f2 = self.layer2_1(B_f2)
        B_f3 = self.layer3_1(B_f3)
        B_f4 = self.layer4_1(B_f4)
        B_f5 = self.layer5_1(B_f5)

        # Using fixed input shape [1, 3, 512, 512], the following code could be removed for ONNX model !!!
        # if A_f5.shape[2] != A_f2.shape[2] or A_f5.shape[3] != A_f2.shape[3]:
        #     A_f5 = F.pad(A_f5, (0, 0, 1, 1), "replicate")
        #     B_f5 = F.pad(B_f5, (0, 0, 1, 1), "replicate")

        A_features = self.layer(torch.cat((A_f2, A_f3, A_f4, A_f5), dim=1))
        B_features = self.layer(torch.cat((B_f2, B_f3, B_f4, B_f5), dim=1))

        # pairwise cosine similarity
        theta = self.theta(A_features).view(batch_size, self.inter_channels, -1)
        theta = theta - theta.mean(dim=-1, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + 1e-6
        theta = torch.div(theta, theta_norm)
        theta_permute = theta.permute(0, 2, 1)

        phi = self.phi(B_features).view(batch_size, self.inter_channels, -1)
        phi = phi - phi.mean(dim=-1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + 1e-6
        phi = torch.div(phi, phi_norm)
        f = torch.matmul(theta_permute, phi)

        f_similarity = f.unsqueeze_(dim=1)
        similarity_map = torch.max(f_similarity, -1, keepdim=True)[0]
        similarity_map = similarity_map.view(batch_size, 1, feature_height, feature_width)

        # f can be negative
        temperature = 1e-10
        f_WTA = f
        f_WTA = f_WTA / temperature
        f_div_C = F.softmax(f_WTA.squeeze_(), dim=-1)

        # downsample the reference color
        B_lab = F.avg_pool2d(IB_lab, 4)
        B_lab = B_lab.view(batch_size, channel, -1)
        B_lab = B_lab.permute(0, 2, 1)

        # multiply the corr map with color
        global_BA_lab = torch.matmul(f_div_C, B_lab)

        global_BA_lab = global_BA_lab.permute(0, 2, 1).contiguous()
        global_BA_lab = global_BA_lab.view(batch_size, channel, feature_height, feature_width)

        global_BA_lab = self.upsampling(global_BA_lab)
        similarity_map = self.upsampling(similarity_map)

        # global_BA_lab.size(), similarity_map.size()
        # [1, 3, 360, 640], [1, 1, 360, 640]

        return global_BA_lab, similarity_map


class AlignModel(nn.Module):
    def __init__(self, input_chans=7):
        super(AlignModel, self).__init__()
        self.vgg19 = VGG19ConvModel()
        self.warp = GlobalWarpModel()

    def forward(self, input):
        A_rgb = input[:, 0:3, :, :]
        B_rgb = input[:, 3:6, :, :]

        A_lab = data.rgb2lab(A_rgb)
        B_lab = data.rgb2lab(B_rgb)

        A_f2, A_f3, A_f4, A_f5 = self.vgg19(A_rgb)
        B_f2, B_f3, B_f4, B_f5 = self.vgg19(B_rgb)

        global_lab, similarity = self.warp(B_lab, A_f2, A_f3, A_f4, A_f5, B_f2, B_f3, B_f4, B_f5)

        # del B_lab
        # del A_rgb, A_f2, A_f3, A_f4, A_f5
        # del B_rgb, B_f2, B_f3, B_f4, B_f5
        # torch.cuda.empty_cache()
        return torch.cat((A_lab[:, 0:1, :, :], global_lab[:, 1:, :, :], similarity), dim=1)


class ColorModel(nn.Module):
    def __init__(self, input_chans=7):
        super(ColorModel, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(input_chans, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
        )
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_2norm = nn.BatchNorm2d(64, affine=False)
        self.conv1_2norm_ss = nn.Conv2d(64, 64, 1, 2, bias=False, groups=64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv2_2norm_ss = nn.Conv2d(128, 128, 1, 2, bias=False, groups=128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv3_3norm_ss = nn.Conv2d(256, 256, 1, 2, bias=False, groups=256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv6_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv7_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv8_1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv3_3_short = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv9_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv2_2_short = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv10_1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.conv1_2_short = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv10_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv10_ab = nn.Conv2d(128, 2, 1, 1)

        # add self.relux_x
        self.relu_x = nn.ReLU(inplace=True)
        self.relu10_2 = nn.LeakyReLU(0.2, True)

        # replace all deconv with [nearest + conv]
        self.conv8_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 256, 3, 1, 1),
        )
        self.conv9_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, 1, 1),
        )
        self.conv10_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, 3, 1, 1),
        )

        # replace all batchnorm with instancenorm
        self.conv1_2norm = nn.InstanceNorm2d(64)
        self.conv2_2norm = nn.InstanceNorm2d(128)
        self.conv3_3norm = nn.InstanceNorm2d(256)
        self.conv4_3norm = nn.InstanceNorm2d(512)
        self.conv5_3norm = nn.InstanceNorm2d(512)
        self.conv6_3norm = nn.InstanceNorm2d(512)
        self.conv7_3norm = nn.InstanceNorm2d(512)
        self.conv8_3norm = nn.InstanceNorm2d(256)
        self.conv9_2norm = nn.InstanceNorm2d(128)

    def forward(self, x):
        """x: gray image (1 channel), ab(2 channel), ab_err, ba_err"""
        # x.size -- [1, 7, 216, 384] why 7 ? The answer is:
        # x -- torch.cat((IA_l, nonlocal_BA_ab, similarity_map, IA_last_lab), dim=1)
        conv1_1 = self.relu_x(self.conv1_1(x))
        conv1_2 = self.relu_x(self.conv1_2(conv1_1))
        conv1_2norm = self.conv1_2norm(conv1_2)
        conv1_2norm_ss = self.conv1_2norm_ss(conv1_2norm)
        conv2_1 = self.relu_x(self.conv2_1(conv1_2norm_ss))
        conv2_2 = self.relu_x(self.conv2_2(conv2_1))
        conv2_2norm = self.conv2_2norm(conv2_2)
        conv2_2norm_ss = self.conv2_2norm_ss(conv2_2norm)
        conv3_1 = self.relu_x(self.conv3_1(conv2_2norm_ss))
        conv3_2 = self.relu_x(self.conv3_2(conv3_1))
        conv3_3 = self.relu_x(self.conv3_3(conv3_2))
        conv3_3norm = self.conv3_3norm(conv3_3)
        conv3_3norm_ss = self.conv3_3norm_ss(conv3_3norm)
        conv4_1 = self.relu_x(self.conv4_1(conv3_3norm_ss))
        conv4_2 = self.relu_x(self.conv4_2(conv4_1))
        conv4_3 = self.relu_x(self.conv4_3(conv4_2))
        conv4_3norm = self.conv4_3norm(conv4_3)
        conv5_1 = self.relu_x(self.conv5_1(conv4_3norm))
        conv5_2 = self.relu_x(self.conv5_2(conv5_1))
        conv5_3 = self.relu_x(self.conv5_3(conv5_2))
        conv5_3norm = self.conv5_3norm(conv5_3)
        conv6_1 = self.relu_x(self.conv6_1(conv5_3norm))
        conv6_2 = self.relu_x(self.conv6_2(conv6_1))
        conv6_3 = self.relu_x(self.conv6_3(conv6_2))
        conv6_3norm = self.conv6_3norm(conv6_3)
        conv7_1 = self.relu_x(self.conv7_1(conv6_3norm))
        conv7_2 = self.relu_x(self.conv7_2(conv7_1))
        conv7_3 = self.relu_x(self.conv7_3(conv7_2))
        conv7_3norm = self.conv7_3norm(conv7_3)
        conv8_1 = self.conv8_1(conv7_3norm)
        conv3_3_short = self.conv3_3_short(conv3_3norm)
        conv8_1_comb = self.relu_x(conv8_1 + conv3_3_short)
        conv8_2 = self.relu_x(self.conv8_2(conv8_1_comb))
        conv8_3 = self.relu_x(self.conv8_3(conv8_2))
        conv8_3norm = self.conv8_3norm(conv8_3)
        conv9_1 = self.conv9_1(conv8_3norm)
        conv2_2_short = self.conv2_2_short(conv2_2norm)
        conv9_1_comb = self.relu_x(conv9_1 + conv2_2_short)
        conv9_2 = self.relu_x(self.conv9_2(conv9_1_comb))
        conv9_2norm = self.conv9_2norm(conv9_2)
        conv10_1 = self.conv10_1(conv9_2norm)
        conv1_2_short = self.conv1_2_short(conv1_2norm)
        conv10_1_comb = self.relu_x(conv10_1 + conv1_2_short)
        conv10_2 = self.relu10_2(self.conv10_2(conv10_1_comb))
        conv10_ab = self.conv10_ab(conv10_2)

        # del (
        #     conv1_1,
        #     conv1_2,
        #     conv1_2norm,
        #     conv1_2norm_ss,
        # )
        # del conv2_1, conv2_2, conv2_2norm, conv2_2norm_ss
        # del conv1_2_short
        # del conv3_1, conv3_2, conv3_3, conv3_3norm, conv3_3norm_ss
        # del conv4_1, conv4_2, conv4_3, conv4_3norm
        # del conv5_1, conv5_2, conv5_3, conv5_3norm
        # del (
        #     conv6_1,
        #     conv6_2,
        #     conv6_3,
        #     conv6_3norm,
        # )
        # del conv7_1, conv7_2, conv7_3, conv7_3norm
        # del (
        #     conv8_1,
        #     conv3_3_short,
        #     conv8_1_comb,
        #     conv8_2,
        #     conv8_3,
        #     conv8_3norm,
        # )
        # del conv9_1, conv2_2_short, conv9_1_comb, conv9_2, conv9_2norm
        # del conv10_1, conv10_1_comb, conv10_2
        # torch.cuda.empty_cache()

        return torch.tanh(conv10_ab) * 128


class VideoColor(nn.Module):
    def __init__(self):
        super(VideoColor, self).__init__()
        self.align = AlignModel()
        self.color = ColorModel()
        self.RESIZE = REFERENCE_RESIZE  # 512
        self.A_last_lab = torch.zeros((1, 3, self.RESIZE, self.RESIZE))

    def forward_x(self, x, B_rgb):
        """x is video frame: 1x3xHxW, B_rgb is refenence 1x3x512x512"""

        H, W = B_rgb.size(2), B_rgb.size(3)
        if H != self.RESIZE or W != self.RESIZE:
            B_rgb = F.interpolate(B_rgb, size=(self.RESIZE, self.RESIZE), mode="bilinear", align_corners=True)

        A_rgb = F.interpolate(x, size=(self.RESIZE, self.RESIZE), mode="bilinear", align_corners=True)
        A_lab = data.rgb2lab(A_rgb)

        align_input = torch.cat((A_rgb, B_rgb), dim=1)
        align_output = self.align(align_input)
        # align_output -- (a_l, global_ab, similarity)

        color_input = torch.cat(
            (align_output, self.A_last_lab.to(align_output.device)), dim=1
        )  # size() -- [1, 7, 512, 512]
        color_output_ab = self.color(color_input)

        # Update self.A_last_lab
        self.A_last_lab = torch.cat((A_lab[:, 0:1, :, :], color_output_ab), dim=1)

        # Blend
        H, W = x.size(2), x.size(3)
        color_output_ab = F.interpolate(color_output_ab, size=(H, W), mode="bilinear", align_corners=True)

        input_lab = data.rgb2lab(x)
        output_lab = torch.cat((input_lab[:, 0:1, :, :], color_output_ab), dim=1)
        output_rgb = data.lab2rgb(output_lab)

        return output_rgb.clamp(0.0, 1.0)

    def forward(self, x, B_rgb):
        # Define max GPU/CPU memory -- 4G
        max_h = 1024
        max_W = 1024
        multi_times = 2

        # Need Resize ?
        B, C, H, W = x.size()
        if H > max_h or W > max_W:
            s = min(max_h / H, max_W / W)
            SH, SW = int(s * H), int(s * W)
            resize_x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=False)
        else:
            resize_x = x

        # Need Pad ?
        PH, PW = resize_x.size(2), resize_x.size(3)
        if PH % multi_times != 0 or PW % multi_times != 0:
            r_pad = multi_times - (PW % multi_times)
            b_pad = multi_times - (PH % multi_times)
            resize_pad_x = F.pad(resize_x, (0, r_pad, 0, b_pad), mode="replicate")
        else:
            resize_pad_x = resize_x

        y = self.forward_x(resize_pad_x, B_rgb)
        del resize_pad_x, resize_x  # Release memory !!!

        y = y[:, :, 0:PH, 0:PW]  # Remove Pads
        if PH != H or PW != W:
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)

        return y
