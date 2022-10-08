"""Video Color Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import redos
import todos

from . import data, color

import pdb


def get_model():
    """Create model.
    pre-trained model video_color.pth comes from
    https://github.com/delldu/TorchScript.git/video_color
    """

    model_path = "models/video_color.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = color.VideoColor()
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/video_color.torch"):
        model.save("output/video_color.torch")

    return model, device


def model_forward(model, device, input_tensor, reference_tensor, multi_times=1):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    output_tensor = todos.model.two_forward(model, device, input_tensor, reference_tensor)

    return output_tensor[:, :, 0:H, 0:W]


def video_predict(input_file, color_file, output_file):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # Reference
    reference_tensor = todos.data.load_tensor(color_file)

    print(f"  Color {input_file} with {color_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def color_video_frame(no, datax):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(datax)

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_rgb = model_forward(model, device, input_tensor, reference_tensor)

        # save the frames
        temp_output_file = "{}/{:06d}.png".format(output_dir, no + 1)
        todos.data.save_tensor(output_rgb, temp_output_file)

    video.forward(callback=color_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)

    os.removedirs(output_dir)
    todos.model.reset_device()

    return True
