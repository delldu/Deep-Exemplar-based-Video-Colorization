# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
import random
import torch
import todos

import video_color

from tqdm import tqdm

if __name__ == "__main__":
    model, device = video_color.get_color_model()

    N = 100
    B, C, H, W = 1, 3, model.max_h, model.max_w

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        h = random.randint(0, 32)
        w = random.randint(0, 32)
        x = torch.randn(B, C, H + h, W + w)
        # print("x: ", x.size())
        c = torch.randn(B, C, 512, 512)

        start_time = time.time()
        y = todos.model.two_forward(model, device, x, c)
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")
