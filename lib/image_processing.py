import torch
import cv2
import os
import numpy as np
from tqdm import tqdm

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]

    if width and height:
        return cv2.resize(image, divisible((width, height)),  interpolation=inter)

    if width is None and height is None:
        return cv2.resize(image, divisible((w, h)),  interpolation=inter)

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, divisible(dim), interpolation=inter)

# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss

def gram(input):
    b, c, w, h = input.size()

    x = input.contiguous().view(b * c, w * h)

    G = torch.mm(x, x.T)
    G = torch.clamp(G, -64990.0, 64990.0)
    # normalize by total elements
    result = G.div(b * c * w * h)
    return result

def normalize_input(images):
    return images / 127.5 - 1.0

def denormalize_input(images, dtype=None):
    # Normalize to [-1, 1] => [0, 255]

    images = images * 127.5 + 127.5

    if dtype is not None:
        if isinstance(images, torch.Tensor):
            images = images.type(dtype)
        else:
            images = images.astype(dtype)

    return images

def divisible(dim):
    width, height = dim
    return width - (width % 32), height - (height % 32)


def prepare_images(images):
    images = images.astype(np.float32)
    images = normalize_input(images)
    images = torch.from_numpy(images)
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    images = images.permute(0, 3, 1, 2)
    return images

def calc_mean(data_folder):
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f'Folder {data_folder} does not exits')

    image_files = os.listdir(data_folder)
    total = np.zeros(3)

    for img_file in tqdm(image_files):
        path = os.path.join(data_folder, img_file)
        image = cv2.imread(path)
        total += image.mean(axis=(0, 1))

    channel_mean = total / len(image_files)
    mean = np.mean(channel_mean)

    return mean - channel_mean[...,::-1]

