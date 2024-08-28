import os
import time

import cv2
from color_transfer import color_transfer_pytorch
import torch
import numpy as np

from models.stylegen import StyleGen

from lib.common import load_checkpoint, read_image, is_image_file
from lib.image_processing import resize_image, normalize_input, denormalize_input


def profile(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - started_at
        print(f"Stylized in {elapsed:.3f}s")
        return result
    return wrap


def auto_load_weight(weight, map_location=None):
    model = StyleGen()
    load_checkpoint(model, weight, strip_optimizer=True, map_location=map_location)
    model.eval()
    return model


class Predictor:
    def __init__(
        self,
        amp=True,
        weight='style',
        device='cuda',
        retain_color=False
    ):
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = 'mps'
                print("Use MPS device")
            else:
                device = 'cpu'
                print("Use CPU device")
        else:
            print(f"Use GPU {torch.cuda.get_device_name()}!")

        self.amp = amp  # Automatic Mixed Precision
        self.retain_color = retain_color
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        self.device = torch.device(device)

        self.style_gen = auto_load_weight(weight, map_location=device)
        self.style_gen.to(self.device)

    def transform(self, image, denorm=True):
        '''
        Transform a image to animation

        @Arguments:
            - image: np.array, shape = (Batch, width, height, channels)

        @Returns:
            - anime version of image: np.array
        '''
        with torch.no_grad():
            image = self.prepare_images(image)
            fake = self.style_gen(image)
            if self.retain_color:
                fake = color_transfer_pytorch(fake, image)
                fake = (fake / 0.5) - 1.0  # remap to [-1. 1]
            fake = fake.detach().cpu().numpy()
            fake = fake.transpose(0, 2, 3, 1)

            if denorm:
                fake = denormalize_input(fake, dtype=np.uint8)
            return fake

    def read_and_resize(self, path, max_size=1536):
        image = read_image(path)
        h, w = image.shape[:2]

        if max(h, w) > max_size:
            print(f"Too big Image {os.path.basename(path)}  ({h}x{w}), resize to max size {max_size}")
            image = resize_image(
                image,
                width=max_size if w > h else None,
                height=max_size if w < h else None,
            )
            _, ext = os.path.splitext(path)
            cv2.imwrite(path.replace(ext, ".jpg"), image[:,:,::-1])
        else:
            image = resize_image(image)
        return image

    @profile
    def transform_file(self, file_path, save_path):
        if not is_image_file(save_path):
            raise ValueError(f"{save_path} is not valid")

        image = self.read_and_resize(file_path)
        anime_img = self.transform(image)[0]
        cv2.imwrite(save_path, anime_img[..., ::-1])
        print(f"Image saved to {save_path}")


    def prepare_images(self, images):
        '''
        Preprocess image for inference

        @Arguments:
            - images: np.ndarray

        @Returns
            - images: torch.tensor
        '''
        images = images.astype(np.float32)

        # Normalize to [-1, 1]
        images = normalize_input(images)
        images = torch.from_numpy(images)

        images = images.to(self.device)

        # Add batch dim
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        # channel first
        images = images.permute(0, 3, 1, 2)

        return images


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to an image that will be stylized')
    parser.add_argument('--save-as', type=str, default='inference_images', help='Output file path')
    parser.add_argument(
        '--weight',
        type=str,
        help=f'Model weight, need be path to weight file (*.pt)'
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device, cuda or cpu')
    parser.add_argument(
        '--retain-color',
        action='store_true',
        help='Retain color of the image')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    predictor = Predictor(
        args.device,
        args.weight,
        retain_color=args.retain_color
    )

    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)

    if os.path.isfile(args.image):
        save_path = args.save_as
        if not is_image_file(args.save_as):
            raise NotImplementedError(f"{args.out} is not valid image filename")

        predictor.transform_file(args.image, save_path)
    else:
        raise NotImplementedError(f"{args.image} is not supported")
