import os
import cv2
import numpy as np
import torch
import random
from tqdm.auto import tqdm
from glob import glob
from torch.utils.data import Dataset
from lib.image_processing import normalize_input, calc_mean
from lib.fast_numpyio import save, load

CACHE_DIR = '.tmp'

def get_random_crop(image, height, width):

    max_x = max(image.shape[1] - width, 0)
    max_y = max(image.shape[0] - height, 0)

    x = np.random.randint(0, max_x) if max_x != 0 else 0
    y = np.random.randint(0, max_y) if max_y != 0 else 0

    crop = image[y: y + height, x: x + width]

    return crop

class SamplesDataSet(Dataset):
    def __init__(
            self,
            samples_image_dir,
            real_image_dir,
            debug_samples=0,
            cache=False,
            transform=None,
            image_size=512,
            resize_method="resize"
    ):
        """
        folder structure:
        - {samples_image_dir}  # E.g train
            smooth
                a.jpg, ..., n.jpg
            style
                a.jpg, ..., n.jpg
        """
        self.cache = cache

        if isinstance(image_size, list):
            image_size = image_size[0]

        self.debug_samples = debug_samples
        self.resize_method = resize_method
        self.image_files = {}
        self.train_data = 'data'
        self.style = 'style'
        self.smooth = 'smooth'
        self.cache_files = {}
        self.samples_dirname = os.path.basename(samples_image_dir)
        self.image_size = image_size
        for dir, opt in [
            (real_image_dir, self.train_data),
            (os.path.join(samples_image_dir, self.style), self.style),
            (os.path.join(samples_image_dir, self.smooth), self.smooth)
        ]:
            self.image_files[opt] = glob(os.path.join(dir, "*.*"))
            self.cache_files[opt] = [False] * len(self.image_files[opt])

        self.transform = transform
        self.cache_data()

        print(f'Dataset: real {self.len_photo}, style {self.len_anime} and smooth {self.len_smooth}')

    def __len__(self):
        return self.debug_samples or self.len_anime

    @property
    def len_photo(self):
        return len(self.image_files[self.train_data])

    @property
    def len_anime(self):
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        photo_idx = random.randint(0, self.len_photo - 1)
        anm_idx = index

        image = self.load_photo(photo_idx)
        anime, anime_gray = self.load_sample(anm_idx)
        smooth_gray = self.load_sample_smooth(anm_idx)

        return {
            "image": torch.tensor(image).contiguous(),
            "anime": torch.tensor(anime).contiguous(),
            "anime_gray": torch.tensor(anime_gray).contiguous(),
            "smooth_gray": torch.tensor(smooth_gray).contiguous()
        }

    def set_image_size(self, image_size):
        self.image_size = image_size

    def cache_data(self):
        if not self.cache:
            return

        cache_dir = os.path.join(CACHE_DIR, self.samples_dirname)
        os.makedirs(cache_dir, exist_ok=True)
        print("Caching data..")
        cache_nbytes = 0
        for opt, image_files in self.image_files.items():
            cache_sub_dir = os.path.join(cache_dir, opt)
            os.makedirs(cache_sub_dir, exist_ok=True)
            for index, img_file in enumerate(tqdm(image_files)):
                save_path = os.path.join(cache_sub_dir, f"{index}.npy")
                if os.path.exists(save_path):
                    continue
                if opt == self.train_data:
                    image = self.load_photo(index)
                    cache_nbytes += image.nbytes
                    save(save_path, image)
                    self.cache_files[opt][index] = save_path
                elif opt == self.smooth:
                    image = self.load_sample_smooth(index)
                    cache_nbytes += image.nbytes
                    save(save_path, image)
                    self.cache_files[opt][index] = save_path
                elif opt == self.style:
                    image, image_gray = self.load_sample(index)
                    cache_nbytes += image.nbytes + image_gray.nbytes
                    save(save_path, image)
                    save_path_gray = os.path.join(cache_sub_dir, f"{index}_gray.npy")
                    save(save_path_gray, image_gray)
                    self.cache_files[opt][index] = (save_path, save_path_gray)
                else:
                    raise ValueError(opt)
        print(f"Cache saved to {cache_dir}, size={cache_nbytes/1e9} Gb")

    def load_photo(self, index) -> np.ndarray:
        if self.cache_files[self.train_data][index]:
            fpath = self.cache_files[self.train_data][index]
            image = load(fpath)
        else:
            fpath = self.image_files[self.train_data][index]
            image = cv2.imread(fpath)[:,:,::-1]
            if self.resize_method == "resize":
                image = cv2.resize(image, (self.image_size, self.image_size))
            else:
                random_size = random.randint(
                    int(self.image_size * 0.5),
                    int(self.image_size * 1)
                )
                image = get_random_crop(image, random_size, random_size)
                image = cv2.resize(image, (self.image_size, self.image_size))

            image = self._transform(image, addmean=False)
            image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
        return image

    def load_sample(self, index) -> np.ndarray:
        if self.cache_files[self.style][index]:
            fpath, fpath_gray = self.cache_files[self.style][index]
            image = load(fpath)
            image_gray = load(fpath_gray)
        else:
            fpath = self.image_files[self.style][index]
            image = cv2.imread(fpath)[:,:,::-1]
            image = cv2.resize(image, (self.image_size, self.image_size))

            image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)

            image_gray = self._transform(image_gray, addmean=False)
            image_gray = image_gray.transpose(2, 0, 1)
            image_gray = np.ascontiguousarray(image_gray)

            image = self._transform(image, addmean=False)
            image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)

        return image, image_gray

    def load_sample_smooth(self, index) -> np.ndarray:
        if self.cache_files[self.smooth][index]:
            fpath = self.cache_files[self.smooth][index]
            image = load(fpath)
        else:
            fpath = self.image_files[self.smooth][index]
            image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = np.stack([image, image, image], axis=-1)
            image = self._transform(image, addmean=False)
            image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
        return image

    def _transform(self, img, addmean=False):
        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = img.astype(np.float32)
        if addmean:
            img += self.mean

        return normalize_input(img)
