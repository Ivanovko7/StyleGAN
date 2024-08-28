import os
import time
import shutil

import torch
import cv2
import torch.optim as optim
import numpy as np
from glob import glob
from torch.amp import GradScaler, autocast
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from color_transfer import color_transfer_pytorch
from lib.image_processing import denormalize_input, prepare_images, resize_image
from lib.common import load_checkpoint, save_checkpoint, read_image, set_lr
from lib.losses import LossSummary, AnimeGanLoss


def convert_to_readable(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def transfer_color_and_rescale(src, target):
    """Transfer color from src image to target then rescale to [-1, 1]"""
    out = color_transfer_pytorch(src, target)  # [0, 1]
    out = (out / 0.5) - 1
    return out

def generate_gaussian_noise():
    """Generate a tensor of gaussian noise with mean 0 and standard deviation 0.1"""
    mean = torch.tensor(0.0)
    std_dev = torch.tensor(0.1)
    return torch.normal(mean, std_dev)


def convert_tensor_to_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor representing an image to a numpy array in the range [0, 255]"""
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = denormalize_input(image, dtype=np.uint8)
    return image[..., ::-1]

def save_generated_images(generated_images: torch.Tensor, output_dir: str) -> None:
    """
    Saves a batch of generated images to disk.

    Args:
        generated_images (torch.Tensor): A tensor containing images with shape
            `(*, 3, H, W)` and pixel values in the range [-1, 1].
        output_dir (str): The directory path where the images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    images = generated_images.clone().detach().cpu().numpy()
    images = images.transpose(0, 2, 3, 1)
    num_images = len(images)

    for i, img in enumerate(images):
        img = denormalize_input(img, dtype=np.uint8)
        img = img[..., ::-1]
        cv2.imwrite(os.path.join(output_dir, f"G{i}.jpg"), img)


class DDPTrainer:
    def _init_distributed(self):
        if self.cfg.ddp:
            self.logger.info("Setting up DDP")
            self.pg = torch.distributed.init_process_group(
                backend="nccl",
                rank=self.cfg.local_rank,
                world_size=self.cfg.world_size
            )
            self.style_gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.style_gen, self.pg)
            self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator, self.pg)
            torch.cuda.set_device(self.cfg.local_rank)
            self.style_gen.cuda(self.cfg.local_rank)
            self.discriminator.cuda(self.cfg.local_rank)
            self.logger.info("Setting up DDP Done")

    def _init_amp(self, enabled=False):
        # self.scaler = torch.cuda.amp.GradScaler(enabled=enabled, growth_interval=100)
        self.scaler_g = GradScaler(enabled=enabled)
        self.scaler_d = GradScaler(enabled=enabled)
        if self.cfg.ddp:
            self.style_gen = DistributedDataParallel(
                self.style_gen, device_ids=[self.cfg.local_rank],
                output_device=self.cfg.local_rank,
                find_unused_parameters=False)

            self.discriminator = DistributedDataParallel(
                self.discriminator, device_ids=[self.cfg.local_rank],
                output_device=self.cfg.local_rank,
                find_unused_parameters=False)
            self.logger.info("Set DistributedDataParallel")


class Trainer(DDPTrainer):

    def __init__(
        self,
        style_gen,
        discriminator,
        config,
        logger,
    ) -> None:
        """
        Initialize Trainer.

        Parameters
        ----------
        style_gen : nn.Module
            The composed network.
        discriminator : nn.Module
            The discriminator network.
        config : Config
            The configuration.
        logger : Logger
            The logger.
        """
        self.cfg = config
        self.style_gen = style_gen
        self.discriminator = discriminator
        self.max_norm = 10
        self.device_type = 'cuda' if self.cfg.device.startswith('cuda') else 'cpu'
        self.optimizer_g = optim.Adam(self.style_gen.parameters(), lr=self.cfg.style_gen_lr, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.cfg.discriminator_lr, betas=(0.5, 0.999))
        self.loss_tracker = LossSummary()
        if self.cfg.ddp:
            self.device = torch.device(f"cuda:{self.cfg.local_rank}")
            logger.info(f"---------{self.cfg.local_rank} {self.device}")
        else:
            self.device = torch.device(self.cfg.device)
        self.loss_fn = AnimeGanLoss(self.cfg, self.device)
        self.logger = logger
        self._init_working_dir()
        self._init_distributed()
        self._init_amp(enabled=self.cfg.amp)

    def _init_working_dir(self):
        """Init working directory for saving checkpoint, ..."""
        os.makedirs(self.cfg.exp_dir, exist_ok=True)
        Gname = self.style_gen.name
        Dname = self.discriminator.name
        self.checkpoint_path_G_init = os.path.join(self.cfg.exp_dir, f"{Gname}_init.pt")
        self.checkpoint_path_G = os.path.join(self.cfg.exp_dir, f"{Gname}.pt")
        self.checkpoint_path_D = os.path.join(self.cfg.exp_dir, f"{Dname}.pt")
        self.save_image_dir = os.path.join(self.cfg.exp_dir, "example_images")
        self.example_image_dir = os.path.join(self.cfg.exp_dir, "train_images")
        os.makedirs(self.save_image_dir, exist_ok=True)
        os.makedirs(self.example_image_dir, exist_ok=True)

    def init_weight_G(self, weight: str):
        """Init Composition weight"""
        return load_checkpoint(self.style_gen, weight)

    def init_weight_D(self, weight: str):
        """Init Discriminator weight"""
        return load_checkpoint(self.discriminator, weight)

    def pretrain_generator(self, train_loader, start_epoch):
        """
        Pretrain Composition to reconstruct input image.
        """
        init_losses = []
        set_lr(self.optimizer_g, self.cfg.init_lr)
        for epoch in range(start_epoch, self.cfg.init_epochs):
            # Train with content loss only

            pbar = tqdm(train_loader)
            for data in pbar:
                img = data["image"].to(self.device)

                self.optimizer_g.zero_grad()

                with autocast(enabled=self.cfg.amp, device_type=self.device_type):
                    fake_img = self.style_gen(img)
                    loss = self.loss_fn.content_loss_vgg(img, fake_img)

                self.scaler_g.scale(loss).backward()
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()

                if self.cfg.ddp:
                    torch.distributed.barrier()

                init_losses.append(loss.cpu().detach().numpy())
                avg_content_loss = sum(init_losses) / len(init_losses)
                pbar.set_description(f'[Init Training] content loss: {avg_content_loss:2f}')

            save_checkpoint(self.style_gen, self.checkpoint_path_G_init, self.optimizer_g, epoch)
            if self.cfg.local_rank == 0:
                self.generate_and_save(self.cfg.test_images_dir, subname='init')
                self.logger.info(f"Epoch {epoch}/{self.cfg.init_epochs}")

        set_lr(self.optimizer_g, self.cfg.style_gen_lr)

    def train_epoch(self, epoch, train_loader):
        pbar = tqdm(train_loader, total=len(train_loader))
        for data in pbar:
            img = data["image"].to(self.device)
            anime = data["anime"].to(self.device)
            anime_gray = data["anime_gray"].to(self.device)
            anime_smt_gray = data["smooth_gray"].to(self.device)

            # ---------------- TRAIN Discriminator ---------------- #
            self.optimizer_d.zero_grad()

            with autocast(enabled=self.cfg.amp, device_type=self.device_type):
                fake_img = self.style_gen(img)
                # Add some Gaussian noise to images before feeding to D
                if self.cfg.d_noise:
                    fake_img += generate_gaussian_noise()
                    anime += generate_gaussian_noise()
                    anime_gray += generate_gaussian_noise()
                    anime_smt_gray += generate_gaussian_noise()

                fake_d = self.discriminator(fake_img)
                real_anime_d = self.discriminator(anime)
                real_anime_gray_d = self.discriminator(anime_gray)
                real_anime_smt_gray_d = self.discriminator(anime_smt_gray)

                loss_d = self.loss_fn.compute_loss_D(
                    fake_d,
                    real_anime_d,
                    real_anime_gray_d,
                    real_anime_smt_gray_d
                )

            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.unscale_(self.optimizer_d)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.max_norm)
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
            if self.cfg.ddp:
                torch.distributed.barrier()
            self.loss_tracker.update_loss_D(loss_d)

            # ---------------- TRAIN style_gen ---------------- #
            self.optimizer_g.zero_grad()

            with autocast(enabled=self.cfg.amp, device_type=self.device_type):
                fake_img = self.style_gen(img)
                fake_d = self.discriminator(fake_img)

                (
                    adv_loss, con_loss,
                    gra_loss, col_loss,
                    tv_loss
                ) = self.loss_fn.compute_loss_G(
                    fake_img,
                    img,
                    fake_d,
                    anime_gray,
                )
                loss_g = adv_loss + con_loss + gra_loss + col_loss + tv_loss
                if torch.isnan(adv_loss).any():
                    self.logger.info("--------------------------------------------")
                    self.logger.info(fake_d)
                    self.logger.info(adv_loss)
                    self.logger.info("--------------------------------------------")
                    raise ValueError("NAN loss!!")

            self.scaler_g.scale(loss_g).backward()
            self.scaler_d.unscale_(self.optimizer_g)
            grad = torch.nn.utils.clip_grad_norm_(self.style_gen.parameters(), max_norm=self.max_norm)
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
            if self.cfg.ddp:
                torch.distributed.barrier()

            self.loss_tracker.update_loss_G(adv_loss, gra_loss, col_loss, con_loss)
            pbar.set_description(f"{self.loss_tracker.get_loss_description()} - {grad:.3f}")


    def create_train_loader(self, dataset):
        """
        Create and return a DataLoader for training.

        Parameters
        ----------
        dataset : Dataset
            The dataset to load data from.

        Returns
        -------
        DataLoader
            A DataLoader configured with the specified settings for training.
            If distributed data parallel (DDP) is enabled, a DistributedSampler
            is used to partition the dataset across multiple processes.
        """
        if self.cfg.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = None
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
        )

    def maybe_increase_image_size(self, epoch, train_dataset):
        """
        Increase image size at specific epoch
            + 50% epochs train at image_size[0]
            + the rest 50% will increase every `len(epochs) / 2 / (len(image_size) - 1)`

        Args:
            epoch: Current epoch
            train_dataset: Dataset

        Examples:
        ```
        epochs = 100
        image_size = [256, 352, 416, 512]
        => [(0, 256), (50, 352), (66, 416), (82, 512)]
        ```
        """
        epochs = self.cfg.epochs
        image_size = self.cfg.image_size
        num_size_remains = len(image_size) - 1
        half_epochs = epochs // 2

        if len(image_size) == 1:
            new_size = image_size[0]
        elif epoch < half_epochs:
            new_size = image_size[0]
        else:
            per_epoch_increment = int(half_epochs / num_size_remains)
            found = None
            for i, size in enumerate(image_size[:]):
                if epoch < half_epochs + per_epoch_increment * i:
                    found = size
                    break
            if not found:
                found = image_size[-1]
            new_size = found

        self.logger.info(f"Check {image_size}, {new_size}, {train_dataset.image_size}")
        if new_size != train_dataset.image_size:
            train_dataset.set_image_size(new_size)
            self.logger.info(f"Increase image size to {new_size} at epoch {epoch}")

    def train(self, train_dataset: Dataset, start_epoch=0, start_epoch_g=0):
        """
        Train Composition and Discrimination.
        """
        self.logger.info(self.device)
        self.style_gen.to(self.device)
        self.discriminator.to(self.device)

        self.pretrain_generator(self.create_train_loader(train_dataset), start_epoch_g)

        if self.cfg.local_rank == 0:
            self.logger.info(f"Begin training for {self.cfg.epochs} epochs")

        for i, data in enumerate(train_dataset):
            for k in data.keys():
                image = data[k]
                cv2.imwrite(
                    os.path.join(self.example_image_dir, f"data_{k}_{i}.jpg"),
                    convert_tensor_to_image(image)
                )
            if i == 2:
                break

        end = None
        num_iter = 0
        per_epoch_times = []
        for epoch in range(start_epoch, self.cfg.epochs):
            self.maybe_increase_image_size(epoch, train_dataset)

            start = time.time()
            self.train_epoch(epoch, self.create_train_loader(train_dataset))

            if epoch % self.cfg.save_interval == 0 and self.cfg.local_rank == 0:
                save_checkpoint(self.style_gen, self.checkpoint_path_G, self.optimizer_g, epoch)
                save_checkpoint(self.discriminator, self.checkpoint_path_D, self.optimizer_d, epoch)
                self.generate_and_save(self.cfg.test_images_dir)

                if epoch % 10 == 0:
                    self.copy_weight(epoch)

            num_iter += 1

            if self.cfg.local_rank == 0:
                end = time.time()
                if end is None:
                    eta = 9999
                else:
                    per_epoch_time = (end - start)
                    per_epoch_times.append(per_epoch_time)
                    eta = np.mean(per_epoch_times) * (self.cfg.epochs - epoch)
                    eta = convert_to_readable(eta)
                self.logger.info(f"Epoch {epoch}/{self.cfg.epochs}, ETA: {eta}")

    def generate_and_save(
        self,
        image_dir,
        max_imgs=20,
        subname='example'
    ):
        start = time.time()
        self.style_gen.eval()

        max_iter = max_imgs
        fake_imgs = []
        real_imgs = []
        image_files = glob(os.path.join(image_dir, "*"))

        for i, image_file in enumerate(image_files):
            image = read_image(image_file)
            image = resize_image(image)
            real_imgs.append(image.copy())
            image = prepare_images(image)
            image = image.to(self.device)
            with torch.no_grad():
                with autocast(enabled=self.cfg.amp, device_type=self.device_type):
                    fake_img = self.style_gen(image)
                fake_img = fake_img.detach().cpu().numpy()
                # Channel first -> channel last
                fake_img  = fake_img.transpose(0, 2, 3, 1)
                fake_imgs.append(denormalize_input(fake_img, dtype=np.int16)[0])

            if i + 1 == max_iter:
                break

        for i, (real_img, fake_img) in enumerate(zip(real_imgs, fake_imgs)):
            img = np.concatenate((real_img, fake_img), axis=1)  # Concate aross width
            save_path = os.path.join(self.save_image_dir, f'{subname}_{i}.jpg')
            if not cv2.imwrite(save_path, img[..., ::-1]):
                self.logger.info(f"Save styled image failed, {save_path}, {img.shape}")
        elapsed = time.time() - start
        self.logger.info(f"Stylized {len(fake_imgs)} images in {elapsed:.3f}s.")

    def copy_weight(self, epoch):
        copy_dir = os.path.join(self.cfg.exp_dir, f"epoch_{epoch}")
        os.makedirs(copy_dir, exist_ok=True)

        shutil.copy2(
            self.checkpoint_path_G,
            copy_dir
        )

        dest = os.path.join(copy_dir, os.path.basename(self.save_image_dir))
        shutil.copytree(
            self.save_image_dir,
            dest,
            dirs_exist_ok=True
        )
