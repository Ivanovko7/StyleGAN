import torch
import torch.nn.functional as F
import torch.nn as nn
from models.vgg import Vgg19
from lib.image_processing import gram

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()
        # self._rgb_to_yuv_kernel = torch.tensor([
        #     [0.299, -0.14714119, 0.61497538],
        #     [0.587, -0.28886916, -0.51496512],
        #     [0.114, 0.43601035, -0.10001026]
        # ]).float()

        self._rgb_to_yuv_kernel = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14714119, -0.28886916, 0.43601035],
            [0.61497538, -0.51496512, -0.10001026],
        ]).float()

    def to(self, device):
        new_self = super(ColorLoss, self).to(device)
        new_self._rgb_to_yuv_kernel = new_self._rgb_to_yuv_kernel.to(device)
        return new_self

    def rgb_to_yuv(self, image):
        image = (image + 1.0) / 2.0
        image = image.permute(0, 2, 3, 1) # To channel last
        yuv_img = image @ self._rgb_to_yuv_kernel.T

        return yuv_img

    def forward(self, image, image_g):
        image = self.rgb_to_yuv(image)
        image_g = self.rgb_to_yuv(image_g)
        return (
            self.l1(image[:, :, :, 0], image_g[:, :, :, 0])
            + self.huber(image[:, :, :, 1], image_g[:, :, :, 1])
            + self.huber(image[:, :, :, 2], image_g[:, :, :, 2])
        )


class AnimeGanLoss:
    def __init__(self, args, device):
        if isinstance(device, str):
            device = torch.device(device)

        self.content_loss = nn.L1Loss().to(device)
        self.gram_loss = nn.L1Loss().to(device)
        self.color_loss = ColorLoss().to(device)
        self.advcw = args.advcw
        self.advdw = args.advdw
        self.ctlw = args.ctlw
        self.stlw = args.stlw
        self.colw = args.colw
        self.tvlw = args.tvlw
        self.vgg19 = Vgg19().to(device).eval()
        self.adv_type = 'lsgan'
        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_loss_G(self, fake_img, img, fake_logit, anime_gray):
        '''
        Compute loss for style_gen

        @Args:
            - fake_img: generated image
            - img: real image
            - fake_logit: output of Discriminator given fake image
            - anime_gray: grayscale of anime image

        @Returns:
            - Adversarial Loss of fake logits
            - Content loss between real and fake features (vgg19)
            - Gram loss between style and fake features (Vgg19)
            - Color loss between image and fake image
            - Total variation loss of fake image
        '''
        fake_feat = self.vgg19(fake_img)
        gray_feat = self.vgg19(anime_gray)
        img_feat = self.vgg19(img)

        return [
            # Want to be real image.
            self.advcw * self.adv_loss_g(fake_logit),
            self.ctlw * self.content_loss(img_feat, fake_feat),
            self.stlw * self.gram_loss(gram(gray_feat), gram(fake_feat)),
            self.colw * self.color_loss(img, fake_img),
            self.tvlw * self.total_variation_loss(fake_img)
        ]

    def compute_loss_D(
        self,
        fake_img_d,
        real_anime_d,
        real_anime_gray_d,
        real_anime_smooth_gray_d=None
    ):

        return self.advdw * (
            # Classify real anime as real
            self.adv_loss_d_real(real_anime_d)
            # Classify generated as fake
            + self.adv_loss_d_fake(fake_img_d)
            # Classify real anime gray as fake
            + self.adv_loss_d_fake(real_anime_gray_d)
            # Classify real anime as fake
            + 0.1 * self.adv_loss_d_fake(real_anime_smooth_gray_d)
        )

    def total_variation_loss(self, fake_img):
        """
        A smooth loss in fact. Like the smooth prior in MRF.
        V(y) = || y_{n+1} - y_n ||_2
        """
        # Channel first -> channel last
        fake_img = fake_img.permute(0, 2, 3, 1)
        def _l2(x):
            # sum(t ** 2) / 2
            return torch.sum(x ** 2) / 2

        dh = fake_img[:, :-1, ...] - fake_img[:, 1:, ...]
        dw = fake_img[:, :, :-1, ...] - fake_img[:, :, 1:, ...]
        return _l2(dh) / dh.numel() + _l2(dw) / dw.numel()

    def content_loss_vgg(self, image, recontruction):
        feat = self.vgg19(image)
        re_feat = self.vgg19(recontruction)
        feature_loss = self.content_loss(feat, re_feat)
        content_loss = self.content_loss(image, recontruction)
        return feature_loss + 0.5 * content_loss

    def adv_loss_d_real(self, pred):
        """Push pred to class 1 (real)"""
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 - pred))

        elif self.adv_type == 'lsgan':
            # pred = torch.sigmoid(pred)
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'bce':
            return self.bce_loss(pred, torch.ones_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')

    def adv_loss_d_fake(self, pred):
        """Push pred to class 0 (fake)"""
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 + pred))

        elif self.adv_type == 'lsgan':
            # pred = torch.sigmoid(pred)
            return torch.mean(torch.square(pred))

        elif self.adv_type == 'bce':
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')

    def adv_loss_g(self, pred):
        """Push pred to class 1 (real)"""
        if self.adv_type == 'hinge':
            return -torch.mean(pred)

        elif self.adv_type == 'lsgan':
            # pred = torch.sigmoid(pred)
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'bce':
            return self.bce_loss(pred, torch.ones_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')


class LossSummary:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_g_adv = []
        self.loss_content = []
        self.loss_gram = []
        self.loss_color = []
        self.loss_d_adv = []

    def update_loss_G(self, adv, gram, color, content):
        self.loss_g_adv.append(adv.cpu().detach().numpy())
        self.loss_gram.append(gram.cpu().detach().numpy())
        self.loss_color.append(color.cpu().detach().numpy())
        self.loss_content.append(content.cpu().detach().numpy())

    def update_loss_D(self, loss):
        self.loss_d_adv.append(loss.cpu().detach().numpy())

    def avg_loss_G(self):
        return (
            self._avg(self.loss_g_adv),
            self._avg(self.loss_gram),
            self._avg(self.loss_color),
            self._avg(self.loss_content),
        )

    def avg_loss_D(self):
        return self._avg(self.loss_d_adv)

    def get_loss_description(self):
        avg_adv, avg_gram, avg_color, avg_content = self.avg_loss_G()
        avg_adv_d = self.avg_loss_D()
        return f'loss G: adv {avg_adv:2f} con {avg_content:2f} gram {avg_gram:2f} color {avg_color:2f} / loss D: {avg_adv_d:2f}'

    @staticmethod
    def _avg(losses):
        return sum(losses) / len(losses)
