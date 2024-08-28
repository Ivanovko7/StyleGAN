import torch
import argparse
import os

from models.stylegen import StyleGen
from models.discriminator import Discriminator
from lib.common import load_checkpoint
from lib.datasets import SamplesDataSet
from lib.trainer import Trainer
from lib.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amp', action='store_true', help="Automatic Mixed Precision")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--cache', action='store_true', help="Use disk cache")
    parser.add_argument('--debug_samples', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--real_images_dir', type=str, default='dataset/real')
    parser.add_argument('--resume', action='store_true', help="Resume from last checkpoint")
    parser.add_argument('--resume_style_gen', type=str, default='False')
    parser.add_argument('--resume_style_gen_init', type=str, default='False')
    parser.add_argument('--resume_discriminator', type=str, default='False')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--sn', action='store_true')
    parser.add_argument('--samples_images_dir', type=str, default='dataset/samples')
    parser.add_argument('--test_images_dir', type=str, default='dataset/test')
    parser.add_argument('--image_size', type=int, nargs="+", default=[256],
                        help="Image sizes")
    parser.add_argument('--resize_method', type=str, default="crop",
                        help="Resize image method if origin photo larger than image_size")
    # Loss stuff
    parser.add_argument('--style_gen_lr', type=float, default=0.00003)
    parser.add_argument('--discriminator_lr', type=float, default=0.00006)
    parser.add_argument('--init_lr', type=float, default=0.0001)
    parser.add_argument('--advcw', type=float, default=300.0, help='Adversarial loss weight for style_gen')
    parser.add_argument('--advdw', type=float, default=300.0, help='Adversarial loss weight for Discriminator')
    # Loss weight VGG19
    parser.add_argument('--ctlw', type=float, default=1.5, help='Gram matrix content loss weight')
    parser.add_argument('--stlw', type=float, default=4.0, help='Gram matrix style loss weight')
    parser.add_argument('--colw', type=float, default=20.0, help='Color loss weight')
    parser.add_argument('--tvlw', type=float, default=1.0, help='Total variation loss weight')
    parser.add_argument('--d_layers', type=int, default=2, help='Discriminator conv layers')
    parser.add_argument('--d_noise', action='store_true')

    # DDP
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--world-size", default=2, type=int)

    return parser.parse_args()


def check_params(args):
    args.dataset = f"{os.path.basename(args.real_images_dir)}_{os.path.basename(args.samples_images_dir)}"


def main(args, logger):
    check_params(args)

    if torch.cuda.is_available():
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        logger.info("Using MPS")
        args.device = 'mps'
    else:
        logger.info("Using CPU")
        args.device = 'cpu'

    style_generator = StyleGen(args.dataset)
    discriminator = Discriminator(
        args.dataset,
        num_layers=args.d_layers,
        use_spectral_norm=args.sn,
        normalization="layer",
    )

    start_epoch = 0
    start_epoch_init = 0

    trainer = Trainer(
        style_gen=style_generator,
        discriminator=discriminator,
        config=args,
        logger=logger,
    )

    if args.resume_style_gen_init.lower() != 'false':
        start_epoch_init = load_checkpoint(style_generator, args.resume_style_gen_init) + 1
        if args.local_rank == 0:
            logger.info(f"Style generator content weights loaded from {args.resume_style_gen_init}")
    elif args.resume_style_gen.lower() != 'false' and args.resume_discriminator.lower() != 'false':
        try:
            start_epoch = load_checkpoint(style_generator, args.resume_style_gen)
            if args.local_rank == 0:
                logger.info(f"Style generator weights loaded from {args.resume_style_gen}")
            load_checkpoint(discriminator, args.resume_discriminator)
            if args.local_rank == 0:
                logger.info(f"Discriminator weights loaded from {args.resume_discriminator}")
            args.init_epochs = 0
        except Exception as e:
            logger.error('Could not load weights, training from scratch', e)
    elif args.resume:
        logger.info(f"Loading weights from {trainer.checkpoint_path_G}")
        start_epoch = load_checkpoint(style_generator, trainer.checkpoint_path_G)
        logger.info(f"Loading weights from {trainer.checkpoint_path_D}")
        load_checkpoint(discriminator, trainer.checkpoint_path_D)
        args.init_epochs = 0

    dataset = SamplesDataSet(
        args.samples_images_dir,
        args.real_images_dir,
        args.debug_samples,
        args.cache,
        image_size=args.image_size,
        resize_method=args.resize_method,
    )

    if args.local_rank == 0:
        logger.info(f"Starting training from epoch {start_epoch}, {start_epoch_init}")
    trainer.train(dataset, start_epoch, start_epoch_init)

if __name__ == '__main__':
    args = parse_args()
    real_name = os.path.basename(args.real_images_dir)
    args.exp_dir = os.path.join("runs",os.path.basename(args.samples_images_dir))

    os.makedirs(args.exp_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.exp_dir, "training.log"))

    if args.local_rank == 0:
        logger.info("# ******* Configuration ******* #")
        for arg in vars(args):
            logger.info(f"{arg} {getattr(args, arg)}")
        logger.info("----------------------------------")

    main(args, logger)
