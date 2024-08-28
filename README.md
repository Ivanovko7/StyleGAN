# StyleGAN
A unofficial implementation of style Generative Adversarial Network, a deep learning model for transforming real-world images into cartoon like style images.
This project is for my study purposes only. I am not guaranteed that it works perfectly or work for you.

## Overview
This project implements Generative adversarial network, a deep learning model for transforming real-world images into styled (cartoons/animation etc) images.

## Usage

### Training
To train the model, use the `train.py` script. Example:
```bash
python train.py --real_images_dir path/to/real/images --samples_images_dir path/to/anime/images --exp_dir path/to/experiment
```

### Transforming Image
To transform an image using a pre-trained model, use the `transform.py` script. Example:
```bash
python transform.py --image path/to/input/image.jpg --save-as path/to/output/image.jpg --weight path/to/model/weight.pt
```

### TensorBoard
To visualize training progress with TensorBoard:
```bash
tensorboard --logdir=path/to/experiment
```

## Project Structure
- `train.py`: Script for training the AnimeGAN model.
- `transform.py`: Script for transforming images using a pre-trained model.
- `models/`: Directory containing model definitions.
- `lib/`: Directory containing utility functions and classes.
- `datasets/`: Directory for storing datasets.
- `runs/`: Directory for storing experiment logs and checkpoints.

## Configuration
The training and inference scripts accept various command-line arguments for configuration. Use the `--help` flag to see all available options:
```bash
python train.py --help
python transform.py --help
```

## License
All what I made is under MIT License.
