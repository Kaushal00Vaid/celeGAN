# celeGAN

Exploring and implementing different types of GANs on the CelebA dataset.

Dataset: [CelebA Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

## Project Overview

This repository contains PyTorch implementations of various Generative Adversarial Networks (GANs) trained on the CelebA dataset. The project is structured progressively, starting from a basic MLP-based GAN and moving towards more advanced convolutional and conditional models.

## Implemented Models

1. **[Vanilla GAN](src/VanillaGAN/README.md)**: A basic implementation using Multi-Layer Perceptrons (MLPs) to understand the fundamental concepts and limitations of fully connected networks in image generation. *(Paper: [Generative Adversarial Nets, Goodfellow et al. 2014](https://arxiv.org/abs/1406.2661))*
2. **[DCGAN (Deep Convolutional GAN)](src/DCGAN/README.md)**: Replaces MLPs with Convolutional Neural Networks (CNNs) to capture spatial hierarchies, resulting in significantly sharper and more realistic face generation. *(Paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Radford et al. 2015](https://arxiv.org/abs/1511.06434))*
3. **[cGAN (Conditional GAN)](src/cGAN/README.md)**: Extends the architecture by conditioning the generation process on specific attributes or labels, allowing for controlled generation. *(Paper: [Conditional Generative Adversarial Nets, Mirza & Osindero 2014](https://arxiv.org/abs/1411.1784))*
4. **[SRGAN (Super-Resolution GAN)](src/SRGAN/README.md)**: Focuses on upscaling low-resolution facial images to high resolution while recovering photo-realistic textures using perceptual loss. *(Paper: [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, Ledig et al. 2016](https://arxiv.org/abs/1609.04802))*

## Project Structure

- `src/`: Contains the implementations for all the GANs. Each model has its own directory with its respective `model.py`, `train.py`, and `dataset.py`.
- `dataset/`: Expected directory for the downloaded CelebA dataset. - **gitignored (get from kaggle or make pt files by following `generate_pt_files.py`)**
- `notebooks/`: Jupyter notebooks for exploratory data analysis and experiments.
- `requirements.txt`: Python dependencies required to run the code.

## Getting Started

1. Clone the repository and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the CelebA dataset and place it in the `dataset/` directory. - **gitignored (get from kaggle or make pt files by following `generate_pt_files.py`)**
3. Navigate to any of the specific model directories inside `src/` to start training.

## Sample Outputs

### DCGAN (Generated Faces)

![DCGAN Epoch 50](src/DCGAN/outputs/epoch_050.png)

### SRGAN (Super Resolution)

Here is a comparison showing the Low Resolution input, the Super Resolved output (Fake HR), and the Ground Truth (Real HR).

|              Low Res (16x16)              |       Fake HR - by epoch 10 (Generated 64x64)       |               Real HR (Target 64x64)                |
| :---------------------------------------: | :-------------------------------------------------: | :-------------------------------------------------: |
| ![LR](src/SRGAN/outputs/lr/epoch_010.png) | ![Fake HR](src/SRGAN/outputs/fake_hr/epoch_010.png) | ![Real HR](src/SRGAN/outputs/real_hr/epoch_010.png) |
