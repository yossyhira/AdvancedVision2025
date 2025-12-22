[日本語](README.md) || [English](README_EN.md)

# AdvancedVision2025

This repository contains an assignment developed for the **Advanced Vision** course
(Graduate Program in Mirai Robotics, Chiba Institute of Technology, 2025).

The implementation is provided as a **Jupyter Notebook (`.ipynb`)** written in
**Python using PyTorch**.
Execution has not been verified on local environments; the code is intended to be run
on **Google Colab**.

---

## Overview

This model is a convolutional neural network (CNN) that performs
**10-class image classification** using **28×28 grayscale images** as input.

For performance evaluation, the model was trained on the **MNIST dataset**
for **5 epochs**, achieving an **accuracy of 98.6%** and a **loss of 4.3%**.

---

## Features

- MNIST dataset (28×28 grayscale images)
- Intermediate layers: ReLU activation
- Output layer: Softmax
- Directly executable on Google Colab

---

## Network Architecture

The model consists of **two convolutional layers with pooling layers**
followed by **three fully connected layers**.
ReLU activation functions are used in the intermediate layers, and
**10-class classification based on Softmax** is performed at the final layer.

A block diagram of the network architecture is shown below.

<p align="center">
  <img src="./README_Fig/block.png" width="500">
</p>

---

## Execution method (Google Colab)

1. You can directly open the notebook on Google Colab using the button below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yossyhira/AdvancedVision2025/blob/main/adv.ipynb)

2. Set the runtime type to **Python 3**, and configure the hardware accelerator
   to **GPU (Tesla T4 recommended)** as shown in the figure below.

<p align="center">
  <img src="./README_Fig/Colab_jp.png" width="500">
</p>

3. Execute the notebook cells sequentially from top to bottom.

---

## Execution Environment

- Platform: Google Colab
- GPU: NVIDIA Tesla T4
- Python: 3.12.12
- PyTorch: 2.9.0+cu126
- torchvision: 0.24.0+cu126
- NumPy: 2.0.2
- Matplotlib: 3.10.0
- scikit-learn: 1.6.1
- CUDA: 12.6

---

## Performance Evaluation

The performance of this model was evaluated using the **MNIST dataset**.
The MNIST dataset consists of handwritten digit images (digits 0–9),
with **60,000 training samples** and **10,000 test samples**.

The training procedure and evaluation results are described below.

---

### Training

The 60,000 training samples were split into **training and validation sets**
with an **8:2 ratio**
(48,000 training images and 12,000 validation images).

The following figure shows the transition of **accuracy** and **loss**
during the 5-epoch training process.

<p align="center">
  <img src="./README_Fig/accloss.png" width="1000">
</p>

---

### Results

The following results were obtained by applying the trained model
to the test dataset.

#### Accuracy

- Test accuracy: **98.6%**
- Test loss: **4.3%**

---

#### Confusion Matrix

The confusion matrix below visualizes the prediction results on the test dataset.
The vertical axis represents the **ground-truth labels**, and the horizontal axis
represents the **predicted labels** produced by the model.

<p align="center">
  <img src="./README_Fig/mat.png" width="750">
</p>

---

#### Correct / Incorrect Classification Examples

The top row shows **correctly classified examples**, and
the bottom row shows **misclassified examples**.

<p align="center">
  <img src="./README_Fig/exp.png" width="500">
</p>

---

## References

- The MNIST dataset used for performance evaluation was originally created by
  **Y. LeCun et al.** and was obtained via
  [torchvision.datasets.MNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

- The following resource was referenced for implementing CNN models in Python:
  - [畳み込みニューラルネットワーク(CNN)をわかりやすく基本から実装まで解説](https://zero2one.jp/learningblog/cnn-for-beginners/)

- The network architectures were designed with reference to the following papers:
  - [LeNet](https://direct.mit.edu/neco/article-abstract/1/4/541/5515/Backpropagation-Applied-to-Handwritten-Zip-Code)
  - [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

- The block diagram of the CNN architecture was referenced from:
  - [File:AlexNet_block_diagram.svg](https://commons.wikimedia.org/wiki/File:AlexNet_block_diagram.svg)

---

## License

This software is released under the **BSD 3-Clause License**.

© 2025 Yoshitaka Hirata
