---
layout: default
title: "Lab 2 — Image Classification"
---

# Lab 2 — Image Classification

## Goals

After completing this laboratory you should be able to:

- build image classifiers in PyTorch;
- compare a single-layer perceptron, a fully-connected deep network and a CNN;
- understand the role of training, validation and test sets;
- recognize overfitting and apply regularization;
- evaluate a classifier using accuracy and a confusion matrix.

---

## 1. Preparation

Review:

- multinomial classification and softmax;
- fully-connected neural networks;
- convolutional layers and pooling;
- overfitting and regularization;
- training, validation and test sets.

---

## 2. Environment

The notebook uses:

- Python 3
- NumPy
- Matplotlib
- PyTorch
- torchvision

GPU is optional.

---

## 3. Experiments

We use the MNIST dataset of handwritten digits and compare three neural-network models:

- a single-layer **Perceptron**;
- a **Deep** fully-connected network with one hidden layer;
- a **Convolutional Neural Network (CNN)**.

Train the models and compare their training, validation and test performance.

---

## 4. Tasks

- Apply regularization to the **Deep** model to reduce overfitting. Start with nonzero `weight_decay` ($L^2$ regularization) in the optimizer.
- Explain why the validation loss of the **CNN** can be lower than the training loss. Test your explanation by turning off **dropout**.
- Tune one of the models to obtain **Test Set Accuracy > 99%**.
- Plot the **confusion matrix** for all classes. Which digits are most often confused with each other?

For the confusion matrix, rows represent the ground-truth classes and columns the predicted classes. For example, element $(0,0)$ counts images of digit 0 correctly classified as 0, while element $(0,4)$ counts images of digit 0 incorrectly classified as 4.

---

## 5. Generalization to wallpaper groups

Repeat the classifier training on a [dataset of 2D crystallographic structures](https://drive.google.com/file/d/1BUz9eZdU-8wMGkEEmPk1IIUi05pH5ZUK/view?usp=sharing).

- Can we extract similarities between classes from the confusion matrix?

---

## 6. Notebook

Notebook: [`CNNs.ipynb`](CNNs.ipynb)

[← Laboratory list](../) · [Course page](https://jarek-pawlowski.github.io/MLphys_2026)
