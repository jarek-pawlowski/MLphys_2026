---
layout: default
title: "Lab 1 — Preliminary Problems — Neural-Network Fundamentals"
---

# Lab 1 — Preliminary Problems — Neural-Network Fundamentals

## Goals

After completing this laboratory you should be able to:

- construct and train simple perceptron and MLP models in PyTorch;
- understand why a single perceptron cannot represent XOR;
- relate network width and depth to model capacity and expressivity;
- understand the basic idea of the Universal Approximation Theorem;
- distinguish expressivity from optimization and generalization.

---

## 1. Preparation

Review:

- perceptrons and multilayer perceptrons (MLPs);
- activation functions;
- loss functions, gradient descent and backpropagation;
- the basic statement of the Universal Approximation Theorem.

---

## 2. Environment

The notebook uses:

- Python 3
- NumPy
- Matplotlib
- PyTorch

GPU is recommended.
The task can be completed using [google colab](https://colab.research.google.com/gist/jarek-pawlowski/d2f6b2c34fb91b985b51be9ee604af2e/preliminary_problems.ipynb)

---

## 3. Exercises

### Exercise 1 — Logic gates and perceptrons

Train a single perceptron to reproduce the OR, AND and XOR logic gates. Compare the results and explain why XOR cannot be represented by a single perceptron.

Then stack perceptrons into a simple deep network and train it on XOR.

**Tasks:**

- draw the *learned* decision boundary in the \(X_1,X_2\) space;
- train a neural network to add two binary numbers or multiply a binary number by two (remember the carry bit);
- two-digit numbers are sufficient, but you may try larger numbers and test generalization.

### Exercise 2 — Function approximation and expressivity

Use MLPs to approximate the function introduced in the notebook and investigate how network architecture affects the quality of the approximation.

Compare two MLPs:

- **shallow:** 1 hidden layer (2 `Linear` layers);
- **deep:** 3 hidden layers (4 `Linear` layers).

Design the networks so that the number of trainable parameters is similar in both cases. For example, the shallow network has \(3w+1\) parameters — do you understand why?

Use

```python
shallow_widths = [4, 8, 16, 32, 64, 128]
```

and adjust the deep-network width accordingly. Increase the number of parameters, train each model and plot the prediction MSE as a function of the number of trainable parameters for both architectures.

Compare training for 5k and 50k epochs. Why does this matter when comparing expressivity?

### Exercise 3 — Generalization: a gap in the data

Remove all training points from the interval

\[
x\in[-0.5,0.5]
\]

and train MLPs with different widths and depths on the remaining data.

Compare their predictions inside the missing region.

**Questions:**

- Does increasing the number of parameters improve the prediction inside the gap?
- Does higher expressivity necessarily imply better generalization?
- What determines the network prediction in a region where it has seen no data?

---

## 4. Main concepts

The experiments in this laboratory illustrate three related but distinct properties of neural networks:

\[
\text{expressivity} \neq \text{optimization} \neq \text{generalization}.
\]

A network may have enough capacity to represent a function without being easy to train, and increasing model capacity does not by itself guarantee better predictions in regions not constrained by training data.

---

## 5. Notebook

Notebook: [`preliminary_problems.ipynb`](preliminary_problems.ipynb)

[← Laboratory list](../) · [Course page]({{ site.baseurl }}/)
