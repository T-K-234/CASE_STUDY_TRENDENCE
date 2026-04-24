# 🧠 Self-Pruning Neural Network (AI Engineering Case Study)

## 📌 Overview

This project implements a **Self-Pruning Neural Network** that learns to remove unnecessary connections **during training itself**, instead of relying on post-training pruning.

The model is trained on the **CIFAR-10 dataset** and uses a **custom gating mechanism** with L1 regularization to enforce sparsity.

---

## 🚀 Key Idea

Each weight in the neural network is associated with a **learnable gate**:

* Gate value ∈ [0, 1]
* If gate → 0 → connection is pruned ❌
* If gate → 1 → connection is retained ✅

This allows the network to dynamically **adapt its architecture** while learning.

---

## 🏗️ Architecture

### 🔹 Custom Layer: `PrunableLinear`

* Extends standard linear layer
* Introduces **gate_scores** (learnable parameters)
* Uses sigmoid to generate gates

```
gates = sigmoid(gate_scores)
pruned_weights = weights * gates
```

---

## 🧪 Loss Function

Total Loss is defined as:

```
Total Loss = Classification Loss + λ × Sparsity Loss
```

### 🔹 Components:

* **Classification Loss** → CrossEntropyLoss
* **Sparsity Loss** → L1 norm of gate values

This encourages many gates to become **exactly zero**, resulting in a sparse model.

## 🧰 Tech Stack

* Python 🐍
* PyTorch 🔥
* Torchvision
* NumPy
* Matplotlib



## ▶️ How to Run

### 1. Install dependencies

```
pip install torch torchvision matplotlib
```

### 2. Run training

```
python main.py
```

---

## 📌 Output

* Training logs
* Test accuracy
* Sparsity percentage
* Gate distribution plot

---

## 🧠 Key Learnings

* Implemented **custom neural network layers**
* Designed **differentiable pruning mechanism**
* Applied **L1 regularization for sparsity**
* Explored **accuracy vs sparsity trade-off**
* Built end-to-end training + evaluation pipeline



## 👨‍💻 Author

**Tharun Kumar**

---

