# 🧠 Contrastive Minds — Self-Supervised Learning for Image Classification

This repository contains our course project **“Self-Supervised Learning for Image Classification”**, where we reproduce the **SimCLR** framework (Chen et al., ICML 2020) and apply it to a small-scale image classification task (CIFAR-10 / STL-10).  
The project demonstrates how contrastive learning can extract meaningful visual representations without using human-labeled data.

---

## 📘 Project Overview

| Item | Description |
|------|--------------|
| **Framework** | PyTorch |
| **Model** | SimCLR (ResNet-18 backbone + 2-layer MLP projection head) |
| **Dataset** | CIFAR-10 (unlabeled pretraining) + labeled fine-tuning |
| **Goal** | Learn transferable image representations via contrastive self-supervision |
| **Group Name** | Contrastive Minds |

---

## 🚀 Objectives

1. Reproduce the **SimCLR** self-supervised training pipeline.
2. Implement the **NT-Xent contrastive loss** for positive and negative pairs.
3. Evaluate learned representations using a **linear classifier** on downstream tasks.
4. Compare SSL results with a **fully supervised baseline**.

---

## 🧩 Methodology

### 1. Data Augmentation
We follow SimCLR’s strong augmentation strategy:
- Random Resized Crop  
- Random Horizontal Flip  
- Color Jitter (brightness, contrast, saturation, hue)  
- Gaussian Blur  
- Normalization  

### 2. Model Architecture
- Encoder: **ResNet-18**
- Projection Head: **2-layer MLP** (Linear → ReLU → Linear)
- Loss: **NT-Xent** (Normalized Temperature-scaled Cross-Entropy)
- Optimizer: Adam / LARS  
- Temperature parameter: τ = 0.5

### 3. Training & Evaluation
1. Pretrain the model using unlabeled data with contrastive learning.  
2. Freeze the encoder and train a linear classifier on top.  
3. Evaluate classification accuracy and visualize feature embeddings (e.g., t-SNE).

---

## 📊 Expected Results

- Lower contrastive loss and meaningful feature clustering.
- SSL-pretrained model performs significantly better than random initialization.
- Visualization shows semantic grouping of similar images.

---

## 👥 Team Members

| Name | Contribution |
|------|---------------|
| **Member 1** | Data preprocessing and augmentation pipeline |
| **Member 2** | Model implementation (encoder & projection head) |
| **Member 3** | Training, evaluation, and report writing |

---

## 📁 Repository Structure

# Self-Supervised-Learning-for-Image-Classification
