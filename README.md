# Fish-Disease-MAE

Official PyTorch implementation of the paper: **"A Self-supervised Learning Based Method for Fish Disease Diagnosis Utilizing Extremely Limited Samples"**.

This repository focuses on the **Pre-training Stage** of our framework, designed to extract discriminative pathological features from limited, unannotated aquatic imagery.

---

## 🔬 Methodology: Generative Pre-training

To overcome the domain shift of general-purpose models and the lack of high-quality labels, we propose a domain-specific **Asymmetric Masked Autoencoder (MAE)**. Unlike standard MAE, our framework is optimized to capture **microscopic textural deviations** rather than just global shapes.

### 1. Zero-shot Background Denoising
We utilize a "Localize First, Segment Later" strategy to eliminate complex aquatic backgrounds. By combining **Grounding-DINO** (detection) and **SAM** (segmentation), fish bodies are automatically extracted and placed on pure white backgrounds, ensuring the model focuses exclusively on intrinsic pathological lesions.

### 2. Adaptive Multi-View Hybrid Loss
This is the core of our pre-training. To force the decoder to reconstruct sharp, high-frequency pathological details, we define the total loss $\mathcal{L}_{total}$ as:

$$\mathcal{L}_{total} = \mathcal{L}_{pixel} + \lambda_{dct}\mathcal{L}_{dct} + \lambda_{perc}\mathcal{L}_{perc} + \lambda_{grad}\mathcal{L}_{grad}$$

* **$\mathcal{L}_{pixel}$ (Spatial Domain):** Standard MSE loss to ensure basic structural reconstruction.
* **$\mathcal{L}_{dct}$ (Frequency Domain):** We apply a Discrete Cosine Transform (DCT) and assign higher weights to high-frequency components using a distance-based power function. This forces the model to recover subtle lesion textures.
* **$\mathcal{L}_{perc}$ (Semantic Domain):** Utilizes a frozen VGG-19 to maintain perceptual consistency between the original and reconstructed images, preserving fine-grained pathological features.
* **$\mathcal{L}_{grad}$ (Gradient Domain):** Constrains the gradient difference to suppress checkerboard artifacts and sharpen the edges of pathological regions.

---

## 🖼️ Principles & Visualizations

### MAE Architecture
Our MAE employs an asymmetric design: a heavy **ViT-Small** encoder and a lightweight decoder. A high masking ratio (65%) is used to force the model to learn a robust internal representation of fish pathology.

<p align="center">
  <img src="docs/mae_architecture.pdf" width="800" alt="MAE Architecture">
  <br>
  <em>Figure 1: The proposed asymmetric MAE architecture with hybrid loss constraints.</em>
</p>

---

## 📁 Repository Structure
```text
├── config.py           # Hyperparameters & Loss weights
├── dataset.py          # In-memory data loader for accelerated training
├── losses.py           # Implementation of DCT, Perceptual, and Gradient losses
├── preprocess.py       # Grounding-DINO + SAM zero-shot denoising pipeline
├── train.py            # Main pre-training script with Mixed Precision (AMP)
└── utils.py            # Visualization tools for the 3-column reconstruction plots
