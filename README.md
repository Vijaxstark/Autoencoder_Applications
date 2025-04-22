# 🔍 Autoencoder Applications: MNIST Compression & Reconstruction

This project explores dimensionality reduction and image reconstruction on the MNIST dataset using two techniques:

- ✅ Principal Component Analysis (PCA)
- ✅ Deep Autoencoder (Neural Network)

Both methods are trained to compress MNIST digit images to **4-dimensional latent vectors**, and their outputs are compared visually and quantitatively.

## 🧠 Objectives

- Compress 28x28 MNIST images into 4-D latent vectors using PCA and Autoencoder.
- Visualize these latent vectors using t-SNE.
- Reconstruct original images from the compressed representations.
- Compare results across techniques and compute the **compression ratio**.

---

## 🗃 Dataset

- **MNIST Handwritten Digits** from `tf.keras.datasets`
- Total: 70,000 grayscale images of shape 28x28

---

## 🧪 Workflow

1. **Preprocessing**:
   - Normalize images to [0, 1]
   - Flatten from 28x28 → 784

2. **Compression**:
   - PCA with 4 components
   - Autoencoder with 4-dimensional bottleneck

3. **Visualization**:
   - Use t-SNE (`sklearn.manifold.TSNE`) to reduce latent vectors to 2D
   - Plot results for both PCA and Autoencoder

4. **Reconstruction**:
   - Decode latent vectors back to image space
   - Display original and reconstructed images for 20 samples

5. **Compression Ratio**:
   - `784 / 4 = 196 : 1`

---

## 📊 Results

- Visual comparison of PCA vs Autoencoder in terms of reconstruction quality
- Latent spaces visualized with t-SNE
- Discusses trade-offs in linear (PCA) vs nonlinear (Autoencoder) compression

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
# Then open and run the notebook:
Autoencoder_MNIST.ipynb
