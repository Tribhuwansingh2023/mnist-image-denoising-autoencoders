# Image Denoising Using Autoencoders (MNIST)
### Deep Learning Laboratory Assignment 04 | Empirical Study, Benchmarking & Deployment

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21.0-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

An end-to-end deep learning project implementing and systematically comparing **five distinct autoencoder paradigms** for removing additive white Gaussian noise from handwritten digits in the MNIST dataset ($28 \times 28$ grayscale images).

---

## 🌟 Key Features

1. **5 Autoencoder Architectures Implemented & Compared**:
   - **Undercomplete Autoencoder**: Strict dimensional bottleneck ($k = 32 \ll 784$).
   - **Sparse Autoencoder**: $L_1$ activity regularization on latent representations ($\lambda = 10^{-5}$).
   - **Denoising Autoencoder (DAE)**: Trained with stochastic dropout ($p = 0.2$) on noisy inputs $\to$ clean targets.
   - **Convolutional Autoencoder (CAE)**: Preserves 2D spatial topology via Conv2D, MaxPooling2D, and UpSampling2D kernels.
   - **Contractive Autoencoder (CAE-Jac)**: Frobenius norm penalty on the encoder's Jacobian matrix ($\lambda = 10^{-4}$).

2. **Controlled Noise Studies**:
   - Evaluated across noise factors $\sigma \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$.
   - Analyzed latent representation capacities for $k \in \{8, 16, 32, 64, 128\}$.

3. **Multi-Metric Quantitative Evaluation**:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Peak Signal-to-Noise Ratio (PSNR in dB)
   - Structural Similarity Index (SSIM)

4. **Production Deployment**:
   - **Interactive Frontend (Streamlit)**: Custom image upload, MNIST digit gallery, live noise slider, side-by-side comparison, and one-click denoised image download.
   - **RESTful Backend (FastAPI)**: Swagger documentation, health checks, and JSON/binary image endpoints.

---

## 📊 Experimental Results (Test Set, $\sigma = 0.5$)

Baseline Corrupted Noisy Images ($\sigma = 0.5$): $\text{MSE} = 0.11557$, $\text{MAE} = 0.20410$, $\text{PSNR} = 9.37\text{ dB}$, $\text{SSIM} = 0.3610$.

| Model Architecture | Trainable Parameters | Train Time (s) | MSE ($\downarrow$) | MAE ($\downarrow$) | PSNR (dB, $\uparrow$) | SSIM ($\uparrow$) | Key Architectural Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Denoising AE (Top Performer)** | 484,944 | 62.27 | **0.01856** | **0.05258** | **17.31 dB** | **0.7584** | **Best overall denoising**; sharp edges and clean backgrounds |
| **Sparse AE** | 218,192 | 43.64 | 0.02103 | 0.05992 | 16.77 dB | 0.7015 | Selective activation keeps latent codes sparse |
| **Undercomplete AE** | 222,384 | 48.96 | 0.02272 | 0.06132 | 16.44 dB | 0.7012 | Effective compression but minor background blur |
| **Contractive AE** | 101,200 | 38.31 | 0.05823 | 0.14051 | 12.35 dB | 0.2384 | High robustness; slight attenuation on digit strokes |
| **Convolutional AE** | 8,001 | 275.83 | 0.11396 | 0.13252 | 9.43 dB | 0.2557 | Ultra-lightweight (1.6% of DAE parameters) |

---

## 🚀 Quickstart & Setup

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Tribhuwansingh2023/mnist-image-denoising-autoencoders.git
cd mnist-image-denoising-autoencoders
pip install -r requirements.txt
```

### 2. Run the Interactive Web Frontend
Launch the Streamlit web application:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Run the REST API Backend
Launch the FastAPI inference server:
```bash
uvicorn backend_api:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000/docs`** for interactive Swagger documentation.

### 4. Run the Full Training Pipeline
To retrain models from scratch and regenerate all figures:
```bash
python train_and_evaluate.py
```

---

## 📁 Repository Structure

```
├── Lab4_Image_Denoising_Autoencoders.ipynb  # Executed notebook with all outputs & plots
├── LAB_REPORT_04.md                         # Comprehensive 21-section academic lab report
├── train_and_evaluate.py                    # Master training, evaluation & plotting pipeline
├── app.py                                   # Interactive Streamlit frontend web app
├── backend_api.py                           # FastAPI REST backend for model inference
├── models/
│   ├── best_denoising_autoencoder.keras     # Serialized top-performing DAE model
│   ├── sample_test_digits.npz               # Cached test arrays for UI demo
│   └── *.weights.h5                         # Saved weights for all 5 autoencoders
├── outputs/
│   ├── loss_curves.png                      # Training & validation convergence plots
│   ├── reconstruction_10_samples_all_models.png # 7x10 qualitative restoration gallery
│   ├── metrics_comparison_bar.png           # Comparative bar charts (MSE, MAE, PSNR, SSIM)
│   ├── noise_factor_analysis.png            # Noise severity curves (0.1 to 0.5)
│   └── latent_dim_analysis.png              # Bottleneck scaling curves (8 to 128)
└── README.md                                # Project documentation
```

---

## 📜 Academic Deliverables

* **Executed Jupyter Notebook**: [Lab4_Image_Denoising_Autoencoders.ipynb](Lab4_Image_Denoising_Autoencoders.ipynb)
* **Formal Lab Report**: [LAB_REPORT_04.md](LAB_REPORT_04.md)
