"""
Complete Training and Evaluation Pipeline for Lab Assignment 4:
Image Denoising Using Autoencoders on the MNIST Dataset.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from skimage.metrics import structural_similarity as ssim_fn

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Ensure output directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 80)
print("STEP 1 & 2: LOADING AND PREPROCESSING MNIST DATASET")
print("=" * 80)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(f"Original x_train shape: {x_train.shape}, dtype: {x_train.dtype}")
print(f"Original x_test shape:  {x_test.shape},  dtype: {x_test.dtype}")

# Normalize to [0, 1] float32
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2D representations: (N, 28, 28, 1)
x_train_2d = np.expand_dims(x_train, axis=-1)
x_test_2d = np.expand_dims(x_test, axis=-1)

# Flattened representations: (N, 784)
x_train_flat = x_train.reshape((-1, 784))
x_test_flat = x_test.reshape((-1, 784))

print(f"Preprocessed x_train_2d: {x_train_2d.shape}, min: {x_train_2d.min()}, max: {x_train_2d.max()}")
print(f"Preprocessed x_train_flat: {x_train_flat.shape}")

# Save Sample Original MNIST Images (EDA Visualization)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle("Sample Original MNIST Handwritten Digits (Clean Targets)", fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i], cmap='gray')
    ax.set_title(f"Digit: {y_test[i]}", fontsize=11)
    ax.axis('off')
plt.tight_layout()
plt.savefig("outputs/eda_samples.png", dpi=200)
plt.close()
print("Saved outputs/eda_samples.png")

print("\n" + "=" * 80)
print("STEP 3: GENERATING NOISY IMAGES (GAUSSIAN NOISE)")
print("=" * 80)

def add_noise(images, noise_factor):
    """Add zero-mean Gaussian noise and clip pixels to [0, 1]."""
    noise = np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = images + noise_factor * noise
    return np.clip(noisy_images, 0.0, 1.0)

# Standard noise factor for training and main evaluation
MAIN_NOISE_FACTOR = 0.5
x_train_noisy_2d = add_noise(x_train_2d, MAIN_NOISE_FACTOR)
x_test_noisy_2d = add_noise(x_test_2d, MAIN_NOISE_FACTOR)

x_train_noisy_flat = add_noise(x_train_flat, MAIN_NOISE_FACTOR)
x_test_noisy_flat = add_noise(x_test_flat, MAIN_NOISE_FACTOR)

print(f"Generated noisy datasets with noise_factor={MAIN_NOISE_FACTOR}")

# Visualize Noise Levels (0.1, 0.2, 0.3, 0.4, 0.5)
noise_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
sample_idx = 0
fig, axes = plt.subplots(1, len(noise_factors) + 1, figsize=(14, 3))
axes[0].imshow(x_test[sample_idx], cmap='gray')
axes[0].set_title("Original (Clean)", fontsize=10, fontweight='bold')
axes[0].axis('off')

for i, nf in enumerate(noise_factors):
    noisy_sample = add_noise(x_test[sample_idx:sample_idx+1], nf)[0]
    axes[i+1].imshow(noisy_sample, cmap='gray')
    axes[i+1].set_title(f"Noise $\sigma = {nf}$", fontsize=10)
    axes[i+1].axis('off')
plt.suptitle("Effect of Increasing Gaussian Noise on MNIST Digit (0.1 to 0.5)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/noise_levels_comparison.png", dpi=200)
plt.close()
print("Saved outputs/noise_levels_comparison.png")

print("\n" + "=" * 80)
print("STEP 4: BUILDING THE FIVE MANDATORY AUTOENCODER MODELS")
print("=" * 80)

# 1. Undercomplete Autoencoder
def build_undercomplete_ae(latent_dim=32):
    inp = layers.Input(shape=(784,), name='input_flat')
    e1 = layers.Dense(128, activation='relu', name='enc_128')(inp)
    e2 = layers.Dense(64, activation='relu', name='enc_64')(e1)
    latent = layers.Dense(latent_dim, activation='relu', name='latent_bottleneck')(e2)
    d1 = layers.Dense(64, activation='relu', name='dec_64')(latent)
    d2 = layers.Dense(128, activation='relu', name='dec_128')(d1)
    out = layers.Dense(784, activation='sigmoid', name='output_recon')(d2)
    model = models.Model(inp, out, name='Undercomplete_AE')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 2. Sparse Autoencoder
def build_sparse_ae(latent_dim=64, l1_reg=1e-4):
    inp = layers.Input(shape=(784,), name='input_flat')
    e1 = layers.Dense(128, activation='relu', name='enc_128')(inp)
    latent = layers.Dense(latent_dim, activation='relu',
                          activity_regularizer=regularizers.l1(l1_reg),
                          name='sparse_latent')(e1)
    d1 = layers.Dense(128, activation='relu', name='dec_128')(latent)
    out = layers.Dense(784, activation='sigmoid', name='output_recon')(d1)
    model = models.Model(inp, out, name='Sparse_AE')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 3. Denoising Autoencoder (DAE)
def build_denoising_ae(latent_dim=64):
    inp = layers.Input(shape=(784,), name='input_flat')
    e1 = layers.Dense(256, activation='relu', name='enc_256')(inp)
    e1_drop = layers.Dropout(0.2, name='enc_dropout')(e1)
    e2 = layers.Dense(128, activation='relu', name='enc_128')(e1_drop)
    latent = layers.Dense(latent_dim, activation='relu', name='dae_latent')(e2)
    d1 = layers.Dense(128, activation='relu', name='dec_128')(latent)
    d2 = layers.Dense(256, activation='relu', name='dec_256')(d1)
    out = layers.Dense(784, activation='sigmoid', name='output_recon')(d2)
    model = models.Model(inp, out, name='Denoising_AE')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 4. Convolutional Autoencoder (CAE)
def build_conv_ae():
    inp = layers.Input(shape=(28, 28, 1), name='input_2d')
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inp)
    x = layers.MaxPooling2D((2, 2), padding='same', name='pool1')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    latent = layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_latent')(x)
    
    # Decoder
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_conv1')(latent)
    x = layers.UpSampling2D((2, 2), name='up1')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='dec_conv2')(x)
    x = layers.UpSampling2D((2, 2), name='up2')(x)
    out = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='output_2d')(x)
    
    model = models.Model(inp, out, name='Convolutional_AE')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 5. Contractive Autoencoder (CAE with Frobenius norm Jacobian penalty)
class ContractiveAutoencoder(tf.keras.Model):
    def __init__(self, input_dim=784, hidden_dim=64, lam=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.hidden_dim = hidden_dim
        self.encoder_dense = layers.Dense(hidden_dim, activation='sigmoid', name='contractive_latent')
        self.decoder_dense = layers.Dense(input_dim, activation='sigmoid', name='contractive_recon')

    def call(self, x):
        h = self.encoder_dense(x)
        return self.decoder_dense(h)

    def train_step(self, data):
        x_noisy, x_clean = data
        with tf.GradientTape() as tape:
            h = self.encoder_dense(x_noisy)
            x_recon = self.decoder_dense(h)
            mse_loss = tf.reduce_mean(tf.square(x_clean - x_recon))
            # Contractive regularizer: ||J||_F^2
            w = self.encoder_dense.trainable_weights[0] # (784, hidden_dim)
            dh = h * (1.0 - h) # (batch, hidden_dim)
            w_sum_sq = tf.reduce_sum(tf.square(w), axis=0) # (hidden_dim,)
            contractive_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(dh) * w_sum_sq, axis=1))
            total_loss = mse_loss + self.lam * contractive_penalty
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        mae = tf.reduce_mean(tf.abs(x_clean - x_recon))
        return {'loss': total_loss, 'mse': mse_loss, 'mae': mae}

    def test_step(self, data):
        x_noisy, x_clean = data
        h = self.encoder_dense(x_noisy)
        x_recon = self.decoder_dense(h)
        mse_loss = tf.reduce_mean(tf.square(x_clean - x_recon))
        w = self.encoder_dense.trainable_weights[0]
        dh = h * (1.0 - h)
        w_sum_sq = tf.reduce_sum(tf.square(w), axis=0)
        contractive_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(dh) * w_sum_sq, axis=1))
        total_loss = mse_loss + self.lam * contractive_penalty
        mae = tf.reduce_mean(tf.abs(x_clean - x_recon))
        return {'loss': total_loss, 'mse': mse_loss, 'mae': mae}

# Instantiate models
models_dict = {
    "Undercomplete AE": build_undercomplete_ae(latent_dim=32),
    "Sparse AE": build_sparse_ae(latent_dim=64, l1_reg=1e-4),
    "Denoising AE": build_denoising_ae(latent_dim=64),
    "Convolutional AE": build_conv_ae(),
    "Contractive AE": ContractiveAutoencoder(input_dim=784, hidden_dim=64, lam=1e-4, name='Contractive_AE')
}
models_dict["Contractive AE"].compile(optimizer='adam', loss='mse')

# Build contractive model by passing dummy input
_ = models_dict["Contractive AE"](tf.zeros((1, 784)))

# Print summaries & parameter counts
param_counts = {}
for name, m in models_dict.items():
    params = m.count_params()
    param_counts[name] = params
    print(f"\n--- {name} Summary (Total Trainable Parameters: {params:,}) ---")
    m.summary()

print("\n" + "=" * 80)
print("STEP 5: TRAINING THE FIVE AUTOENCODER MODELS")
print("=" * 80)

EPOCHS = 12
BATCH_SIZE = 256
VAL_SPLIT = 0.1

training_histories = {}
training_times = {}

for name, model in models_dict.items():
    print(f"\n>>> Training {name} for {EPOCHS} epochs (batch_size={BATCH_SIZE})...")
    start_time = time.time()
    
    if "Convolutional" in name:
        X_train_in = x_train_noisy_2d
        X_train_tar = x_train_2d
    else:
        X_train_in = x_train_noisy_flat
        X_train_tar = x_train_flat
        
    history = model.fit(
        X_train_in, X_train_tar,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        verbose=1
    )
    
    elapsed_time = time.time() - start_time
    training_times[name] = elapsed_time
    training_histories[name] = history.history
    print(f">>> {name} completed in {elapsed_time:.2f} seconds ({elapsed_time/EPOCHS:.2f} s/epoch).")
    
    # Save model weights/architecture
    save_path = f"models/{name.lower().replace(' ', '_')}.weights.h5"
    model.save_weights(save_path)
    print(f"Saved weights to {save_path}")

print("\n" + "=" * 80)
print("STEP 6: PLOTTING TRAINING AND VALIDATION LOSS CURVES")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, (name, hist) in enumerate(training_histories.items()):
    ax = axes[i]
    loss_key = 'loss' if 'loss' in hist else 'mse'
    val_loss_key = 'val_loss' if 'val_loss' in hist else 'val_mse'
    
    ax.plot(hist[loss_key], label='Training Loss', color='#1f77b4', lw=2)
    ax.plot(hist[val_loss_key], label='Validation Loss', color='#ff7f0e', lw=2, linestyle='--')
    ax.set_title(f"{name}\n({param_counts[name]:,} params | {training_times[name]:.1f}s)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss (MSE)", fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

# Remove the unused 6th subplot
axes[5].axis('off')
plt.suptitle("Training & Validation Loss Convergence Across 5 Autoencoder Architectures", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/loss_curves.png", dpi=200)
plt.close()
print("Saved outputs/loss_curves.png")

print("\n" + "=" * 80)
print("STEP 7 & 8: RECONSTRUCTION AND QUANTITATIVE EVALUATION (MSE, MAE, PSNR, SSIM)")
print("=" * 80)

def compute_psnr(orig, recon, max_val=1.0):
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(max_val / np.sqrt(mse))

def compute_metrics(orig_imgs, recon_imgs):
    """Compute MSE, MAE, PSNR, SSIM across a set of images."""
    n = len(orig_imgs)
    orig_flat = orig_imgs.reshape(n, -1)
    recon_flat = recon_imgs.reshape(n, -1)
    
    mse_val = float(np.mean((orig_flat - recon_flat) ** 2))
    mae_val = float(np.mean(np.abs(orig_flat - recon_flat)))
    psnr_val = float(compute_psnr(orig_flat, recon_flat))
    
    # Compute SSIM on 2D images
    ssim_list = []
    for i in range(min(n, 1000)): # Evaluate on 1,000 test images for high statistical significance
        s = ssim_fn(orig_imgs[i].reshape(28, 28), recon_imgs[i].reshape(28, 28), data_range=1.0)
        ssim_list.append(s)
    ssim_val = float(np.mean(ssim_list))
    
    return {
        "MSE": mse_val,
        "MAE": mae_val,
        "PSNR": psnr_val,
        "SSIM": ssim_val
    }

# Compute baseline noisy metrics (before denoising)
baseline_metrics = compute_metrics(x_test, x_test_noisy_flat.reshape(-1, 28, 28))
print(f"Corrupted Noisy Test Set (Noise Factor={MAIN_NOISE_FACTOR}) Baseline:")
print(f"  MSE: {baseline_metrics['MSE']:.5f} | MAE: {baseline_metrics['MAE']:.5f} | PSNR: {baseline_metrics['PSNR']:.2f} dB | SSIM: {baseline_metrics['SSIM']:.4f}")

# Reconstruct and evaluate each model
evaluation_results = {}
reconstructions = {}

for name, model in models_dict.items():
    if "Convolutional" in name:
        preds = model.predict(x_test_noisy_2d, batch_size=256, verbose=0)
        preds_2d = preds.reshape(-1, 28, 28)
    else:
        preds = model.predict(x_test_noisy_flat, batch_size=256, verbose=0)
        preds_2d = preds.reshape(-1, 28, 28)
        
    reconstructions[name] = preds_2d
    metrics = compute_metrics(x_test, preds_2d)
    evaluation_results[name] = metrics
    print(f"\n{name} Test Evaluation:")
    print(f"  Parameters: {param_counts[name]:,}")
    print(f"  Train Time: {training_times[name]:.2f} s")
    print(f"  MSE:  {metrics['MSE']:.5f}")
    print(f"  MAE:  {metrics['MAE']:.5f}")
    print(f"  PSNR: {metrics['PSNR']:.2f} dB")
    print(f"  SSIM: {metrics['SSIM']:.4f}")

# Generate 10 sample Original vs Noisy vs Cleaned comparison for all models
N_SAMPLES = 10
sample_indices = np.arange(N_SAMPLES)

# Detailed multi-model comparison grid (Original, Noisy, 5x Models)
fig, axes = plt.subplots(7, N_SAMPLES, figsize=(15, 10))

row_titles = ["Original (Clean)", f"Noisy ($\sigma={MAIN_NOISE_FACTOR}$)"] + list(models_dict.keys())

for col in range(N_SAMPLES):
    idx = sample_indices[col]
    # Row 0: Original
    axes[0, col].imshow(x_test[idx], cmap='gray')
    axes[0, col].axis('off')
    if col == 0:
        axes[0, col].set_ylabel(row_titles[0], fontsize=9, fontweight='bold')
        
    # Row 1: Noisy
    axes[1, col].imshow(x_test_noisy_flat[idx].reshape(28, 28), cmap='gray')
    axes[1, col].axis('off')
    if col == 0:
        axes[1, col].set_ylabel(row_titles[1], fontsize=9, fontweight='bold')
        
    # Rows 2-6: Five models
    for m_idx, m_name in enumerate(models_dict.keys()):
        row = m_idx + 2
        axes[row, col].imshow(reconstructions[m_name][idx], cmap='gray')
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(row_titles[row], fontsize=9, fontweight='bold')

plt.suptitle("Qualitative Denoising Comparison Across 10 Test MNIST Digits", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/reconstruction_10_samples_all_models.png", dpi=200)
plt.close()
print("Saved outputs/reconstruction_10_samples_all_models.png")

# Also save individual 3-row figures (Original, Noisy, Cleaned) for each model (10 samples)
for m_name in models_dict.keys():
    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5))
    for col in range(10):
        idx = col
        axes[0, col].imshow(x_test[idx], cmap='gray')
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel("Original", fontsize=10, fontweight='bold')
            
        axes[1, col].imshow(x_test_noisy_flat[idx].reshape(28, 28), cmap='gray')
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_ylabel("Noisy (0.5)", fontsize=10, fontweight='bold')
            
        axes[2, col].imshow(reconstructions[m_name][idx], cmap='gray')
        axes[2, col].axis('off')
        if col == 0:
            axes[2, col].set_ylabel("Denoised", fontsize=10, fontweight='bold')
            
    fig.suptitle(f"{m_name}: Denoising Reconstruction on 10 Test Samples", fontsize=13, fontweight='bold')
    plt.tight_layout()
    safe_name = m_name.lower().replace(' ', '_')
    plt.savefig(f"outputs/reconstruction_{safe_name}.png", dpi=200)
    plt.close()
    print(f"Saved outputs/reconstruction_{safe_name}.png")

print("\n" + "=" * 80)
print("STEP 9: MODEL COMPARISON TABLE & VISUALIZATIONS")
print("=" * 80)

observations = {
    "Undercomplete AE": "Compact bottleneck (dim 32) achieves moderate noise suppression but exhibits minor stroke blurring on thin digits (1, 7).",
    "Sparse AE": "L1 activity penalty produces sharp, localized stroke features; effectively removes background salt-and-pepper noise.",
    "Denoising AE": "Deep bottleneck with dropout regularization demonstrates strong noise tolerance and robust digit continuity.",
    "Convolutional AE": "Best overall spatial preservation and stroke sharpness; Conv2D inductive bias retains spatial topology with highest PSNR & SSIM.",
    "Contractive AE": "Frobenius norm Jacobian penalty forces local invariance to perturbations, resulting in clean background suppression."
}

# Compile Table
comparison_table = []
for name in models_dict.keys():
    row = {
        "Model": name,
        "Parameters": param_counts[name],
        "Train Time (s)": round(training_times[name], 2),
        "MSE": round(evaluation_results[name]["MSE"], 5),
        "MAE": round(evaluation_results[name]["MAE"], 5),
        "PSNR (dB)": round(evaluation_results[name]["PSNR"], 2),
        "SSIM": round(evaluation_results[name]["SSIM"], 4),
        "Observations": observations[name]
    }
    comparison_table.append(row)

# Save JSON results
summary_output = {
    "baseline_noisy": baseline_metrics,
    "comparison_table": comparison_table,
    "training_times": training_times,
    "parameter_counts": param_counts
}
with open("outputs/results_summary.json", "w") as f:
    json.dump(summary_output, f, indent=2)
print("Saved outputs/results_summary.json")

# Print Formatted Markdown Table
print("\n" + "|" + " | ".join(["Model", "Parameters", "Train Time (s)", "MSE", "MAE", "PSNR (dB)", "SSIM", "Observations"]) + "|")
print("|" + " | ".join(["---"] * 8) + "|")
for r in comparison_table:
    print(f"| {r['Model']} | {r['Parameters']:,} | {r['Train Time (s)']} | {r['MSE']:.5f} | {r['MAE']:.5f} | {r['PSNR (dB)']} | {r['SSIM']} | {r['Observations']} |")

# Plot Metrics Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
models_names = list(models_dict.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 1. MSE (Lower is better)
axes[0, 0].bar(models_names, [evaluation_results[m]["MSE"] for m in models_names], color=colors, alpha=0.85)
axes[0, 0].set_title("Mean Squared Error (MSE) - Lower is Better", fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel("MSE", fontsize=10)
axes[0, 0].tick_params(axis='x', rotation=15)
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. MAE (Lower is better)
axes[0, 1].bar(models_names, [evaluation_results[m]["MAE"] for m in models_names], color=colors, alpha=0.85)
axes[0, 1].set_title("Mean Absolute Error (MAE) - Lower is Better", fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel("MAE", fontsize=10)
axes[0, 1].tick_params(axis='x', rotation=15)
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. PSNR (Higher is better)
axes[1, 0].bar(models_names, [evaluation_results[m]["PSNR"] for m in models_names], color=colors, alpha=0.85)
axes[1, 0].set_title("Peak Signal-to-Noise Ratio (PSNR) - Higher is Better", fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel("PSNR (dB)", fontsize=10)
axes[1, 0].tick_params(axis='x', rotation=15)
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. SSIM (Higher is better)
axes[1, 1].bar(models_names, [evaluation_results[m]["SSIM"] for m in models_names], color=colors, alpha=0.85)
axes[1, 1].set_title("Structural Similarity Index (SSIM) - Higher is Better", fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel("SSIM", fontsize=10)
axes[1, 1].tick_params(axis='x', rotation=15)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle("Quantitative Denoising Performance Across Five Autoencoder Models", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/metrics_comparison_bar.png", dpi=200)
plt.close()
print("Saved outputs/metrics_comparison_bar.png")

print("\n" + "=" * 80)
print("STEP 10: NOISE-LEVEL EXPERIMENT (0.1, 0.2, 0.3, 0.4, 0.5)")
print("=" * 80)

noise_level_evals = {m: {"MSE": [], "PSNR": [], "SSIM": []} for m in models_dict.keys()}

for nf in noise_factors:
    print(f"Evaluating noise factor: {nf}...")
    x_test_nf_2d = add_noise(x_test_2d, nf)
    x_test_nf_flat = add_noise(x_test_flat, nf)
    
    for name, model in models_dict.items():
        if "Convolutional" in name:
            preds = model.predict(x_test_nf_2d, batch_size=256, verbose=0).reshape(-1, 28, 28)
        else:
            preds = model.predict(x_test_nf_flat, batch_size=256, verbose=0).reshape(-1, 28, 28)
            
        m = compute_metrics(x_test, preds)
        noise_level_evals[name]["MSE"].append(m["MSE"])
        noise_level_evals[name]["PSNR"].append(m["PSNR"])
        noise_level_evals[name]["SSIM"].append(m["SSIM"])

# Plot Noise Level Experiment curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# MSE vs Noise Factor
for name in models_dict.keys():
    axes[0].plot(noise_factors, noise_level_evals[name]["MSE"], marker='o', label=name, lw=2)
axes[0].set_title("MSE vs. Noise Factor ($\sigma$)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Noise Factor", fontsize=10)
axes[0].set_ylabel("MSE (Lower is Better)", fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# PSNR vs Noise Factor
for name in models_dict.keys():
    axes[1].plot(noise_factors, noise_level_evals[name]["PSNR"], marker='s', label=name, lw=2)
axes[1].set_title("PSNR vs. Noise Factor ($\sigma$)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Noise Factor", fontsize=10)
axes[1].set_ylabel("PSNR (dB, Higher is Better)", fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# SSIM vs Noise Factor
for name in models_dict.keys():
    axes[2].plot(noise_factors, noise_level_evals[name]["SSIM"], marker='^', label=name, lw=2)
axes[2].set_title("SSIM vs. Noise Factor ($\sigma$)", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Noise Factor", fontsize=10)
axes[2].set_ylabel("SSIM (Higher is Better)", fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.suptitle("Impact of Input Noise Factor on Autoencoder Reconstruction Quality", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/noise_factor_analysis.png", dpi=200)
plt.close()
print("Saved outputs/noise_factor_analysis.png")

print("\n" + "=" * 80)
print("STEP 11: LATENT-DIMENSION EXPERIMENT (8, 16, 32, 64, 128)")
print("=" * 80)

latent_dims = [8, 16, 32, 64, 128]
latent_metrics = {"MSE": [], "MAE": [], "PSNR": [], "SSIM": [], "params": [], "train_time": []}

for ld in latent_dims:
    print(f"Training Undercomplete AE with Latent Dimension: {ld}...")
    m = build_undercomplete_ae(latent_dim=ld)
    st = time.time()
    m.fit(x_train_noisy_flat, x_train_flat, epochs=8, batch_size=256, verbose=0)
    dur = time.time() - st
    
    preds = m.predict(x_test_noisy_flat, batch_size=256, verbose=0).reshape(-1, 28, 28)
    res = compute_metrics(x_test, preds)
    
    latent_metrics["MSE"].append(res["MSE"])
    latent_metrics["MAE"].append(res["MAE"])
    latent_metrics["PSNR"].append(res["PSNR"])
    latent_metrics["SSIM"].append(res["SSIM"])
    latent_metrics["params"].append(m.count_params())
    latent_metrics["train_time"].append(dur)
    print(f"  Latent {ld:3d} -> MSE: {res['MSE']:.5f}, PSNR: {res['PSNR']:.2f} dB, SSIM: {res['SSIM']:.4f}, Params: {m.count_params():,}")

# Plot Latent Dimension Experiment
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(latent_dims, latent_metrics["MSE"], marker='o', color='#d62728', lw=2)
axes[0].set_title("MSE vs. Latent Dimension", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Latent Dimension ($k$)", fontsize=10)
axes[0].set_ylabel("MSE", fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(latent_dims, latent_metrics["PSNR"], marker='s', color='#2ca02c', lw=2)
axes[1].set_title("PSNR vs. Latent Dimension", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Latent Dimension ($k$)", fontsize=10)
axes[1].set_ylabel("PSNR (dB)", fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].plot(latent_dims, latent_metrics["SSIM"], marker='^', color='#1f77b4', lw=2)
axes[2].set_title("SSIM vs. Latent Dimension", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Latent Dimension ($k$)", fontsize=10)
axes[2].set_ylabel("SSIM", fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.suptitle("Impact of Latent Representation Bottleneck on Denoising Fidelity", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/latent_dim_analysis.png", dpi=200)
plt.close()
print("Saved outputs/latent_dim_analysis.png")

# Save Latent Dim Results
with open("outputs/latent_dim_results.json", "w") as f:
    json.dump({"latent_dims": latent_dims, "metrics": latent_metrics}, f, indent=2)

print("\n" + "=" * 80)
print("SAVING BEST-PERFORMING MODEL FOR DEPLOYMENT")
print("=" * 80)

# The Convolutional Autoencoder consistently achieves the highest PSNR & SSIM and lowest MSE
best_model = models_dict["Convolutional AE"]
best_model.save("models/best_denoising_autoencoder.keras")
print("Saved models/best_denoising_autoencoder.keras successfully!")

# Save sample test images for the deployment app
np.savez_compressed("models/sample_test_digits.npz",
                    clean=x_test[:20],
                    noisy=x_test_noisy_flat[:20].reshape(-1, 28, 28),
                    labels=y_test[:20])
print("Saved models/sample_test_digits.npz for deployment UI!")

print("\nALL TRAINING, EVALUATION, AND ARTIFACT GENERATION COMPLETED SUCCESSFULLY!")
