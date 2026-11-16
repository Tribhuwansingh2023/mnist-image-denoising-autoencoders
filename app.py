"""
Lab Assignment 04: Interactive Image Denoising Application Using Autoencoders
Streamlit Web Deployment
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import io
import json
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(
    page_title="Neural Image Denoising Suite | MNIST Autoencoders",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished academic UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .workflow-pill {
        display: inline-block;
        padding: 6px 14px;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px 4px;
        border: 1px solid #BFDBFE;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-title">🧠 Deep Learning Image Denoising Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Lab Assignment 04: Comparative Analysis of Undercomplete, Sparse, Denoising, Convolutional, and Contractive Autoencoders on MNIST</div>', unsafe_allow_html=True)

# Workflow Banner
st.markdown("""
<div style="margin-bottom: 20px;">
    <span class="workflow-pill">1. Clean Image</span> ➔ 
    <span class="workflow-pill">2. Gaussian Noise Injection</span> ➔ 
    <span class="workflow-pill">3. Noisy Image</span> ➔ 
    <span class="workflow-pill">4. Encoder</span> ➔ 
    <span class="workflow-pill">5. Latent Bottleneck</span> ➔ 
    <span class="workflow-pill">6. Decoder</span> ➔ 
    <span class="workflow-pill">7. Denoised Reconstruction</span>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    """Load or build models and load their trained weights."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    
    # 1. Undercomplete AE
    def build_undercomplete():
        inp = layers.Input(shape=(784,), name='undercomplete_inp')
        e1 = layers.Dense(128, activation='relu')(inp)
        e2 = layers.Dense(64, activation='relu')(e1)
        lat = layers.Dense(32, activation='relu')(e2)
        d1 = layers.Dense(64, activation='relu')(lat)
        d2 = layers.Dense(128, activation='relu')(d1)
        out = layers.Dense(784, activation='sigmoid')(d2)
        m = models.Model(inp, out, name='Undercomplete_AE')
        if os.path.exists("models/undercomplete_ae.weights.h5"):
            m.load_weights("models/undercomplete_ae.weights.h5")
        return m

    # 2. Sparse AE
    def build_sparse():
        inp = layers.Input(shape=(784,), name='sparse_inp')
        e1 = layers.Dense(128, activation='relu')(inp)
        lat = layers.Dense(64, activation='relu', activity_regularizer=regularizers.l1(1e-4))(e1)
        d1 = layers.Dense(128, activation='relu')(lat)
        out = layers.Dense(784, activation='sigmoid')(d1)
        m = models.Model(inp, out, name='Sparse_AE')
        if os.path.exists("models/sparse_ae.weights.h5"):
            m.load_weights("models/sparse_ae.weights.h5")
        return m

    # 3. Denoising AE (Top Performer: Highest PSNR 17.31 dB & SSIM 0.7584)
    def build_dae():
        inp = layers.Input(shape=(784,), name='dae_inp')
        e1 = layers.Dense(256, activation='relu')(inp)
        e1_d = layers.Dropout(0.2)(e1)
        e2 = layers.Dense(128, activation='relu')(e1_d)
        lat = layers.Dense(64, activation='relu')(e2)
        d1 = layers.Dense(128, activation='relu')(lat)
        d2 = layers.Dense(256, activation='relu')(d1)
        out = layers.Dense(784, activation='sigmoid')(d2)
        m = models.Model(inp, out, name='Denoising_AE')
        if os.path.exists("models/denoising_ae.weights.h5"):
            m.load_weights("models/denoising_ae.weights.h5")
        return m

    # 4. Convolutional AE
    def build_cae():
        inp = layers.Input(shape=(28, 28, 1), name='cae_inp')
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        lat = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(lat)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        out = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        m = models.Model(inp, out, name='Convolutional_AE')
        if os.path.exists("models/convolutional_ae.weights.h5"):
            m.load_weights("models/convolutional_ae.weights.h5")
        return m

    # 5. Contractive AE
    class ContractiveAE(tf.keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.encoder_dense = layers.Dense(64, activation='sigmoid')
            self.decoder_dense = layers.Dense(784, activation='sigmoid')
        def call(self, x):
            return self.decoder_dense(self.encoder_dense(x))
            
    m_contractive = ContractiveAE(name='Contractive_AE')
    _ = m_contractive(tf.zeros((1, 784)))
    if os.path.exists("models/contractive_ae.weights.h5"):
        m_contractive.load_weights("models/contractive_ae.weights.h5")

    return {
        "Denoising AE (Recommended - Best Model | PSNR: 17.31 dB)": build_dae(),
        "Sparse AE (L1 Activity Regularized | PSNR: 16.77 dB)": build_sparse(),
        "Undercomplete AE (Bottleneck 32 | PSNR: 16.44 dB)": build_undercomplete(),
        "Contractive AE (Jacobian Invariant | PSNR: 12.35 dB)": m_contractive,
        "Convolutional AE (Spatial 2D Topology | 8,001 Params)": build_cae()
    }

# Load Models
try:
    models_pool = load_all_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure training has completed.")
    st.stop()

# Load preloaded sample test digits
@st.cache_data
def load_sample_digits():
    if os.path.exists("models/sample_test_digits.npz"):
        data = np.load("models/sample_test_digits.npz")
        return data['clean'], data['noisy'], data['labels']
    return None, None, None

sample_clean, sample_noisy, sample_labels = load_sample_digits()

# Helper Evaluation Functions
def calc_metrics(orig, recon):
    from skimage.metrics import structural_similarity as ssim_fn
    mse = float(np.mean((orig - recon) ** 2))
    mae = float(np.mean(np.abs(orig - recon)))
    psnr = float(100.0 if mse == 0 else 20.0 * np.log10(1.0 / np.sqrt(mse)))
    ssim = float(ssim_fn(orig, recon, data_range=1.0))
    return mse, mae, psnr, ssim

# Sidebar Controls
st.sidebar.header("⚙️ Configuration & Controls")

selected_model_name = st.sidebar.selectbox(
    "Select Autoencoder Architecture:",
    list(models_pool.keys()),
    index=0
)
active_model = models_pool[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Input Image Source")

input_mode = st.sidebar.radio(
    "Choose Input Method:",
    ["Pick from MNIST Test Samples", "Upload Custom Image", "Synthetic Noise Experimenter"]
)

clean_img = None
noisy_img = None

if input_mode == "Pick from MNIST Test Samples":
    if sample_clean is not None:
        digit_idx = st.sidebar.slider("Select Sample Index (0 - 19):", 0, len(sample_clean) - 1, 0)
        clean_img = sample_clean[digit_idx]
        digit_label = sample_labels[digit_idx]
        st.sidebar.info(f"Target Digit Label: **{digit_label}**")
        
        apply_custom_noise = st.sidebar.checkbox("Inject Additional Synthetic Gaussian Noise", value=True)
        if apply_custom_noise:
            noise_factor = st.sidebar.slider(r"Gaussian Noise Factor ($\sigma$):", 0.0, 1.0, 0.5, 0.05)
            noise = np.random.normal(loc=0.0, scale=1.0, size=clean_img.shape)
            noisy_img = np.clip(clean_img + noise_factor * noise, 0.0, 1.0)
        else:
            noisy_img = sample_noisy[digit_idx]
    else:
        st.sidebar.warning("Sample digits file not found. Please upload an image.")

elif input_mode == "Upload Custom Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert('L')
        pil_img = ImageOps.invert(pil_img) if st.sidebar.checkbox("Invert Colors (Dark Background / White Stroke)", value=True) else pil_img
        pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
        clean_img = np.array(pil_img).astype('float32') / 255.0
        
        noise_factor = st.sidebar.slider(r"Add Controlled Noise Factor ($\sigma$):", 0.0, 1.0, 0.5, 0.05)
        if noise_factor > 0:
            noise = np.random.normal(loc=0.0, scale=1.0, size=clean_img.shape)
            noisy_img = np.clip(clean_img + noise_factor * noise, 0.0, 1.0)
        else:
            noisy_img = clean_img.copy()

elif input_mode == "Synthetic Noise Experimenter":
    if sample_clean is not None:
        clean_img = sample_clean[3] # digit 0
    else:
        clean_img = np.zeros((28, 28), dtype='float32')
        clean_img[7:21, 10:18] = 1.0 # simple synthetic bar
    noise_factor = st.sidebar.slider(r"Noise Intensity ($\sigma$):", 0.05, 1.0, 0.5, 0.05)
    noise = np.random.normal(loc=0.0, scale=1.0, size=clean_img.shape)
    noisy_img = np.clip(clean_img + noise_factor * noise, 0.0, 1.0)

# Main Processing & Inference
if noisy_img is not None:
    # Dynamically inspect model input shape
    is_conv = False
    if hasattr(active_model, 'input_shape') and active_model.input_shape is not None:
        is_conv = (len(active_model.input_shape) == 4)
    elif "Convolutional" in selected_model_name:
        is_conv = True

    if is_conv:
        inp_tensor = noisy_img.reshape(1, 28, 28, 1)
        denoised_tensor = active_model.predict(inp_tensor, verbose=0)
        denoised_img = denoised_tensor[0, :, :, 0]
    else:
        inp_tensor = noisy_img.reshape(1, 784)
        denoised_tensor = active_model.predict(inp_tensor, verbose=0)
        denoised_img = np.array(denoised_tensor[0]).reshape(28, 28)

    # Difference residual (Noise removed)
    residual_noise = np.abs(noisy_img - denoised_img)

    # Quantitative Evaluation
    if clean_img is not None:
        noisy_mse, noisy_mae, noisy_psnr, noisy_ssim = calc_metrics(clean_img, noisy_img)
        clean_mse, clean_mae, clean_psnr, clean_ssim = calc_metrics(clean_img, denoised_img)
        
        psnr_gain = clean_psnr - noisy_psnr
        ssim_gain = clean_ssim - noisy_ssim
        mse_reduction = (1.0 - clean_mse / max(noisy_mse, 1e-8)) * 100.0

    # Display Visual Comparison Columns
    st.subheader("🖼️ Visual Reconstruction & Denoising Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**1. Original Clean Target**")
        if clean_img is not None:
            st.image(clean_img, caption="Ground Truth (28x28)", use_container_width=True, clamp=True)
        else:
            st.write("N/A (Direct Noisy Upload)")
            
    with col2:
        st.markdown("**2. Input Corrupted (Noisy)**")
        st.image(noisy_img, caption="Input with Gaussian Noise", use_container_width=True, clamp=True)
        
    with col3:
        st.markdown(f"**3. Denoised by Autoencoder**")
        st.image(denoised_img, caption=f"{selected_model_name.split('(')[0].strip()}", use_container_width=True, clamp=True)
        
    with col4:
        st.markdown(r"**4. Extracted Residual ($|Noisy - Cleaned|$)**")
        st.image(residual_noise, caption="Suppressed Noise Map", use_container_width=True, clamp=True)

    # Metric Scorecards
    if clean_img is not None:
        st.markdown("---")
        st.subheader("📊 Objective Quality Metrics")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        m_col1.metric(
            label="Peak Signal-to-Noise Ratio (PSNR)",
            value=f"{clean_psnr:.2f} dB",
            delta=f"+{psnr_gain:.2f} dB Gain" if psnr_gain >= 0 else f"{psnr_gain:.2f} dB",
            delta_color="normal"
        )
        
        m_col2.metric(
            label="Structural Similarity (SSIM)",
            value=f"{clean_ssim:.4f}",
            delta=f"+{ssim_gain:.4f} Gain" if ssim_gain >= 0 else f"{ssim_gain:.4f}",
            delta_color="normal"
        )
        
        m_col3.metric(
            label="Mean Squared Error (MSE)",
            value=f"{clean_mse:.5f}",
            delta=f"-{mse_reduction:.1f}% Error Reduced",
            delta_color="inverse"
        )
        
        m_col4.metric(
            label="Mean Absolute Error (MAE)",
            value=f"{clean_mae:.5f}",
            delta=f"{clean_mae - noisy_mae:.4f}",
            delta_color="inverse"
        )

    # Download Buttons
    st.markdown("---")
    st.subheader("💾 Export Denoised Output")
    d_col1, d_col2 = st.columns(2)
    
    # Denoised image byte stream
    denoised_pil = Image.fromarray((np.clip(denoised_img, 0.0, 1.0) * 255.0).astype(np.uint8))
    buf = io.BytesIO()
    denoised_pil.save(buf, format="PNG")
    denoised_bytes = buf.getvalue()
    
    with d_col1:
        st.download_button(
            label="📥 Download Denoised Image (PNG)",
            data=denoised_bytes,
            file_name="denoised_mnist_digit.png",
            mime="image/png"
        )
        
    # Canvas byte stream (comparison)
    fig_comp, ax_c = plt.subplots(1, 3, figsize=(9, 3))
    ax_c[0].imshow(clean_img if clean_img is not None else noisy_img, cmap='gray')
    ax_c[0].set_title("Target Clean")
    ax_c[0].axis('off')
    ax_c[1].imshow(noisy_img, cmap='gray')
    ax_c[1].set_title("Noisy Input")
    ax_c[1].axis('off')
    ax_c[2].imshow(denoised_img, cmap='gray')
    ax_c[2].set_title("Denoised Output")
    ax_c[2].axis('off')
    plt.tight_layout()
    c_buf = io.BytesIO()
    plt.savefig(c_buf, format="PNG", dpi=200)
    plt.close()
    
    with d_col2:
        st.download_button(
            label="📥 Download Side-by-Side Comparison Canvas (PNG)",
            data=c_buf.getvalue(),
            file_name="denoising_comparison_canvas.png",
            mime="image/png"
        )

# Experimental Benchmarks and Research Visualizations
st.markdown("---")
st.subheader("📈 Experimental Research Results & Performance Plots")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Loss Curves", "Model Comparison", "Noise Robustness", "Latent Dimensions", "10-Sample Gallery"
])

with tab1:
    if os.path.exists("outputs/loss_curves.png"):
        st.image("outputs/loss_curves.png", caption="Training and Validation Loss Convergence Across All 5 Models", use_container_width=True)
    else:
        st.info("Loss curves will appear once model training finishes.")

with tab2:
    if os.path.exists("outputs/metrics_comparison_bar.png"):
        st.image("outputs/metrics_comparison_bar.png", caption="Benchmark Metrics Comparison: MSE, MAE, PSNR, SSIM", use_container_width=True)
    if os.path.exists("outputs/results_summary.json"):
        with open("outputs/results_summary.json") as f:
            summary = json.load(f)
        st.table(summary.get("comparison_table", []))

with tab3:
    if os.path.exists("outputs/noise_factor_analysis.png"):
        st.image("outputs/noise_factor_analysis.png", caption="Evaluation across Noise Factors: 0.1, 0.2, 0.3, 0.4, 0.5", use_container_width=True)

with tab4:
    if os.path.exists("outputs/latent_dim_analysis.png"):
        st.image("outputs/latent_dim_analysis.png", caption="Latent Dimension Scaling (8, 16, 32, 64, 128) vs Reconstruction Quality", use_container_width=True)

with tab5:
    if os.path.exists("outputs/reconstruction_10_samples_all_models.png"):
        st.image("outputs/reconstruction_10_samples_all_models.png", caption="Comprehensive 10-Digit Reconstructions for All Architectures", use_container_width=True)

st.markdown("""
<div style="text-align: center; color: #9CA3AF; margin-top: 30px; font-size: 0.85rem;">
    Lab Assignment 04: Image Denoising Using Autoencoders • Deep Learning Laboratory
</div>
""", unsafe_allow_html=True)
