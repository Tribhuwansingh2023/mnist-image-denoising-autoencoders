"""
FastAPI Backend Server for Image Denoising Autoencoder
Exposes RESTful endpoints for model inference, health check, and image denoising.
"""

import io
import os
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

# Initialize FastAPI application
app = FastAPI(
    title="Image Denoising Autoencoder API",
    description="REST API for deep learning autoencoder-based image denoising on MNIST digits.",
    version="1.0.0"
)

# Enable CORS for local and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load best model
MODEL_PATH = "models/best_denoising_autoencoder.keras"
model = None

@app.on_event("startup")
def load_model_on_startup():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"[API] Loaded trained model from {MODEL_PATH}")
        except Exception as e:
            print(f"[API] Error loading model: {e}")
    else:
        print(f"[API] Warning: Model file {MODEL_PATH} not found.")

@app.get("/")
def read_root():
    return {
        "service": "Image Denoising Autoencoder API",
        "status": "online",
        "model_loaded": model is not None,
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "denoise_upload": "/denoise-image",
            "denoise_json": "/denoise-json"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "input_shape": str(model.input_shape) if model else None,
        "output_shape": str(model.output_shape) if model else None
    }

@app.get("/models")
def get_models():
    return {
        "active_model": "Denoising Autoencoder (DAE)",
        "input_resolution": "28x28 grayscale",
        "parameters": model.count_params() if model else 484944,
        "best_benchmark_metrics": {
            "MSE": 0.01856,
            "MAE": 0.05258,
            "PSNR_dB": 17.31,
            "SSIM": 0.7584
        }
    }

def preprocess_pil_image(pil_img: Image.Image) -> np.ndarray:
    """Preprocesses a PIL image to normalized 28x28 float32 array."""
    # Convert to grayscale
    img_gray = pil_img.convert("L")
    # Resize to 28x28
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img_resized, dtype="float32") / 255.0
    return arr

@app.post("/denoise-image")
async def denoise_image_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (PNG, JPG, BMP), processes it through
    the autoencoder, and returns the denoised 28x28 PNG image directly.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
        
    norm_arr = preprocess_pil_image(pil_img)
    
    # Check model input shape
    if len(model.input_shape) == 2:
        model_input = norm_arr.reshape(1, 784)
    else:
        model_input = norm_arr.reshape(1, 28, 28, 1)
        
    reconstructed = model.predict(model_input, verbose=0)
    recon_2d = reconstructed.reshape(28, 28)
    
    # Convert output back to uint8 PIL Image
    out_uint8 = np.clip(recon_2d * 255.0, 0, 255).astype(np.uint8)
    out_pil = Image.fromarray(out_uint8, mode="L")
    
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/denoise-json")
async def denoise_json_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image, processes it, and returns base64-encoded strings and metrics.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
        
    norm_arr = preprocess_pil_image(pil_img)
    
    if len(model.input_shape) == 2:
        model_input = norm_arr.reshape(1, 784)
    else:
        model_input = norm_arr.reshape(1, 28, 28, 1)
        
    reconstructed = model.predict(model_input, verbose=0)
    recon_2d = reconstructed.reshape(28, 28)
    
    # Calculate noise metrics relative to input
    mse = float(np.mean((norm_arr - recon_2d) ** 2))
    psnr = float(100.0 if mse == 0 else 20.0 * np.log10(1.0 / np.sqrt(mse)))
    
    # Encode output to base64 PNG
    out_uint8 = np.clip(recon_2d * 255.0, 0, 255).astype(np.uint8)
    out_pil = Image.fromarray(out_uint8, mode="L")
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    return {
        "status": "success",
        "input_dimensions": [28, 28],
        "metrics_vs_input": {
            "mse": round(mse, 5),
            "psnr_db": round(psnr, 2)
        },
        "denoised_image_base64_png": f"data:image/png;base64,{b64_str}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
