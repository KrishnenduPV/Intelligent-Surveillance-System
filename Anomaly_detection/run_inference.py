# inference/run.py
import sys
sys.path.append('./utils')
from feature_extractor import extract_video_features
from predict import load_model, predict_anomaly, prepare_features
from lstm_model_definition import LSTMAnomalyDetector  # or YourBiLSTMModel
import torch
import os

MODEL_PATH = "models/best_model.pth"

def run_inference(video_path, show_progress=print):  # ✅ Added progress callback
    show_progress(f"📁 Processing video: {os.path.basename(video_path)}")

    # Step 1: Extract features
    show_progress("🖼️ Step 1: Extracting I3D RGB + Flow features...")
    rgb_feat, flow_feat = extract_video_features(video_path)
    show_progress(f"✅ Features extracted. RGB: {rgb_feat.shape}, Flow: {flow_feat.shape}")

    # Step 2: Prepare input
    show_progress("📦 Step 2: Preparing model input...")
    features = prepare_features(rgb_feat, flow_feat)
    show_progress(f"✅ Input shape: {features.shape} | Mean: {features.mean().item():.4f}, Std: {features.std().item():.4f}")

    # Step 3: Load model
    show_progress("📥 Step 3: Loading model...")
    model = load_model(lambda: LSTMAnomalyDetector(input_dim=2048), MODEL_PATH)
    model.eval()
    show_progress("✅ Model loaded and set to eval mode.")

    # Step 4: Predict
    show_progress("🧠 Step 4: Running inference...")
    with torch.no_grad():
        outputs = model(features)
        prob = torch.sigmoid(outputs).item()
    show_progress(f"📊 Output score (sigmoid): {prob:.4f}")

    # Step 5: Decision
    optimal_threshold = 0.62  # Adjust as needed
    label = 'Abnormal' if prob >= optimal_threshold else 'Normal'
    show_progress(f"🏁 Inference complete. Prediction: {label} | Confidence: {prob:.4f}")

    return label, prob
