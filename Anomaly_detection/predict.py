import torch
import numpy as np

# === CONFIG ===
MODEL_PATH = 'models/best_model.pth'  # Path to trained model


DEVICE = torch.device('cpu')  # Explicitly use CPU


# === STEP 1: Load the Model ===
def load_model(model_class, model_path):
    model = model_class()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])  # Only load the model's weights
    model = model.to(DEVICE)
    model.eval()
    return model


# === STEP 2: Prepare Input Features ===
def prepare_features(rgb_features, flow_features):
    """
    Args:
        rgb_features (np.ndarray): shape (32, 1024)
        flow_features (np.ndarray): shape (32, 1024)
    Returns:
        torch.Tensor: shape (1, 32, 2048)
    """
    features = np.concatenate((rgb_features, flow_features), axis=1)  # (32, 2048)
    features = torch.from_numpy(features).unsqueeze(0).float().to(DEVICE)  # (1, 32, 2048)
    return features

# === STEP 3: Predict Anomaly ===
def predict_anomaly(model, features, threshold=0.5):
    with torch.no_grad():
        outputs = model(features)  # (batch_size, 1)
        prob = torch.sigmoid(outputs).item()  # scalar probability

    label = 'Abnormal' if prob >= threshold else 'Normal'
    return label, prob


