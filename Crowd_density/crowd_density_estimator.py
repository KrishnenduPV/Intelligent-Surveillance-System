import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import warnings

warnings.filterwarnings('ignore')

class CSRNet(nn.Module):
    """CSRNet architecture for crowd density estimation"""
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU()
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ])
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

class CrowdDensityEstimator:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        """Safe model loading with error handling"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
    
        model = CSRNet().to(self.device)
    
        try:
        # First try with weights_only=True (secure mode)
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            except:
                # Fallback to weights_only=False if needed (with warning)
                warnings.warn("Using weights_only=False - only do this with trusted model files")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
                state_dict = checkpoint.get('state_dict', checkpoint)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                model.eval()
                return model
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def process_frame(self, frame):
        """Process a single frame and return density map and count"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                density_map = self.model(img_tensor).squeeze().cpu().numpy()
            
            crowd_count = np.sum(density_map)
            return density_map, crowd_count
            
        except Exception as e:
            raise RuntimeError(f"Frame processing failed: {str(e)}")

    def get_density_level(self, count, area_size=None):
        """Classify density level based on count and optional area size"""
        if area_size is not None:
            density = count / area_size
            if density > 0.5:  # persons per m²
                return "HIGH"
            elif density > 0.2:
                return "MEDIUM"
            return "LOW"
        else:
            if count > 30:
                return "HIGH"
            elif count > 15:
                return "MEDIUM"
            return "LOW"