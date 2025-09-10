import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import sys

# Add the pytorch-i3d path
sys.path.append('./pytorch-i3d')
from pytorch_i3d import InceptionI3d

# Configuration Constants
FPS = 30
FRAME_SIZE = (224, 224)
SEGMENT_FRAMES = 16
TOTAL_SEGMENTS = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VideoProcessor:
    def __init__(self, fps=FPS, frame_size=FRAME_SIZE):
        self.fps = fps
        self.frame_size = frame_size

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(native_fps / self.fps))

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
            count += 1

        cap.release()
        return np.array(frames)  # (T, H, W, C)

    def compute_optical_flow(self, frames):
        flows = []
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            flow = tvl1.calc(prev_gray, gray, None)
            flow = cv2.resize(flow, self.frame_size)
            flows.append(flow)
            prev_gray = gray

        return np.array(flows)  # (T-1, H, W, 2)

class I3DFeatureExtractor:
    def __init__(self, modality='rgb'):
        self.modality = modality
        self.model = self.load_i3d_model(modality)
        
    def load_i3d_model(self, modality):
        in_channels = 3 if modality == 'rgb' else 2
        i3d = InceptionI3d(400, in_channels=in_channels)  # Must match pretraining

        weights_path = (
            'pytorch-i3d/models/rgb_imagenet.pt' 
            if modality == 'rgb' 
            else 'pytorch-i3d/models/flow_imagenet.pt'
        )
        state_dict = torch.load(weights_path, map_location=DEVICE)
        i3d.load_state_dict(state_dict)

        i3d.replace_logits(1024)  # Optional: change this layer if needed
        return i3d.to(DEVICE).eval()


    def segment_frames(self, frames, segment_len=SEGMENT_FRAMES, total_segments=TOTAL_SEGMENTS):
        T = len(frames)
        needed_frames = segment_len * total_segments

        if T < needed_frames:
            pad_len = needed_frames - T
            pad = np.repeat(frames[-1][np.newaxis], pad_len, axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        else:
            frames = frames[:needed_frames]

        segments = []
        for i in range(total_segments):
            start = i * segment_len
            end = start + segment_len
            segments.append(frames[start:end])

        return segments  # List of (16, H, W, C)

    def extract_features(self, segments):
        features = []

        for seg in segments:
            # Normalize input to [-1, 1]
            seg = (seg.astype(np.float32) / 127.5) - 1.0

            if self.modality == 'flow':
                seg = seg[..., :2]  # Optical flow has 2 channels

            seg = torch.from_numpy(seg).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
            seg = seg.to(DEVICE).float()

            with torch.no_grad():
                feat = self.model.extract_features(seg)
                feat = torch.nn.functional.adaptive_avg_pool3d(feat, 1).squeeze()

            features.append(feat.cpu().numpy())

        return np.stack(features)  # (32, 1024)

def save_features(features, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, features)
    print(f"[✓] Features Saved: {save_path}, shape={features.shape}")

def extract_video_features(video_path, rgb_save_path=None, flow_save_path=None):
    processor = VideoProcessor()
    frames = processor.extract_frames(video_path)

    # RGB Features
    rgb_extractor = I3DFeatureExtractor(modality='rgb')
    rgb_segments = rgb_extractor.segment_frames(frames)
    rgb_features = rgb_extractor.extract_features(rgb_segments)

    if rgb_save_path:
        save_features(rgb_features, rgb_save_path)

    # Flow Features
    flow = processor.compute_optical_flow(frames)
    flow_extractor = I3DFeatureExtractor(modality='flow')
    flow_segments = flow_extractor.segment_frames(flow)
    flow_features = flow_extractor.extract_features(flow_segments)

    if flow_save_path:
        save_features(flow_features, flow_save_path)

    return rgb_features, flow_features

