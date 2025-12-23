Perceptra – Intelligent Surveillance System


# Perceptra – Intelligent Surveillance System
### AI-Powered Abnormal Behaviour Detection & Crowd Density Monitoring

Perceptra is an AI-powered surveillance system designed to detect and predict abnormal crowd behaviours. It uses deep learning models to analyze video feeds and identify unusual activities such as sudden crowd density changes, aggressive movements, or unauthorized gatherings. This system helps improve public safety in crowded environments such as airports, malls, stadiums, transportation hubs, and large public gatherings.

The system integrates **two independent modules**:
- **Abnormal Behaviour Detection Module** – BiLSTM model trained on I3D features from the UCF-Crime dataset
- **Crowd Density Estimation Module** – Pretrained CSRNet model to generate heatmaps and count crowds

Both modules are implemented separately and can be selected or run individually through a unified **Streamlit dashboard interface** that provides real-time video analysis, heatmap visualizations, alerts, and performance metrics.

---

## Key Features
- **Dual-Function** Surveillance System Integration
- Use of BiLSTM with Attention on I3D Features for **Anomaly Detection**
- **Crowd density estimation** with CSRNet and real-time heatmaps
- Streamlit-Based Real-Time Interface
- **Alerts triggered** automatically based on thresholds
- **Audio and visual warning system**
- **Line charts** showing crowd size changes over time
- Modular Design Enabling Future Expansion


---

## Problem Statement
Traditional surveillance relies heavily on human operators watching multiple screens - leading to fatigue, missed detection, and slow response. These systems are **reactive**, identifying threats only after events occur.

AI-powered surveillance provides:
- Autonomous anomaly detection
- Real-time proactive alerting
- Reduced human burden
- Enhanced public safety

---

## Project Aim & Objectives
To develop a real-time intelligent surveillance system that enhances public safety by automatically detecting abnormal behaviour and monitoring crowd density using deep learning.

### Objectives
- Build a deep learning system that can analyse video footage to detect crowd issues like density changes, aggressive actions, or suspicious behaviour.
- Detect crowd density and identify signs of overcrowding or unusual changes over time.
- Detect  abnormal human behaviours helping to spot potential problems before they escalate.
- Set up alerts that notify security teams, allowing for quick responses.
- Create a user-friendly interface so security teams can easily manage alerts and video analysis.

---

## Dataset
| Module | Dataset | Description |
|--------|-----------|-------------------------------|
| Abnormal Behaviour Detection | UCF-Crime | Real-world surveillance video dataset containing 13 crime categories |

---

## Tech Stack
| Category | Tools / Libraries |
|----------|------------------|
| Language | Python |
| Deep Learning | PyTorch, TorchVision |
| Frontend / Dashboard | Streamlit |
| Computer Vision | OpenCV |
| Data Visualization | Matplotlib, Seaborn |
| Model Components | I3D, BiLSTM + Attention, CSRNet |

---

## Model Architecture
### **Abnormal Behaviour Detection**
1. Frame extraction & preprocessing  
2. Feature extraction with I3D pretrained network  
3. BiLSTM sequence learning  
4. Classification (normal / abnormal) and threshold-based alert activation

### **Crowd Density Estimation**
1. CSRNet pretrained model
2. Generate density maps
3. Estimate crowd count
4. Threshold-based alert activation

---
# System Architecture
#### Dashboard Interface → Upload Video → Select Model → Preprocessing according to chosen model → Deep Learning Inference (BiLSTM / CSRNet) → Visual Output + Graphs + Alerts on Dashboard 
---

## Installation
```bash
git clone https://github.com/KrishnenduPV/Intelligent-Surveillance-System.git
cd Intelligent-Surveillance-System
pip install -r requirements.txt
```
---
## ▶ Usage

### Run Streamlit Interface
```bash
streamlit run streamlit_app/app.py
```
---
## Results
### Abnormal Behaviour Detection (BiLSTM)
| Metric | Value | 
|--------|-----------|
| Accuracy | 89.29% |
| Precision | 88.73% |
| Recall | 90.00% |
| F1-Score |  89.36% |

---
### Crowd Density Estimation
- Accurate density map generation and crowd count estimation
- Threshold-based alerts
---

### Limitations
- Binary abnormal behaviour classification only (normal / abnormal)
- No bounding box localization
- CSRNet not retrained for custom environments
- Limited real-time streaming support
---

### Future Work
- Multi-class anomaly identification (fight, fall, fire, theft, etc.)
- Behaviour localization on frames using bounding boxes
- Real-Time Streaming and Active Surveillance
- Model Fusion and Unified Decision Logic
- Domain Adaptation and Model Fine-Tuning

---




