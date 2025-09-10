import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from PIL import Image
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Custom CSS for dark theme and Perceptra styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# App title and configuration
st.set_page_config(
    page_title="Perceptra - Intelligent Surveillance",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
local_css("style.css")  # We'll create this file

# Add paths for modules
if not 'Anomaly_detection' in sys.path:
    sys.path.append('./Anomaly_detection')
if not 'Crowd_density' in sys.path:
    sys.path.append('./Crowd_density')

# Import anomaly detection modules (wrapped in try-except)
try:
    from feature_extractor import extract_video_features
    from predict import load_model as load_anomaly_model
    from predict import predict_anomaly, prepare_features
    from lstm_model_definition import LSTMAnomalyDetector
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False

# Import crowd density estimation modules (wrapped in try-except)
try:
    from crowd_density_estimator import CrowdDensityEstimator
    # from crowd_video_processor import CrowdVideoProcessor
    from crowd_alert import CrowdAlertManager
    CROWD_ESTIMATION_AVAILABLE = True
except ImportError:
    CROWD_ESTIMATION_AVAILABLE = False

# Session state initialization
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = "output"
if 'anomaly_results' not in st.session_state:
    st.session_state.anomaly_results = None
if 'crowd_results' not in st.session_state:
    st.session_state.crowd_results = None
if 'models_checked' not in st.session_state:
    st.session_state.models_checked = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "dashboard"

# Function to check for model files
def check_model_files(anomaly_model_path, csrnet_model_path):
    anomaly_exists = os.path.exists(anomaly_model_path)
    csrnet_exists = os.path.exists(csrnet_model_path)
    
    warnings = []
    if ANOMALY_DETECTION_AVAILABLE and not anomaly_exists:
        warnings.append(f"⚠️ Anomaly detection model not found at {anomaly_model_path}")
        
    if CROWD_ESTIMATION_AVAILABLE and not csrnet_exists:
        warnings.append(f"⚠️ CSRNet model not found at {csrnet_model_path}")
    
    return warnings

# Function to run anomaly detection
def run_anomaly_detection(video_path, anomaly_model_path, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        with st.expander("Anomaly Detection Progress", expanded=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.markdown("<div class='status-box'>Initializing anomaly detection...</div>", unsafe_allow_html=True)
            
            status_text.markdown("<div class='status-box'>Extracting features...</div>", unsafe_allow_html=True)
            progress_bar.progress(10)
            rgb_feat, flow_feat = extract_video_features(video_path)
            
            status_text.markdown("<div class='status-box'>Preparing model input...</div>", unsafe_allow_html=True)
            progress_bar.progress(40)
            features = prepare_features(rgb_feat, flow_feat)
            
            status_text.markdown("<div class='status-box'>Loading anomaly model...</div>", unsafe_allow_html=True)
            progress_bar.progress(60)
            model = load_anomaly_model(lambda: LSTMAnomalyDetector(input_dim=2048), anomaly_model_path)
            model.eval()
            
            status_text.markdown("<div class='status-box'>Running inference...</div>", unsafe_allow_html=True)
            progress_bar.progress(80)
            with torch.no_grad():
                outputs = model(features)
                prob = torch.sigmoid(outputs).item()
            
            optimal_threshold = 0.62
            label = 'Abnormal' if prob >= optimal_threshold else 'Normal'
            
            status_text.markdown(f"<div class='status-box success'>Anomaly detection complete: {label}</div>", unsafe_allow_html=True)
            progress_bar.progress(100)
            
         
            
            report_path = os.path.join(output_dir, 'anomaly_detection_report.txt')
            with open(report_path, 'w') as f:
                f.write(f"ANOMALY DETECTION REPORT\n")
                f.write(f"=======================\n\n")
                f.write(f"Video: {os.path.basename(video_path)}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Result: {label}\n")
                f.write(f"Confidence Score: {prob:.4f}\n")
                f.write(f"Threshold: {optimal_threshold}\n\n")
                
                if label == 'Abnormal':
                    f.write("WARNING: Abnormal behaviour detected.\n")
                    f.write("Recommend security review of the footage.\n")
                else:
                    f.write("Normal behaviour detected.\n")

            results = {
                'label': label,
                'probability': prob,
                'threshold': optimal_threshold,
                'video_path': video_path,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'report_path': report_path
            }        
            
            return results
            
    except Exception as e:
        status_text.markdown(f"<div class='status-box error'>Anomaly detection failed: {str(e)}</div>", unsafe_allow_html=True)
        return None

# Function to display anomaly results
def display_anomaly_results(results):
    if not results:
        return
   
    
    # Display the original input video at the top
    st.markdown("""
    <div class="analysis-header">
        <h2>Abnormal Behavior Detection</h2>
    </div>
    <div class="video-preview-header">
        <h3>Input Video: {}</h3>
    </div>
    """.format(os.path.basename(results['video_path'])), unsafe_allow_html=True)
    
    # Show the original video
    st.video(results['video_path'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # st.markdown(f"""
        # <div class="info-card">
        #     <h3>Video Information</h3>
        #     <p><strong>File:</strong> {os.path.basename(results['video_path'])}</p>
        #     <p><strong>Analysis Time:</strong> {results['timestamp']}</p>
        # </div>
        # """, unsafe_allow_html=True)
        
        if results['label'] == 'Abnormal':
            st.markdown("""
            <div class="result-card abnormal">
                <h3> ABNORMAL BEHAVIOR DETECTED</h3>
                <p>Warning: Suspicious activity detected in the video.</p>
            </div>
            """, unsafe_allow_html=True)
            
            alarm_html = """
            <audio id="alarm" autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
            Your browser does not support the audio element.
            </audio>
            <script>
            var audio = document.getElementById("alarm");
            audio.volume = 0.2;
            
            </script>
            """
            components.html(alarm_html, height=0)
        else:
            st.markdown("""
            <div class="result-card normal">
                <h3>✅ NORMAL BEHAVIOR</h3>
                <p>No suspicious activity detected in the video.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # st.markdown(f"""
        # <div class="metrics-card">
        #     <h3>Confidence Metrics</h3>
        #     <p><strong>Confidence Score:</strong> {results['probability']:.4f}</p>
        #     <p><strong>Threshold:</strong> {results['threshold']}</p>
        #     <progress value="{results['probability']}" max="1"></progress>
        # </div>
        # """, unsafe_allow_html=True)
        
        if results['label'] == 'Abnormal':
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ SECURITY ALERT</h4>
                <p>Abnormal behaviour detected. Recommend immediate security review of the footage.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = results['probability'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Anomaly Score", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#4a4a4a"},
                'bar': {'color': "#4a4a4a"},
                'bgcolor': "#1a1a1a",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, results['threshold']], 'color': "#7CFC00"},
                    {'range': [results['threshold'], 1], 'color': "#FF6347"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': results['threshold']
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "#ffffff", 'family': "Arial"},
            margin=dict(l=30, r=30, t=50, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    report_path = results.get('report_path')  # Make sure 'report_path' is part of results

    if report_path and os.path.exists(report_path):
        with open(report_path, 'r') as file:
            report_text = file.read()
        
        # Show report inside an expandable section
        with st.expander("View Full Analysis Report"):
            st.text(report_text)
        
        # Add download button
        with open(report_path, "rb") as file:
            st.download_button(
                label="⬇Download Report",
                data=file,
                file_name=os.path.basename(report_path),
                mime="text/plain"
            )
    else:
        st.warning("Analysis report not found.")
        

def run_crowd_estimation(video_path, csrnet_model_path, output_dir, area_size=None, count_threshold=None, avg_threshold=None):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        from crowd_video_processor import CrowdVideoProcessor
        crowd_processor = CrowdVideoProcessor(model_path=csrnet_model_path)
        
        
        output_video_path = os.path.join(output_dir, 'crowd_analysis_output.mp4')
        
        with st.expander("Crowd Analysis Progress", expanded=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.markdown("<div class='status-box'>Initializing crowd analysis...</div>", unsafe_allow_html=True)
            
            def update_progress(message):
                if "Progress:" in message:
                    try:
                        progress_str = message.split("Progress:")[1].split("%")[0].strip()
                        progress = float(progress_str) / 100.0
                        progress_bar.progress(progress)
                        status_text.markdown(f"<div class='status-box'>{message}</div>", unsafe_allow_html=True)
                    except:
                        status_text.markdown(f"<div class='status-box'>{message}</div>", unsafe_allow_html=True)
                else:
                    status_text.markdown(f"<div class='status-box'>{message}</div>", unsafe_allow_html=True)
            
            status_text.markdown("<div class='status-box'>Processing video for crowd analysis...</div>", unsafe_allow_html=True)
            progress_bar.progress(0.1)
            
            summary = crowd_processor._process_video(
                video_path, 
                output_path=output_video_path,
                area_size=area_size,
                count_threshold=count_threshold,
                avg_threshold=avg_threshold,
                skip_frames=5,
                max_frames=None,
                show_progress=update_progress
            )
            
            if summary.get('output_video') and os.path.exists(summary['output_video']):
                video_size = os.path.getsize(summary['output_video']) / (1024 * 1024)
                status_text.markdown(f"<div class='status-box success'>Crowd analysis complete. Output video: {video_size:.1f} MB</div>", unsafe_allow_html=True)
            else:
                status_text.markdown(f"<div class='status-box'>Crowd analysis complete but no video output generated.</div>", unsafe_allow_html=True)
            
            progress_bar.progress(1.0)
            
            visualization_path = os.path.join(output_dir, 'crowd_count_over_time.png')
            generate_crowd_visualization(summary, visualization_path)
            
            return summary
            
    except Exception as e:
        status_text.markdown(f"<div class='status-box error'>Crowd density estimation failed: {str(e)}</div>", unsafe_allow_html=True)
        import traceback
        st.code(traceback.format_exc())
        return None

def generate_crowd_visualization(summary, output_path):
    df = pd.DataFrame({
        'Time (seconds)': summary['timestamps'],
        'Crowd Count': summary['crowd_counts'],
        'Density Level': summary['density_levels']
    })
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (seconds)'], df['Crowd Count'])
    plt.title('Crowd Count Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Estimated Crowd Count')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    
    return df


# def display_crowd_alerts(results):
#     """Display crowd alerts in the UI with clear safety instructions"""
#     if not results or 'alerts' not in results:
#         return

#     # Group alerts based on type keywords
#     critical_alerts = [a for a in results['alerts'] if any(k in a[0] for k in ['CRITICAL', 'SURGE'])]
#     warning_alerts = [a for a in results['alerts'] if 'WARNING' in a[0]]
#     trend_alerts = [a for a in results['alerts'] if 'TREND' in a[0]]
#     abnormal_alerts = [a for a in results['alerts'] if 'ABNORMAL' in a[0]]

#     def format_alert(alert, icon, css_class):
#         return f"""
#             <div class='{css_class}' style='padding: 10px; margin-bottom: 10px;
#                 border-radius: 10px; border: 2px solid #999;
#                 background-color: rgba(255,255,255,0.05);'>
#                 <div style='font-weight: bold; font-size: 16px;'>{icon} {alert[1]}</div>
#             </div>
#         """

#     st.markdown("<hr>", unsafe_allow_html=True)

#     # Display critical and surge alerts
#     if critical_alerts:
#         st.error("🚨 EMERGENCY ALERTS - IMMEDIATE ACTION REQUIRED")
#         for alert in critical_alerts:
#             st.markdown(format_alert(alert, "🔴", "critical-alert"), unsafe_allow_html=True)

#     # Display warning alerts (e.g., density or count warnings)
#     if warning_alerts:
#         st.warning("⚠️ CROWD WARNINGS")
#         for alert in warning_alerts:
#             st.markdown(format_alert(alert, "🟡", "warning-alert"), unsafe_allow_html=True)

#     # Display trend alerts (e.g., increasing count over time)
#     if trend_alerts:
#         st.info("📈 INCREASING CROWD TREND")
#         for alert in trend_alerts:
#             st.markdown(format_alert(alert, "📈", "trend-alert"), unsafe_allow_html=True)

#     # Display abnormal pattern alerts (e.g., Z-score deviation)
#     if abnormal_alerts:
#         st.info("🔍 ABNORMAL CROWD PATTERN DETECTED")
#         for alert in abnormal_alerts:
#             st.markdown(format_alert(alert, "🔍", "abnormal-alert"), unsafe_allow_html=True)

#     st.markdown("<hr>", unsafe_allow_html=True)
    
 
def display_crowd_results(results, visualization_path=None):
    if not results:
        return
    
    with st.container():
        st.markdown("""
        <div class="analysis-header">
            <h2>🚨 Safety Alerts & Instructions</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display crowd alerts if present in results
        if 'alert' in results and results['alert']:
            # Determine the risk level based on the alert content
            alert_text = results['alert']
            
            if "🚨 **HIGH RISK:" in alert_text:
                alert_color = "#FF5252"  # red for high risk

                alarm_html = """
                <audio id="alarm" autoplay>
                <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
                Your browser does not support the audio element.
                </audio>
                <script>
                var audio = document.getElementById("alarm");
                audio.volume = 0.2;
                
                </script>
                """
                components.html(alarm_html, height=0)
                
            elif "⚠️ **MODERATE RISK:" in alert_text:
                alert_color = "#FFB74D"  # amber for moderate risk
                
            else:  # Low risk
                alert_color = "#66BB6A"  # green for low risk
                
            
            st.markdown(f"""
            <div class="alert-card" style="border-left: 4px solid {alert_color}; background-color: {alert_color}20; padding: 16px; border-radius: 6px; margin-bottom: 20px;">
                <div class="alert-content" style="font-size: 15px; line-height: 1.6;">
                    {alert_text.replace('**', '<strong>').replace('**', '</strong>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            
            # If there's peak time info, extract and display it
            if "Peak crowd detected" in alert_text:
                peak_info = alert_text.split("Peak crowd detected")[1]
                st.info(f"📍 **Peak crowd detected{peak_info}**")
        else:
            st.markdown("""
            <div class="info-card">
                <p>No crowd density alerts detected for this video.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display the original input video at the top
    st.markdown("""
    <div class="analysis-header">
        <h2>Crowd Density Analysis</h2>
    </div>
    <div class="video-preview-header">
        <h3>Input Video: {}</h3>
    </div>
    """.format(os.path.basename(results['video_path'])), unsafe_allow_html=True)
    
    # Show the original video
    st.video(results['video_path'])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <h3>Crowd Statistics</h3>
            <div class="stat-item">
                <span class="stat-label">Average:</span>
                <span class="stat-value">{results['average_count']:.1f} people</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Maximum:</span>
                <span class="stat-value">{results['max_count']:.1f} people</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Minimum:</span>
                <span class="stat-value">{results['min_count']:.1f} people</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="density-card">
            <h3>Density Level Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        high_count = results['density_levels'].count('HIGH')
        medium_count = results['density_levels'].count('MEDIUM')
        low_count = results['density_levels'].count('LOW')
        total_frames = len(results['density_levels'])
        
        df_density = pd.DataFrame({
            'Level': ['HIGH', 'MEDIUM', 'LOW'],
            'Frames': [high_count, medium_count, low_count],
            'Percentage': [
                high_count/total_frames*100 if total_frames > 0 else 0,
                medium_count/total_frames*100 if total_frames > 0 else 0,
                low_count/total_frames*100 if total_frames > 0 else 0
            ]
        })
        
        fig = px.pie(df_density, values='Frames', names='Level', 
                     title='',
                     color='Level',
                     color_discrete_map={'HIGH': '#FF5252', 'MEDIUM': '#FFB74D', 'LOW': '#66BB6A'})
        
        fig.update_layout(
            showlegend=True,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#ffffff', width=1)))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="time-series-card">
            <h3>⏱Crowd Count Over Time</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if visualization_path and os.path.exists(visualization_path):
            df = pd.DataFrame({
                'Time (seconds)': results['timestamps'],
                'Crowd Count': results['crowd_counts'],
                'Density Level': results['density_levels']
            })
            
            fig = px.line(df, x='Time (seconds)', y='Crowd Count', 
                         title='',
                         color_discrete_sequence=['#4285F4'])
            
            avg_high = 30
            avg_medium = 15
            
            fig.add_shape(type="rect", 
                         xref="paper", yref="y",
                         x0=0, y0=avg_high,
                         x1=1, y1=max(results['crowd_counts']) * 1.1,
                         fillcolor="rgba(255, 82, 82, 0.1)", 
                         layer="below", line_width=0)
            
            fig.add_shape(type="rect", 
                         xref="paper", yref="y",
                         x0=0, y0=avg_medium,
                         x1=1, y1=avg_high,
                         fillcolor="rgba(255, 183, 77, 0.1)", 
                         layer="below", line_width=0)
            
            fig.add_shape(type="rect", 
                         xref="paper", yref="y",
                         x0=0, y0=0,
                         x1=1, y1=avg_medium,
                         fillcolor="rgba(102, 187, 106, 0.1)", 
                         layer="below", line_width=0)
            
            fig.update_layout(
                xaxis_title="Time (seconds)",
                yaxis_title="Crowd Count",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff'),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="video-output-header">
        <h3>Analysis Video Output</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if results.get('output_video') and os.path.exists(results['output_video']):
        video_path = results['output_video']
        video_size = os.path.getsize(video_path) / 1024
        
        st.markdown(f"""
        <div class="video-container">
            <div class="video-info">
                <span>File: {os.path.basename(video_path)}</span>
                <span>Size: {video_size:.1f} KB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
        except Exception as e:
            st.error(f"Error displaying video: {str(e)}")
            
            with open(video_path, "rb") as file:
                st.download_button(
                    label="Download Video",
                    data=file,
                    file_name=os.path.basename(video_path),
                    mime="video/mp4",
                    key="video-download"
                )

            
    else:
        st.markdown("""
        <div class="warning-card">
            <p>No valid output video available. Showing sample frames instead.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'temp_dir' in results and os.path.exists(results['temp_dir']):
            debug_frames_dir = os.path.join(results['temp_dir'], 'debug_frames')
            if os.path.exists(debug_frames_dir):
                frames = [f for f in os.listdir(debug_frames_dir) if f.endswith('.jpg')]
                
                if frames:
                    st.markdown("""
                    <div class="sample-frames-header">
                        <h4>Sample Frames from Analysis</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    cols = st.columns(min(len(frames), 4))
                    for i, frame_file in enumerate(sorted(frames)[:4]):
                        frame_path = os.path.join(debug_frames_dir, frame_file)
                        cols[i].image(frame_path, use_column_width=True, caption=f"Frame {i+1}")
                else:
                    st.markdown("""
                    <div class="warning-card">
                        <p>No sample frames available</p>
                    </div>
                    """, unsafe_allow_html=True)
            
        elif 'frames_dir' in results and os.path.exists(results['frames_dir']):
            frames = [f for f in os.listdir(results['frames_dir']) if f.endswith('.jpg')]
            if frames:
                st.markdown("""
                <div class="sample-frames-header">
                    <h4>Sample Frames from Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(min(len(frames), 4))
                for i, frame_file in enumerate(sorted(frames)[:4]):
                    frame_path = os.path.join(results['frames_dir'], frame_file)
                    cols[i].image(frame_path, use_column_width=True, caption=f"Frame {i+1}")
            else:
                st.markdown("""
                <div class="warning-card">
                    <p>No sample frames available</p>
                </div>
                """, unsafe_allow_html=True)

    report_path = results.get('report_path')  # Make sure 'report_path' is part of results

    if report_path and os.path.exists(report_path):
        with open(report_path, 'r') as file:
            report_text = file.read()
        
        # Show report inside an expandable section
        with st.expander("View Full Analysis Report"):
            st.text(report_text)
        
        # Add download button
        with open(report_path, "rb") as file:
            st.download_button(
                label="⬇Download Report",
                data=file,
                file_name=os.path.basename(report_path),
                mime="text/plain"
            )
    else:
        st.warning("Analysis report not found.")
            

def dashboard_page():
    # Load the logo image
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("logo2.png", width=1000)  # Adjust width as needed
       
    
    # Upload section
    # st.markdown("""
    # <div class="upload-section">
    #     <h5>Intelligent Surveillance System</h5>
    # </div>
    # """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a surveillance video", type=["mp4", "avi", "mov", "mkv"], key="dashboard_uploader")
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.session_state.video_path = video_path
        
        # Display video preview
        st.markdown("""
        <div class="video-preview-header">
            <h3>Video Preview</h3>
        </div>
        """, unsafe_allow_html=True)
        st.video(video_path)
        
        # Analysis options
        st.markdown("""
        <div class="analysis-options">
            <h3>Select Analysis Type</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Abnormal Behavior Detection", use_container_width=True):
                st.session_state.current_page = "anomaly"
                st.rerun()
        
        with col2:
            if st.button("👥 Crowd Density Estimation", use_container_width=True):
                st.session_state.current_page = "crowd"
                st.rerun()
    
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="upload-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#4a4a4a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
            </div>
            <h3>Upload a video to begin analysis</h3>
            <p>Supported formats: MP4, AVI, MOV, MKV</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Dashboard", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
        
        st.markdown("""
        <div class="sidebar-section">
            <h3>Analysis Tools</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Abnormal Behavior", use_container_width=True):
            st.session_state.current_page = "anomaly"
            st.rerun()
        
        if st.button("Crowd Density", use_container_width=True):
            st.session_state.current_page = "crowd"
            st.rerun()
        
        st.markdown("""
        <div class="sidebar-section">
            <h3>Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        output_dir = st.text_input("Output Directory", value=st.session_state.output_dir, key="sidebar_output_dir")
        os.makedirs(output_dir, exist_ok=True)
        st.session_state.output_dir = output_dir

        count_threshold = st.sidebar.slider("Max People Threshold", 10, 200, 50, step=5)
        avg_threshold = st.sidebar.slider("Avg People Threshold", 5, 100, 30, step=5)
        
        area_size = st.number_input("Area Size (m²)", value=100.0, min_value=1.0, 
                                  help="Used for crowd density per square meter calculation")
        
        anomaly_model_path = st.text_input("Anomaly Model Path", value="models/best_model.pth", key="sidebar_anomaly_path")
        csrnet_model_path = st.text_input("CSRNet Model Path", value="models/PartAmodel_best.pth", key="sidebar_csrnet_path")
        
        if not st.session_state.models_checked:
            warnings = check_model_files(anomaly_model_path, csrnet_model_path)
            for warning in warnings:
                st.markdown(f"""
                <div class="warning-message">
                    {warning}
                </div>
                """, unsafe_allow_html=True)
            st.session_state.models_checked = True
    
    # Page routing
    if st.session_state.current_page == "dashboard":
        dashboard_page()
    elif st.session_state.current_page == "anomaly":
        if st.session_state.video_path:
            if st.button("← Back to Dashboard"):
                st.session_state.current_page = "dashboard"
                st.rerun()

            if (
                "anomaly_results" not in st.session_state 
                or st.session_state.anomaly_results is None
                or st.session_state.anomaly_results.get("video_path") != st.session_state.video_path
            ):    
            
                results = run_anomaly_detection(
                    st.session_state.video_path, 
                    anomaly_model_path, 
                    st.session_state.output_dir
                )
                st.session_state.anomaly_results = results
            
            if st.session_state.anomaly_results:
                display_anomaly_results(st.session_state.anomaly_results)
        else:
            st.warning("Please upload a video first from the dashboard")
            if st.button("Go to Dashboard"):
                st.session_state.current_page = "dashboard"
                st.rerun()

    elif st.session_state.current_page == "crowd":
        if st.session_state.video_path:
            if st.button("← Back to Dashboard"):
                st.session_state.current_page = "dashboard"
                st.rerun()

            if (
                "crowd_results" not in st.session_state 
                or st.session_state.crowd_results is None
                or st.session_state.crowd_results.get("video_path") != st.session_state.video_path
            ):
                results = run_crowd_estimation(
                    st.session_state.video_path, 
                    csrnet_model_path, 
                    st.session_state.output_dir,
                    area_size,
                    count_threshold,
                    avg_threshold
                )
                results["video_path"] = st.session_state.video_path  # Save the video path for future checks
                st.session_state.crowd_results = results
    
            
           
            
            if st.session_state.crowd_results:
                visualization_path = os.path.join(st.session_state.output_dir, 'crowd_count_over_time.png')
                display_crowd_results(st.session_state.crowd_results, visualization_path)
        else:
            st.warning("Please upload a video first from the dashboard")
            if st.button("Go to Dashboard"):
                st.session_state.current_page = "dashboard"
                st.rerun()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Perceptra | Intelligent Surveillance System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()