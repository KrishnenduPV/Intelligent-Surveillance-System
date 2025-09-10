import cv2
import numpy as np
import os
from tqdm import tqdm
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from crowd_alert import generate_crowd_alert

class CrowdVideoProcessor:
    def __init__(self, model_path):
        # Import the estimator here to avoid circular imports
        from crowd_density_estimator import CrowdDensityEstimator
        self.estimator = CrowdDensityEstimator(model_path)
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.temp_dir}")
    
    def _process_video(self, video_path, output_path=None, area_size=None,count_threshold=None, avg_threshold=None ,
                      skip_frames=5, max_frames=None, show_progress=None):
        """Process video and return analysis results"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize results storage
        crowd_counts = []
        density_levels = []
        timestamps = []
        processed_frames = []
        
        frame_count = 0
        processed_count = 0
        
        with tqdm(total=total_frames, desc="Processing Video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames and processed_count >= max_frames):
                    break
                
                frame_count += 1
                if frame_count % skip_frames != 0:
                    pbar.update(1)
                    continue
                
                try:
                    # Process frame
                    density_map, count = self.estimator.process_frame(frame)
                    density_level = self.estimator.get_density_level(count, area_size)
                    
                    # Create visualization
                    overlay = self._create_visualization(frame, density_map,count)
                    
                    # Store results
                    crowd_counts.append(count)
                    density_levels.append(density_level)
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                    processed_frames.append(overlay)
                    processed_count += 1
                    
                    # Update progress if callback provided
                    if show_progress and frame_count % 10 == 0:
                        progress = (frame_count / total_frames) * 100
                        show_progress(f"Progress: {progress:.1f}% - Processed {processed_count} frames")
                    
                except Exception as e:
                    print(f"\nError processing frame {frame_count}: {str(e)}")
                
                pbar.update(1)
        
        cap.release()
        
        # Save some frames for debug purposes
        debug_dir = os.path.join(self.temp_dir, "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)
        for i, frame in enumerate(processed_frames[:5]):  # Save first 5 frames
            cv2.imwrite(f"{debug_dir}/frame_{i}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Save output video if requested
        video_saved = False
        if output_path and processed_frames:
            video_saved = self._save_output_video(processed_frames, output_path, fps/skip_frames)
            if not video_saved:
                print("Failed to save video, frames saved as images instead")
        
        results = {
            'video_path': video_path,
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'average_count': np.mean(crowd_counts) if crowd_counts else 0,
            'max_count': np.max(crowd_counts) if crowd_counts else 0,
            'min_count': np.min(crowd_counts) if crowd_counts else 0,
            'crowd_counts': crowd_counts,
            'density_levels': density_levels,
            'timestamps': timestamps,
            'output_video': output_path if video_saved else None,
            'temp_dir': self.temp_dir,
            'debug_frames_dir': debug_dir,
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'video_saved': video_saved
        }
        
        # Generate report
        report_path = os.path.join(os.path.dirname(output_path), 'crowd_analysis_report.txt')
        self.generate_report(results, os.path.dirname(output_path))
        results['report_path'] = report_path

        # After results dictionary is populated, before returning:
        alert_message = generate_crowd_alert(results,count_threshold,avg_threshold)

        if alert_message:
            print("\n" + alert_message)
            results['alert'] = alert_message
        
        return results
    
    def _create_visualization(self, frame, density_map, model_count):
        """Create visualization overlay using the model's original count"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize ONLY for visualization (not for counting)
        density_map = cv2.resize(density_map, (frame_rgb.shape[1], frame_rgb.shape[0]))
        
        # Normalize for coloring (preserve original count)
        if density_map.max() > density_map.min():
            norm_density = (density_map - density_map.min()) / (density_map.max() - density_map.min())
        else:
            norm_density = np.zeros_like(density_map)
        
        # Apply colormap
        density_colored = cv2.applyColorMap((norm_density * 255).astype(np.uint8), cv2.COLORMAP_JET)
        density_colored_rgb = cv2.cvtColor(density_colored, cv2.COLOR_BGR2RGB)
        
        # Use model_count instead of recalculating
        overlay = cv2.addWeighted(frame_rgb, 0.7, density_colored_rgb, 0.3, 0)
        
        # Display the model's original count
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Count: {model_count:.1f}"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        cv2.rectangle(overlay, (10, 10), (10 + text_size[0] + 10, 10 + text_size[1] + 10), (0, 0, 0), -1)
        cv2.putText(overlay, text, (15, 15 + text_size[1]), font, 1, (255, 255, 255), 2)
        
        return overlay

    
    def _save_output_video(self, frames, output_path, fps):
        """Save processed frames to video file with robust error handling"""
        if not frames:
            print("No frames to save")
            return False
        
        print(f"Saving video to {output_path} with {len(frames)} frames at {fps} FPS")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        height, width = frames[0].shape[:2]
        print(f"Frame dimensions: {width}x{height}")
        
        # Make sure the output has a proper extension
        base, ext = os.path.splitext(output_path)
        if ext.lower() not in ['.mp4', '.avi']:
            output_path = base + '.mp4'  # Force MP4 format
        
        # Try different codecs in order of preference
        success = False
        for codec in ['avc1', 'mp4v', 'XVID', 'H264', 'X264']:
            try:
                print(f"Trying codec: {codec}")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                
                # Ensure we're creating a fresh video file
                if os.path.exists(output_path):
                    os.remove(output_path)
                
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if out.isOpened():
                    for frame in frames:
                        # Convert frame from RGB to BGR for OpenCV
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(bgr_frame)
                    
                    # Make sure to release the writer
                    out.release()
                    
                    # Verify the video was written successfully
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                        print(f"Successfully saved video with codec {codec} to {output_path}")
                        print(f"File size: {os.path.getsize(output_path)} bytes")
                        
                        # Copy the video to the temp directory for debugging
                        debug_video = os.path.join(self.temp_dir, "output_video.mp4")
                        shutil.copy(output_path, debug_video)
                        print(f"Debug copy saved to {debug_video}")
                        
                        success = True
                        break
                    else:
                        print(f"Video file is too small or doesn't exist")
                else:
                    print(f"Failed to open video writer with codec {codec}")
            except Exception as e:
                print(f"Failed with codec {codec}: {str(e)}")
                if os.path.exists(output_path):
                    os.remove(output_path)  # Clean up failed attempt
                continue
        
        # If video creation failed with all codecs, save as image sequence
        if not success:
            print("All codecs failed, saving as image sequence")
            frames_dir = os.path.join(self.temp_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                img_path = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
                cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            print(f"Saved {len(frames)} frames to {frames_dir}")
            
            # Create a simple HTML player for Streamlit
            with open(os.path.join(self.temp_dir, 'player.html'), 'w') as f:
                f.write(f"""
                <html><body>
                <h2>Frame Sequence Player</h2>
                <img src="{os.path.join(frames_dir, 'frame_00000.jpg')}" id="player">
                </body></html>
                """)
        
        return success
    
    def generate_report(self, results, output_dir):
        """Generate a text report of the analysis"""
        report_path = os.path.join(output_dir, 'crowd_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"CROWD DENSITY ANALYSIS REPORT\n")
            f.write(f"=============================\n\n")
            f.write(f"Video: {os.path.basename(results['video_path'])}\n")
            f.write(f"Analysis Time: {results['analysis_time']}\n\n")
            f.write(f"Total Frames: {results['total_frames']}\n")
            f.write(f"Processed Frames: {results['processed_frames']}\n\n")
            f.write(f"Average Crowd Count: {results['average_count']:.1f}\n")
            f.write(f"Maximum Count: {results['max_count']:.1f}\n")
            f.write(f"Minimum Count: {results['min_count']:.1f}\n\n")
            
            # Density level statistics
            high_count = results['density_levels'].count('HIGH')
            medium_count = results['density_levels'].count('MEDIUM')
            low_count = results['density_levels'].count('LOW')
            total = len(results['density_levels'])
            
            if total > 0:
                f.write("Density Level Distribution:\n")
                f.write(f"- HIGH density: {high_count} frames ({high_count/total*100:.1f}%)\n")
                f.write(f"- MEDIUM density: {medium_count} frames ({medium_count/total*100:.1f}%)\n")
                f.write(f"- LOW density: {low_count} frames ({low_count/total*100:.1f}%)\n")
            
            # Video output status
            f.write(f"\nVideo Output Status: {'Success' if results.get('video_saved', False) else 'Failed'}\n")
            if results.get('output_video'):
                f.write(f"Output video: {results['output_video']}\n")
            else:
                f.write("No output video generated\n")
        
        return report_path