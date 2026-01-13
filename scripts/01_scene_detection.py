import os
import sys
import argparse
from dotenv import load_dotenv

# Add src directory to path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from scene_detector import SceneDetector

def main(video_name, detector_type, input_dir, output_dir):
    video_path = os.path.join(input_dir, f"{video_name}.mp4")  # Full path to the video file
    output_file = os.path.join(output_dir, f"{video_name}.txt")

    scene_detector = SceneDetector(video_path, detector_type=detector_type)
    scene_detector.initialize_scene_manager()
    scene_detector.detect_scenes()
    scene_detector.save_shots(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scene Detection in Video')
    parser.add_argument('video_name', type=str, help='Name of the input video file.')
    parser.add_argument('--detector', type=str, choices=['content', 'adaptive', 'hash'], default='adaptive',
                        help='Scene detection method to use. Options: content, adaptive, hash.')
    
    args = parser.parse_args()

    video_name = args.video_name
    detector_type = args.detector
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    input_dir = os.path.join(scratch_dir, "data", "mkv2mp4")
    output_dir = os.path.join(scratch_dir, "output", "01_scene_detection")
    os.makedirs(output_dir, exist_ok=True)

    main(video_name, detector_type, input_dir, output_dir)
