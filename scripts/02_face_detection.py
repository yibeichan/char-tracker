import os
import sys
import argparse
from dotenv import load_dotenv

# Add src directory to path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from face_detector import FaceDetector

def main(video_name, input_dir, output_dir, save_annotated_video=False):
    video_path = os.path.join(input_dir, f"{video_name}.mp4")

    if not os.path.exists(video_path):  # Check if the video file exists
        print(f"Error: Video file {video_path} does not exist.")
        sys.exit(1)

    # Initialize FaceDetector and detect faces
    face_detector = FaceDetector(video_path, output_dir, save_annotated_video=save_annotated_video)
    face_detections = face_detector.detect_faces_in_video()

    # Save the face detection results
    output_file = os.path.join(output_dir, f"{video_name}.json")
    face_detector.save_results(output_file, face_detections)

    print(f"Face detection completed. Results saved to {output_file}")
    if save_annotated_video:
        print(f"Annotated video saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Detection in Video')
    parser.add_argument('video_name', type=str, help='Name of the input video file without extension.')
    parser.add_argument('--save-annotated-video', action='store_true',
                       help='Save annotated video with bounding boxes (slower, uses more disk space)')

    args = parser.parse_args()
    video_name = args.video_name

    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    if scratch_dir is None:  # Check if SCRATCH_DIR is set in the environment
        print("Error: SCRATCH_DIR environment variable is not set.")
        sys.exit(1)

    input_dir = os.path.join(scratch_dir, "data", "mkv2mp4")
    output_dir = os.path.join(scratch_dir, "output", "02_face_detection")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    main(video_name, input_dir, output_dir, save_annotated_video=args.save_annotated_video)