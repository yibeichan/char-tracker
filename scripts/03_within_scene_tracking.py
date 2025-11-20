import os
import sys
import argparse
from dotenv import load_dotenv
import pandas as pd
import json
import cv2

# Add src directory to path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from face_tracker import FaceTracker, FrameSelector

def save2json(data, output_file):
    """Save the selected frames to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        sys.exit(1)

def main(video_name, scratch_dir, output_dir, tracker_kwargs):

    scene_file = os.path.join(scratch_dir, "output", "scene_detection", f"{video_name}.txt")
    face_detection_file = os.path.join(scratch_dir, "output", "face_detection", f"{video_name}.json")
    video_file = os.path.join(scratch_dir, "data", "mkv2mp4", f"{video_name}.mp4")

    scene_data = pd.read_csv(scene_file, sep=",")
    with open(face_detection_file, "r") as f:
        face_data = json.load(f)

    # Auto-detect max_gap based on video FPS if set to -1
    if tracker_kwargs.get('max_gap', -1) == -1:
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Guard against invalid FPS (0, NaN, or negative) - assume 30fps
        if not (fps > 0):
            fps = 30.0
            print(f"Warning: Could not detect valid FPS, assuming fps=30")

        # Use 0.5 seconds as default tolerance for detection gaps
        tracker_kwargs['max_gap'] = int(0.5 * fps)
        print(f"Using FPS: {fps:.2f}, setting max_gap={tracker_kwargs['max_gap']} frames (0.5 seconds)")

    # Initialize the tracker and selector
    face_tracker = FaceTracker(**tracker_kwargs)
    frame_selector = FrameSelector(video_file=video_file, top_n=3, output_dir=output_dir)

    # Track faces across scenes
    tracked_faces = face_tracker.track_faces_across_scenes(scene_data, face_data)
    output_file = os.path.join(output_dir, f"{video_name}_tracked_faces.json")
    save2json(tracked_faces, output_file)

    # Select top frames per face
    selected_frames = frame_selector.select_top_frames_per_face(tracked_data=tracked_faces)
    output_file = os.path.join(output_dir, f"{video_name}_selected_frames_per_face.json")
    save2json(selected_frames, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Within-scene face tracking using IoU matching')
    parser.add_argument('video_name', type=str, help='Name of the input video file without extension.')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='Minimum IoU required to link detections (0-1).')
    parser.add_argument('--max-gap', type=int, default=-1,
                       help='Maximum number of missing frames before track is considered dead. '
                            'Use -1 for auto (0.5 * fps), or specify explicit frame count.')
    parser.add_argument('--box-expansion', type=float, default=0.1,
                       help='Ratio to expand boxes before IoU calculation (tolerates head movement).')
    parser.add_argument('--use-median-box', action='store_true', default=True,
                       help='Use median of recent detections for more stable tracking.')
    parser.add_argument('--no-median-box', dest='use_median_box', action='store_false',
                       help='Use only most recent detection (simpler, faster).')

    args = parser.parse_args()
    video_name = args.video_name

    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    output_dir = os.path.join(scratch_dir, "output", "face_tracking", f"{video_name}")
    os.makedirs(output_dir, exist_ok=True)

    tracker_kwargs = {
        "iou_threshold": args.iou_threshold,
        "max_gap": args.max_gap,
        "box_expansion": args.box_expansion,
        "use_median_box": args.use_median_box
    }

    main(video_name, scratch_dir, output_dir, tracker_kwargs)
