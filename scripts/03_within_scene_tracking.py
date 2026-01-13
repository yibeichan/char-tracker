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
import utils

def save2json(data, output_file):
    """Save the selected frames to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        sys.exit(1)

def main(video_name, scratch_dir, output_dir, tracker_kwargs, use_sequential=True, diverse_frames=True, top_n=3):

    scene_file = utils.get_output_path(scratch_dir, utils.OUTPUT_DIR_SCENE_DETECTION, f"{video_name}.txt")
    face_detection_file = utils.get_output_path(scratch_dir, utils.OUTPUT_DIR_FACE_DETECTION, f"{video_name}.json")
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

    # When top_n=1, diversity doesn't apply - always pick the single best frame
    use_diverse_frames = diverse_frames and top_n > 1
    if diverse_frames and not use_diverse_frames:
        print("With top_n=1, diversity is not applicable. Disabling diverse_frames.")

    # Initialize the tracker and selector
    face_tracker = FaceTracker(**tracker_kwargs)
    frame_selector = FrameSelector(video_file=video_file, top_n=top_n, output_dir=output_dir, diverse_frames=use_diverse_frames)

    if use_diverse_frames:
        print(f"Using diversity-aware frame selection with top_n={top_n} (temporal segments for better clustering)")
    else:
        print(f"Using quality-based frame selection with top_n={top_n} (best frames by quality score)")

    # Track faces across scenes
    tracked_faces = face_tracker.track_faces_across_scenes(scene_data, face_data)
    output_file = os.path.join(output_dir, f"{video_name}_tracked_faces.json")
    save2json(tracked_faces, output_file)

    # Select top frames per face
    if use_sequential:
        print("Using optimized sequential frame reading (5-10x faster)")
    else:
        print("Using legacy random-seek method (slower)")
    selected_frames = frame_selector.select_top_frames_per_face(tracked_data=tracked_faces, use_sequential=use_sequential)
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
    parser.add_argument('--no-sequential', action='store_true',
                       help='Disable optimized sequential frame reading (use legacy random-seek method)')
    parser.add_argument('--no-diverse-frames', dest='diverse_frames', action='store_false', default=True,
                       help='Disable diversity-aware frame selection (pick best quality only)')
    parser.add_argument('--top-n', type=int, default=3,
                       help='Number of frames to select per track (default: 3). '
                            'Use 1 for initial clustering to ensure consistent embeddings. '
                            'Use 3+ with diverse-frames for refinement with pose/lighting variation.')

    args = parser.parse_args()
    video_name = args.video_name

    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    output_dir = utils.get_output_path(scratch_dir, utils.OUTPUT_DIR_FACE_TRACKING, f"{video_name}")
    os.makedirs(output_dir, exist_ok=True)

    tracker_kwargs = {
        "iou_threshold": args.iou_threshold,
        "max_gap": args.max_gap,
        "box_expansion": args.box_expansion,
        "use_median_box": args.use_median_box
    }

    main(video_name, scratch_dir, output_dir, tracker_kwargs,
         use_sequential=not args.no_sequential,
         diverse_frames=args.diverse_frames,
         top_n=args.top_n)
