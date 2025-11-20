import os
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceTracker:
    def __init__(self, iou_threshold=0.5, max_gap=30, box_expansion=0.1, use_median_box=True):
        """
        Simple, effective face tracker for within-scene tracking.

        Args:
            iou_threshold (float): Minimum IoU required to associate a detection with an existing track.
            max_gap (int): Maximum number of missing frames before a track is considered dead.
                          Default 30 frames (~1 second at 30fps). Should be set based on video FPS
                          to tolerate detection failures from head turns, occlusions, and detector misses.
            box_expansion (float): Ratio to expand bounding boxes before IoU calculation (tolerates small movements).
            use_median_box (bool): Use median of recent detections for matching (more stable than last detection).
        """
        self.iou_threshold = iou_threshold
        self.max_gap = max(0, int(max_gap))
        self.box_expansion = box_expansion
        self.use_median_box = use_median_box

    def expand_box(self, box, expansion_ratio=None):
        """Expand the bounding box by a given ratio to tolerate small head movements."""
        if expansion_ratio is None:
            expansion_ratio = self.box_expansion

        width = box[2] - box[0]
        height = box[3] - box[1]

        x_expand = width * expansion_ratio
        y_expand = height * expansion_ratio

        expanded_box = [
            box[0] - x_expand,  # Left
            box[1] - y_expand,  # Top
            box[2] + x_expand,  # Right
            box[3] + y_expand   # Bottom
        ]
        return expanded_box

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes using pure numpy."""
        # Calculate intersection coordinates
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        # Calculate intersection area
        inter_width = max(0.0, x2_inter - x1_inter)
        inter_height = max(0.0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou

    def _get_reference_box(self, track):
        """
        Get the reference box for matching against new detections.

        Uses median of recent detections (more stable) or most recent detection (simpler).
        The median approach is robust to jittery detections while staying responsive.
        """
        if self.use_median_box and len(track["observations"]) >= 3:
            # Use median of up to the last 5 detections (minimum 3) for stability
            recent_boxes = [obs["face"] for obs in track["observations"][-5:]]
            recent_boxes = np.array(recent_boxes)
            median_box = np.median(recent_boxes, axis=0)
            return median_box
        else:
            # Use most recent detection
            return np.array(track["observations"][-1]["face"])

    def track_faces(self, face_data, min_faces_per_cluster):
        """
        Track faces within a scene using IoU matching with best-match assignment.

        Simple and effective: finds the track with highest IoU above threshold.
        Uses expanded boxes to tolerate natural head movement and detection jitter.
        """
        tracks = []

        for frame_number, face, conf in face_data:
            detection_box = np.array(face, dtype=np.float32)
            best_track = None
            best_iou = 0.0

            # Find the best matching track
            for track in tracks:
                frame_gap = frame_number - track["last_frame"]

                # Skip tracks that haven't been updated recently (likely different person)
                if frame_gap > self.max_gap + 1:
                    continue

                # Get reference box (median or most recent)
                reference_box = self._get_reference_box(track)

                # Calculate IoU with expanded boxes to tolerate small movements
                iou = self.calculate_iou(
                    self.expand_box(reference_box.tolist()),
                    self.expand_box(detection_box.tolist())
                )

                # Keep track of best match
                if iou > self.iou_threshold and iou > best_iou:
                    best_track = track
                    best_iou = iou

            # Add detection to best matching track or create new track
            if best_track is not None:
                best_track["observations"].append({
                    "frame": frame_number,
                    "face": detection_box.tolist(),
                    "conf": conf
                })
                best_track["last_frame"] = frame_number
            else:
                # Create new track
                tracks.append({
                    "observations": [{
                        "frame": frame_number,
                        "face": detection_box.tolist(),
                        "conf": conf
                    }],
                    "last_frame": frame_number
                })

        # Filter out short tracks (likely false detections)
        return [track["observations"] for track in tracks
                if len(track["observations"]) > min_faces_per_cluster]

    def track_faces_across_scenes(self, scene_data, face_data):
        """Track faces across all scenes in a video."""
        all_tracked_faces = {}

        for index, row in tqdm(scene_data.iterrows(), total=scene_data.shape[0], desc="Tracking Faces Across Scenes"):
            frame_start, frame_end = int(row["Start Frame"]), int(row["End Frame"])
            scene_id = f"scene_{index + 1}"

            n_frames = frame_end - frame_start + 1
            min_faces_per_cluster = min(max(n_frames // 2, 15), 30)  # 30 is FPS

            face_data_for_scene = []
            
            for i in range(frame_start, frame_end):
                faces = face_data[i]["detections"]
                if len(faces) != 0:
                    for f in faces:
                        face_data_for_scene.append((i, f["box"], f["confidence"]))

            # Skip scenes with no faces detected
            if not face_data_for_scene:
                continue

            tracked_faces = self.track_faces(face_data_for_scene, min_faces_per_cluster)
            all_tracked_faces[scene_id] = tracked_faces

        return all_tracked_faces

class FrameSelector:
    def __init__(self, video_file, top_n=3, output_dir=None, save_images=True):
        self.video_file = video_file
        self.top_n = top_n
        self.output_dir = output_dir
        self.save_images = save_images

        if save_images and output_dir:
            os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def calculate_brightness(image):
        """Calculate the brightness of an image."""
        return np.mean(image)

    @staticmethod
    def calculate_sharpness(image):
        """Calculate the sharpness using Laplacian variance.

        Higher variance = more edges detected = sharper image.
        """
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        return np.var(laplacian)

    def save_cropped_face(self, face_image, unique_face_id, frame_idx):
        """Save the cropped face image to disk and return the relative path."""
        if self.output_dir and self.save_images:
            save_filename = f"{unique_face_id}_frame_{frame_idx}.jpg"
            save_path = os.path.join(self.output_dir, save_filename)
            cv2.imwrite(save_path, face_image)
            return save_filename 

    def select_top_frames_per_face(self, tracked_data):
        """Select top frames per face based on confidence, size, brightness, and blurriness."""
        cap = cv2.VideoCapture(self.video_file)
        selected_frames = {}
        global_face_id = 0

        total_faces = sum(len(faces) for faces in tracked_data.values())

        with tqdm(total=total_faces, desc="Select Top Frames Per Face") as pbar:
            for scene_id, faces in tracked_data.items():
                selected_frames[scene_id] = []

                for face_id, face_group in enumerate(faces):
                    frame_scores = []

                    for entry in face_group:
                        frame_idx = entry['frame']
                        face_coords = entry['face']
                        confidence = entry['conf']

                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            print(f"Warning: Could not read frame {frame_idx}. Skipping.")
                            continue

                        height, width, _ = frame.shape
                        x1, y1, x2, y2 = map(int, face_coords)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)

                        width_cropped = max(0, x2 - x1)
                        height_cropped = max(0, y2 - y1)
                        if width_cropped == 0 or height_cropped == 0:
                            print(f"Warning: Invalid bounding box {face_coords} for frame {frame_idx}. Skipping.")
                            continue

                        face_image = frame[y1:y2, x1:x2]
                        if face_image.size == 0:
                            print(f"Warning: Face image is empty for frame {frame_idx}. Skipping.")
                            continue

                        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        face_size = width_cropped * height_cropped
                        brightness = self.calculate_brightness(gray_face)
                        sharpness = self.calculate_sharpness(gray_face)

                        # Normalize the components
                        normalized_face_size = face_size / (width * height)
                        normalized_brightness = brightness / 255.0
                        # Normalize sharpness: higher Laplacian variance = sharper = better
                        # Use scaling factor to bring into [0, 1] range (typical values: 10-500)
                        normalized_sharpness = sharpness / (sharpness + 100.0)

                        # Combine features into a score (higher is better)
                        score = confidence + 0.5 * normalized_face_size + 0.3 * normalized_brightness + 0.2 * normalized_sharpness

                        # Save the image and get its relative path
                        relative_path = self.save_cropped_face(face_image, f"{scene_id}_face_{face_id}", frame_idx)

                        frame_scores.append({
                            "frame_idx": frame_idx,
                            "total_score": score,
                            "face_coord": face_coords,
                            "image_path": relative_path  # Include the relative path in the JSON
                        })

                    if frame_scores:
                        top_frames = sorted(frame_scores, key=lambda x: x["total_score"], reverse=True)[:self.top_n]

                        unique_face_id = f"{scene_id}_face_{face_id}"
                        global_unique_face_id = f"global_face_{global_face_id}"

                        selected_frames[scene_id].append({
                            "unique_face_id": unique_face_id,
                            "global_face_id": global_unique_face_id,
                            "top_frames": [{"frame_idx": frame['frame_idx'], "total_score": frame['total_score'], "face_coord": frame['face_coord'], "image_path": frame['image_path']} for frame in top_frames]
                        })

                    global_face_id += 1
                    pbar.update(1)

        cap.release()
        return selected_frames
