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
    def __init__(self, video_file, top_n=3, output_dir=None, save_images=True,
                 crop_expansion=0.2, min_crop_size=50, diverse_frames=True):
        """
        Initialize FrameSelector.

        Args:
            video_file: Path to video file
            top_n: Number of top frames to select per face
            output_dir: Directory to save cropped face images
            save_images: Whether to save cropped face images
            crop_expansion: Ratio to expand bounding boxes when cropping (0.2 = 20% padding).
                           This makes faces easier to identify visually while keeping
                           original tight boxes in metadata for embedding extraction.
            min_crop_size: Minimum face size in pixels to save (filters tiny/blurry crops)
            diverse_frames: If True, select frames with temporal diversity (early/mid/late track)
                           for better clustering robustness.
        """
        self.video_file = video_file
        self.top_n = top_n
        self.output_dir = output_dir
        self.save_images = save_images
        self.crop_expansion = crop_expansion
        self.min_crop_size = min_crop_size
        self.diverse_frames = diverse_frames

        if save_images and output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def expand_box_for_crop(self, box, frame_width, frame_height):
        """Expand bounding box for cropping with boundary checks."""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Expand by crop_expansion ratio
        x_expand = width * self.crop_expansion
        y_expand = height * self.crop_expansion

        # Apply expansion with boundary clamping
        new_x1 = max(0, int(x1 - x_expand))
        new_y1 = max(0, int(y1 - y_expand))
        new_x2 = min(frame_width, int(x2 + x_expand))
        new_y2 = min(frame_height, int(y2 + y_expand))

        return new_x1, new_y1, new_x2, new_y2

    def _score_face(self, frame, face_coords, confidence):
        """
        Validate and score a single face based on quality metrics.

        Returns:
            tuple: (score, face_image, face_coords) or (None, None, None) if invalid
        """
        frame_height, frame_width = frame.shape[:2]

        # Get original tight box dimensions for quality metrics
        orig_x1, orig_y1, orig_x2, orig_y2 = map(int, face_coords)
        orig_x1, orig_y1 = max(0, orig_x1), max(0, orig_y1)
        orig_x2, orig_y2 = min(frame_width, orig_x2), min(frame_height, orig_y2)

        width_cropped = max(0, orig_x2 - orig_x1)
        height_cropped = max(0, orig_y2 - orig_y1)

        # Skip faces that are too small (also handles zero-sized crops)
        if width_cropped < self.min_crop_size or height_cropped < self.min_crop_size:
            return None, None, None

        # Calculate quality metrics on ORIGINAL tight box
        tight_face = frame[orig_y1:orig_y2, orig_x1:orig_x2]
        if tight_face.size == 0:
            return None, None, None
        gray_face = cv2.cvtColor(tight_face, cv2.COLOR_BGR2GRAY)

        # Use expanded box for saving cropped image
        x1, y1, x2, y2 = self.expand_box_for_crop(face_coords, frame_width, frame_height)
        face_image = frame[y1:y2, x1:x2]
        if face_image.size == 0:
            return None, None, None

        face_size = width_cropped * height_cropped
        brightness = self.calculate_brightness(gray_face)
        sharpness = self.calculate_sharpness(gray_face)

        # Normalize components
        normalized_face_size = face_size / (frame_width * frame_height)
        normalized_brightness = brightness / 255.0
        normalized_sharpness = sharpness / (sharpness + 100.0)

        # Combine features into a score
        score = (confidence + 0.5 * normalized_face_size +
                 0.3 * normalized_brightness + 0.2 * normalized_sharpness)

        return score, face_image, face_coords

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

    def _collect_frame_requirements(self, tracked_data):
        """
        Collect all frames that need to be read with their face metadata.

        Returns:
            frame_requirements: dict mapping frame_idx -> list of (scene_id, face_id, face_coords, confidence)
            face_metadata: dict mapping (scene_id, face_id) -> metadata for final output
        """
        frame_requirements = {}
        face_metadata = {}
        global_face_id = 0

        for scene_id, faces in tracked_data.items():
            for face_id, face_group in enumerate(faces):
                unique_face_id = f"{scene_id}_face_{face_id}"
                global_unique_face_id = f"global_face_{global_face_id}"

                # Store metadata for this face
                face_metadata[(scene_id, face_id)] = {
                    'unique_face_id': unique_face_id,
                    'global_face_id': global_unique_face_id
                }

                # Add all frames for this face to requirements
                for entry in face_group:
                    frame_idx = entry['frame']
                    if frame_idx not in frame_requirements:
                        frame_requirements[frame_idx] = []

                    frame_requirements[frame_idx].append({
                        'scene_id': scene_id,
                        'face_id': face_id,
                        'face_coords': entry['face'],
                        'confidence': entry['conf']
                    })

                global_face_id += 1

        return frame_requirements, face_metadata

    def _sequential_read_and_score(self, frame_requirements):
        """
        Read video sequentially and score all faces for required frames.

        Returns:
            face_scores: dict mapping (scene_id, face_id) -> list of frame score dicts
        """
        cap = cv2.VideoCapture(self.video_file)
        face_scores = {}

        # Get video dimensions for normalization
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sort frames for sequential access
        sorted_frames = sorted(frame_requirements.keys())
        total_required = len(sorted_frames)

        current_frame_idx = 0
        required_idx = 0

        with tqdm(total=total_required, desc="Reading and Scoring Frames (Sequential)", unit="frame") as pbar:
            while required_idx < total_required:
                target_frame = sorted_frames[required_idx]

                # Read frames until we reach the target
                while current_frame_idx <= target_frame:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Warning: Could not read frame {current_frame_idx}. Stopping sequential read.")
                        break
                    current_frame_idx += 1

                # If read failed, stop processing entirely
                if not ret:
                    break

                if current_frame_idx - 1 != target_frame:
                    print(f"Warning: Skipping frame {target_frame}")
                    required_idx += 1
                    continue

                # Process all faces for this frame
                for face_req in frame_requirements[target_frame]:
                    scene_id = face_req['scene_id']
                    face_id = face_req['face_id']
                    face_coords = face_req['face_coords']
                    confidence = face_req['confidence']

                    # Score and validate face
                    score, face_image, _ = self._score_face(frame, face_coords, confidence)
                    if score is None:
                        continue

                    # Save cropped face
                    unique_face_id = f"{scene_id}_face_{face_id}"
                    relative_path = self.save_cropped_face(face_image, unique_face_id, target_frame)

                    # Store score
                    key = (scene_id, face_id)
                    if key not in face_scores:
                        face_scores[key] = []

                    face_scores[key].append({
                        "frame_idx": target_frame,
                        "total_score": score,
                        "face_coord": face_coords,
                        "image_path": relative_path
                    })

                required_idx += 1
                pbar.update(1)

        cap.release()
        return face_scores

    def _select_best_frames(self, face_scores, face_metadata):
        """
        Select top N frames per face from all scored frames.

        Returns:
            selected_frames: Final output structure with top frames per face
        """
        selected_frames = {}

        for (scene_id, face_id), scores in face_scores.items():
            if scene_id not in selected_frames:
                selected_frames[scene_id] = []

            # Sort by score and select top N
            top_frames = sorted(scores, key=lambda x: x["total_score"], reverse=True)[:self.top_n]

            metadata = face_metadata[(scene_id, face_id)]
            selected_frames[scene_id].append({
                "unique_face_id": metadata['unique_face_id'],
                "global_face_id": metadata['global_face_id'],
                "top_frames": [{
                    "frame_idx": frame['frame_idx'],
                    "total_score": frame['total_score'],
                    "face_coord": frame['face_coord'],
                    "image_path": frame['image_path']
                } for frame in top_frames]
            })

        return selected_frames

    def _select_best_frames_diverse(self, face_scores, face_metadata):
        """
        Select top N frames per face with temporal diversity.

        Instead of picking the absolute best N frames, this method:
        1. Divides the track into N temporal segments (early/mid/late)
        2. Picks the BEST frame (by quality score) from each segment

        This ensures diversity in pose, lighting, and appearance across the selected frames,
        which leads to more robust embeddings for clustering.

        Args:
            face_scores: dict mapping (scene_id, face_id) -> list of frame score dicts
            face_metadata: dict mapping (scene_id, face_id) -> metadata

        Returns:
            selected_frames: Final output structure with diverse top frames per face
        """
        selected_frames = {}

        for (scene_id, face_id), scores in face_scores.items():
            if scene_id not in selected_frames:
                selected_frames[scene_id] = []

            # Sort by frame number to get temporal order
            scores_sorted_by_frame = sorted(scores, key=lambda x: x["frame_idx"])
            n_frames = len(scores_sorted_by_frame)

            if n_frames <= self.top_n:
                # Track is short, just use all frames (sorted by quality)
                top_frames = sorted(scores_sorted_by_frame, key=lambda x: x["total_score"], reverse=True)
            else:
                # Divide track into segments and pick best from each
                segment_size = n_frames / self.top_n
                top_frames = []

                for i in range(self.top_n):
                    # Get frames in this segment
                    start_idx = int(i * segment_size)
                    end_idx = int((i + 1) * segment_size) if i < self.top_n - 1 else n_frames
                    segment = scores_sorted_by_frame[start_idx:end_idx]

                    # Skip empty segments defensively to avoid ValueError in max()
                    if not segment:
                        continue

                    # Pick best frame from this segment (by quality score)
                    best_in_segment = max(segment, key=lambda x: x["total_score"])
                    top_frames.append(best_in_segment)

            metadata = face_metadata[(scene_id, face_id)]
            selected_frames[scene_id].append({
                "unique_face_id": metadata['unique_face_id'],
                "global_face_id": metadata['global_face_id'],
                "top_frames": [{
                    "frame_idx": frame['frame_idx'],
                    "total_score": frame['total_score'],
                    "face_coord": frame['face_coord'],
                    "image_path": frame['image_path']
                } for frame in top_frames]
            })

        return selected_frames

    def select_top_frames_per_face(self, tracked_data, use_sequential=True):
        """
        Select top frames per face based on confidence, size, brightness, and sharpness.

        Args:
            tracked_data: Dictionary mapping scene_id -> list of face tracks
            use_sequential: If True, use optimized sequential reading (5-10x faster).
                          If False, use old random-seek method (for compatibility).

        Returns:
            selected_frames: Dictionary with top frames per face
        """
        if use_sequential:
            # Optimized path: Read video once sequentially
            frame_requirements, face_metadata = self._collect_frame_requirements(tracked_data)
            face_scores = self._sequential_read_and_score(frame_requirements)

            # Choose selection method based on diverse_frames setting
            if self.diverse_frames:
                return self._select_best_frames_diverse(face_scores, face_metadata)
            else:
                return self._select_best_frames(face_scores, face_metadata)
        else:
            # Legacy path: Random seeks (kept for backward compatibility)
            return self._select_top_frames_per_face_legacy(tracked_data)

    def _select_top_frames_per_face_legacy(self, tracked_data):
        """Legacy implementation with random seeks (slower but kept for compatibility)."""
        cap = cv2.VideoCapture(self.video_file)
        selected_frames = {}
        global_face_id = 0

        total_faces = sum(len(faces) for faces in tracked_data.values())

        with tqdm(total=total_faces, desc="Select Top Frames Per Face (Legacy)") as pbar:
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

                        # Score and validate face
                        score, face_image, _ = self._score_face(frame, face_coords, confidence)
                        if score is None:
                            continue

                        # Save the image and get its relative path
                        relative_path = self.save_cropped_face(face_image, f"{scene_id}_face_{face_id}", frame_idx)

                        frame_scores.append({
                            "frame_idx": frame_idx,
                            "total_score": score,
                            "face_coord": face_coords,
                            "image_path": relative_path
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
