#!/usr/bin/env python3
"""
Generate per-second character presence timestamps from refined clustering.

This script takes:
1. Refined clustering output from 04c_refine_with_annotations.py
2. Annotation JSON with character labels
3. Raw tracking data with all frame observations

And produces:
1. Per-second character presence JSON
2. Per-second character presence CSV

Author: Pipeline stage 8
Date: 2026-01-08
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
import cv2

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Main characters in Friends
MAIN_CHARACTERS = ['rachel', 'monica', 'chandler', 'joey', 'phoebe', 'ross']

# Labels to skip (not main characters)
SKIP_LABELS = ['dk', 'not_human', 'background', 'unclear', 'junk',
               'not face', 'not clear', 'guest', 'guy on the wheelchair',
               'kid in the hospital', 'random kid']


def read_json(file_path: str) -> dict:
    """Read JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: dict, output_file: str):
    """Save data to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def build_cluster_to_character_mapping(annotations: dict) -> Dict[int, str]:
    """
    Build mapping from cluster_id to character name from annotations.

    Args:
        annotations: Annotation JSON from ClusterMark

    Returns:
        dict: {cluster_id: character_name}
    """
    cluster_to_char = {}

    for cluster_key, cluster_info in annotations['cluster_annotations'].items():
        label = cluster_info['label'].lower()

        # Skip non-main-character labels
        if label in SKIP_LABELS:
            continue

        # Extract numeric cluster ID from string like "friends_s01e03b_cluster-197"
        # or just "cluster-197" or "197"
        if '-' in cluster_key:
            numeric_id = int(cluster_key.split('-')[-1])
        else:
            numeric_id = int(cluster_key)

        # Only include main characters
        if label in MAIN_CHARACTERS:
            cluster_to_char[numeric_id] = label
        else:
            logger.debug(f"Skipping non-main character label: {label}")

    logger.info(f"Mapped {len(cluster_to_char)} clusters to main characters")
    for cluster_id, char in sorted(cluster_to_char.items()):
        logger.debug(f"  Cluster {cluster_id} -> {char}")

    return cluster_to_char


def build_track_to_character_mapping(
    refined_clustering: dict,
    cluster_to_char: Dict[int, str]
) -> Dict[str, str]:
    """
    Build mapping from track (unique_face_id) to character name.

    Args:
        refined_clustering: Output from 04c refinement
        cluster_to_char: Mapping from cluster_id to character

    Returns:
        dict: {unique_face_id: character_name}
    """
    track_to_char = {}
    unmapped_tracks = []

    for scene_id, faces in refined_clustering.items():
        for face_data in faces:
            unique_face_id = face_data['unique_face_id']
            cluster_id = face_data.get('cluster_id')

            if cluster_id in cluster_to_char:
                track_to_char[unique_face_id] = cluster_to_char[cluster_id]
            else:
                unmapped_tracks.append((unique_face_id, cluster_id))

    logger.info(f"Mapped {len(track_to_char)} tracks to main characters")
    if unmapped_tracks:
        logger.debug(f"Found {len(unmapped_tracks)} tracks not mapped to main characters")

    return track_to_char


def generate_frame_level_presence(
    tracked_faces: dict,
    track_to_char: Dict[str, str]
) -> Dict[int, Set[str]]:
    """
    Generate frame-level character presence from raw tracking data.

    Args:
        tracked_faces: Raw tracking data with all frame observations
        track_to_char: Mapping from track to character

    Returns:
        dict: {frame_idx: set of characters present}
    """
    frame_to_chars = defaultdict(set)

    for scene_id, tracks in tracked_faces.items():
        for track_idx, track_data in enumerate(tracks):
            # Build unique_face_id for this track
            unique_face_id = f"{scene_id}_face_{track_idx}"

            # Skip if not mapped to a main character
            if unique_face_id not in track_to_char:
                continue

            character = track_to_char[unique_face_id]

            # Add character to all frames in this track
            for observation in track_data:
                frame_idx = observation['frame']
                frame_to_chars[frame_idx].add(character)

    logger.info(f"Generated character presence for {len(frame_to_chars)} frames")
    return frame_to_chars


def convert_to_per_second(
    frame_to_chars: Dict[int, Set[str]],
    fps: float
) -> Dict[int, List[str]]:
    """
    Convert frame-level presence to per-second presence.

    Args:
        frame_to_chars: Frame-level character presence
        fps: Frames per second

    Returns:
        dict: {second: [list of characters present]}
    """
    if not frame_to_chars:
        return {}

    max_frame = max(frame_to_chars.keys())
    total_seconds = int(max_frame / fps) + 1

    second_to_chars = {}

    for second in range(total_seconds):
        start_frame = int(second * fps)
        end_frame = int((second + 1) * fps)

        chars_in_second = set()
        for frame_idx in range(start_frame, end_frame):
            if frame_idx in frame_to_chars:
                chars_in_second.update(frame_to_chars[frame_idx])

        # Convert to sorted list for consistent output
        second_to_chars[second] = sorted(list(chars_in_second))

    logger.info(f"Generated per-second presence for {total_seconds} seconds")
    return second_to_chars


def smooth_tracks(
    second_to_chars: Dict[int, List[str]],
    fps: float,
    max_gap_sec: float = 1.0
) -> Dict[int, List[str]]:
    """
    Apply track-level smoothing to fill gaps.

    If a character appears at second A and second B with a small gap,
    fill the gap between them (likely same continuous track).

    Args:
        second_to_chars: Per-second character presence
        fps: Frames per second
        max_gap_sec: Maximum gap (in seconds) to fill

    Returns:
        dict: Smoothed per-second character presence
    """
    if not second_to_chars:
        return {}

    smoothed = second_to_chars.copy()
    max_gap_frames = int(max_gap_sec * fps)

    for char in MAIN_CHARACTERS:
        # Find all seconds where this character appears
        char_seconds = sorted([
            s for s, chars in smoothed.items()
            if char in chars
        ])

        if len(char_seconds) < 2:
            continue

        # Find gaps and fill them
        for i in range(len(char_seconds) - 1):
            start_sec = char_seconds[i]
            end_sec = char_seconds[i + 1]

            # Calculate gap in seconds
            gap_seconds = end_sec - start_sec

            # Only fill if gap is within threshold (and > 0)
            if 0 < gap_seconds <= max_gap_sec + 1:
                for fill_sec in range(start_sec + 1, end_sec):
                    if fill_sec not in smoothed:
                        smoothed[fill_sec] = []
                    if char not in smoothed[fill_sec]:
                        smoothed[fill_sec].append(char)
                        smoothed[fill_sec].sort()

    logger.info(f"Applied track-level smoothing (max_gap={max_gap_sec}s)")
    return smoothed


def save_timestamps(
    second_to_chars: Dict[int, List[str]],
    output_dir: str,
    episode_id: str
) -> Tuple[str, str]:
    """
    Save timestamps in both JSON and CSV formats.

    Args:
        second_to_chars: Per-second character presence
        output_dir: Output directory
        episode_id: Episode identifier

    Returns:
        tuple: (json_path, csv_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_data = {
        'metadata': {
            'episode_id': episode_id,
            'main_characters': MAIN_CHARACTERS,
            'total_seconds': len(second_to_chars)
        },
        'timestamps': {str(k): v for k, v in second_to_chars.items()}
    }

    json_path = os.path.join(output_dir, f"{episode_id}_timestamps.json")
    save_json(json_data, json_path)
    logger.info(f"Saved JSON timestamps to: {json_path}")

    # Save CSV
    csv_data = []
    for second in sorted(second_to_chars.keys()):
        row = {'second': second}
        for char in MAIN_CHARACTERS:
            row[char] = 1 if char in second_to_chars[second] else 0
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, f"{episode_id}_timestamps.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV timestamps to: {csv_path}")

    return json_path, csv_path


def get_video_fps(video_path: str) -> float:
    """Get FPS from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video {video_path}, assuming 30 fps")
        return 30.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not fps or fps <= 0:
        logger.warning(f"Invalid FPS {fps}, assuming 30 fps")
        return 30.0

    return fps


def main(episode_id: str, annotation_file: str, scratch_dir: str,
         max_gap_sec: float = 1.0):
    """
    Generate per-second character presence timestamps.

    Args:
        episode_id: Episode identifier (e.g., 'friends_s01e03b')
        annotation_file: Path to annotation JSON
        scratch_dir: SCRATCH_DIR from environment
        max_gap_sec: Maximum gap (seconds) for track smoothing
    """
    logger.info(f"=" * 70)
    logger.info(f"Generating Character Timestamps for {episode_id}")
    logger.info(f"=" * 70)

    # Load annotations
    logger.info(f"Loading annotations from: {annotation_file}")
    annotations = read_json(annotation_file)

    # Validate episode_id matches
    annotation_episode_id = annotations.get('metadata', {}).get('episode_id')
    if annotation_episode_id and annotation_episode_id != episode_id:
        logger.warning(
            f"Episode ID mismatch: annotation={annotation_episode_id}, argument={episode_id}"
        )

    # Build cluster-to-character mapping
    cluster_to_char = build_cluster_to_character_mapping(annotations)

    # Load refined clustering
    refined_clustering_file = os.path.join(
        scratch_dir, "output", "face_clustering",
        f"{episode_id}_matched_faces_with_clusters_refined.json"
    )
    logger.info(f"Loading refined clustering from: {refined_clustering_file}")
    refined_clustering = read_json(refined_clustering_file)

    # Build track-to-character mapping
    track_to_char = build_track_to_character_mapping(refined_clustering, cluster_to_char)

    # Load raw tracking data
    tracking_file = os.path.join(
        scratch_dir, "output", "face_tracking",
        f"{episode_id}", f"{episode_id}_tracked_faces.json"
    )
    logger.info(f"Loading tracking data from: {tracking_file}")
    tracked_faces = read_json(tracking_file)

    # Get video FPS
    video_file = os.path.join(scratch_dir, "data", "mkv2mp4", f"{episode_id}.mp4")
    fps = get_video_fps(video_file)
    logger.info(f"Video FPS: {fps:.2f}")

    # Generate frame-level character presence
    logger.info("Generating frame-level character presence...")
    frame_to_chars = generate_frame_level_presence(tracked_faces, track_to_char)

    # Convert to per-second
    logger.info("Converting to per-second presence...")
    second_to_chars = convert_to_per_second(frame_to_chars, fps)

    # Apply smoothing
    logger.info(f"Applying track-level smoothing (max_gap={max_gap_sec}s)...")
    second_to_chars = smooth_tracks(second_to_chars, fps, max_gap_sec)

    # Save results
    output_dir = os.path.join(scratch_dir, "output", "character_timestamps")
    json_path, csv_path = save_timestamps(second_to_chars, output_dir, episode_id)

    # Summary statistics
    logger.info(f"\n" + "=" * 70)
    logger.info("TIMESTAMP GENERATION SUMMARY")
    logger.info("=" * 70)

    # Character appearance statistics
    char_seconds = {char: 0 for char in MAIN_CHARACTERS}
    for chars in second_to_chars.values():
        for char in chars:
            if char in char_seconds:
                char_seconds[char] += 1

    total_seconds = len(second_to_chars)
    logger.info(f"Total seconds: {total_seconds}")
    logger.info(f"Character appearance time (seconds):")
    for char in MAIN_CHARACTERS:
        percentage = (char_seconds[char] / total_seconds * 100) if total_seconds > 0 else 0
        logger.info(f"  {char.capitalize():10s}: {char_seconds[char]:4d} ({percentage:5.1f}%)")

    logger.info(f"\n✓ Timestamps saved:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  CSV:  {csv_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate per-second character presence timestamps from refined clustering'
    )
    parser.add_argument('episode_id', type=str,
                       help='Episode ID (e.g., friends_s01e03b)')
    parser.add_argument('annotation_file', type=str,
                       help='Path to annotation JSON from ClusterMark')
    parser.add_argument('--max-gap', type=float, default=1.0,
                       help='Maximum gap (seconds) for track smoothing (default: 1.0)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')

    args = parser.parse_args()

    # Configure logging level
    logger.setLevel(getattr(logging, args.log_level))

    from dotenv import load_dotenv
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")

    if not scratch_dir:
        logger.error("SCRATCH_DIR not found in environment")
        sys.exit(1)

    try:
        main(args.episode_id, args.annotation_file, scratch_dir, args.max_gap)
    except Exception as e:
        logger.error(f"Error during timestamp generation: {e}", exc_info=True)
        sys.exit(1)
