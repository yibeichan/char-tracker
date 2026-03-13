"""
Shared utilities and constants for the char-tracker pipeline.
"""

import os

# Output directory names (numbered by pipeline stage)
OUTPUT_DIR_SCENE_DETECTION = "01_scene_detection"
OUTPUT_DIR_FACE_DETECTION = "02_face_detection"
OUTPUT_DIR_FACE_TRACKING = "03_face_tracking"
OUTPUT_DIR_FACE_CLUSTERING = "04a_face_clustering"
OUTPUT_DIR_FACE_TRACKING_BY_CLUSTER = "04b_face_tracking_by_cluster"
OUTPUT_DIR_FACE_TRACKING_REFINED = "04c_face_tracking_by_cluster_refined"
# Stages 05-07 removed (functionality moved to 04c refinement)
# Stage 08 may be renumbered once harmonization step is defined
OUTPUT_DIR_CHARACTER_TIMESTAMPS = "08_character_timestamps"


def get_video_path(video_dir, episode_id):
    """Resolve episode_id to video file path.

    Maps episode_id (e.g. 'friends_s01e03b') to the correct season subdir
    and .mkv file: video_dir/s1/friends_s01e03b.mkv

    Args:
        video_dir: Root directory containing season subdirs (s1/, s2/, ...).
        episode_id: Episode identifier (e.g. 'friends_s01e03b').

    Returns:
        Full path to the .mkv video file.
    """
    season = episode_id.split('_s')[1][:2].lstrip('0')
    return os.path.join(video_dir, f"s{season}", f"{episode_id}.mkv")


def get_output_path(scratch_dir, *parts):
    """Construct an output path under scratch_dir/output/.

    Args:
        scratch_dir: Base scratch directory from environment.
        *parts: Path components to join under output/.

    Returns:
        Full path to the output location.
    """
    return os.path.join(scratch_dir, "output", *parts)
