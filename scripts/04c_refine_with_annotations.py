#!/usr/bin/env python3
"""
Refine face clustering using annotations from ClusterMark.

This script uses the ClusterRefiner module to:
1. Propagate labels to unannotated faces via track matching
2. Match entirely unannotated clusters to known characters
3. Apply constraints to merge/split clusters
4. Optionally reorganize images into cluster folders

Usage:
    python 04c_refine_with_annotations.py <episode_id> <annotation_file> [options]
    python 04c_refine_with_annotations.py <episode_id> <annotation_file> --dry-run
    python 04c_refine_with_annotations.py <episode_id> <annotation_file> --reorganize

Author: Pipeline stage 04c
Date: 2026-01-09
"""

import os
import sys
import json
import argparse
import logging
import shutil
from collections import defaultdict
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cluster_refiner import ClusterRefiner
import utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def read_json(file_path):
    """Read JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, output_file):
    """Save JSON file with numpy conversion."""
    def convert_np(obj):
        if hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, (int, float)) and hasattr(obj, 'dtype'):  # numpy scalar
            return obj.item()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, default=convert_np)


def reorganize_by_refined_cluster(refined_clustering, episode_id, source_dir, output_base_dir, mode='copy'):
    """
    Reorganize face images by refined cluster ID.

    Cluster IDs are strings in the format:
    - "cluster-{character}" for named characters (e.g., "cluster-rachel")
    - "cluster-{number:03d}" for others (e.g., "cluster-001", "cluster-002")

    Args:
        refined_clustering: Refined clustering data
        episode_id: Episode ID (e.g., 'friends_s01e05')
        source_dir: Directory containing original images
        output_base_dir: Base directory for output
        mode: 'copy', 'move', or 'symlink'

    Returns:
        Path to reorganized directory
    """
    cluster_images = defaultdict(list)

    for scene_id, faces in refined_clustering.items():
        for face_data in faces:
            cluster_id = face_data.get('cluster_id')
            unique_face_id = face_data.get('unique_face_id')
            image_paths = face_data.get('image_paths', [])

            if cluster_id is None:
                logger.warning(f"{unique_face_id} has no cluster assignment. Skipping.")
                continue

            parts = unique_face_id.split('_')
            if len(parts) >= 4 and parts[0] == 'scene' and parts[2] == 'face':
                scene_num = parts[1]
                track_num = parts[3]
            else:
                logger.warning(f"Unexpected unique_face_id format: {unique_face_id}")
                scene_num = scene_id
                track_num = "unknown"

            for img_path in image_paths:
                filename = os.path.basename(img_path)
                frame_parts = os.path.splitext(filename)[0].split('_')
                if 'frame' in frame_parts:
                    frame_idx = frame_parts[frame_parts.index('frame') + 1]
                else:
                    frame_idx = "unknown"

                cluster_images[cluster_id].append({
                    'source_path': os.path.join(source_dir, img_path),
                    'scene': scene_num,
                    'track': track_num,
                    'frame': frame_idx
                })

    episode_output_dir = os.path.join(output_base_dir, episode_id)
    os.makedirs(episode_output_dir, exist_ok=True)

    total_files = sum(len(images) for images in cluster_images.values())
    processed = 0

    logger.info(f"Found {len(cluster_images)} refined clusters with {total_files} total images")

    for cluster_id, images in sorted(cluster_images.items()):
        # cluster_id is already in the format "cluster-{name}" or "cluster-{number}"
        # Just prepend episode_id
        cluster_name = f"{episode_id}_{cluster_id}"
        cluster_dir = os.path.join(episode_output_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)

        for img_info in images:
            source_path = img_info['source_path']

            if not os.path.exists(source_path):
                logger.warning(f"Source file not found: {source_path}")
                continue

            new_filename = f"scene_{img_info['scene']}_track_{img_info['track']}_frame_{img_info['frame']}.jpg"
            dest_path = os.path.join(cluster_dir, new_filename)

            try:
                if mode == 'copy':
                    shutil.copy2(source_path, dest_path)
                elif mode == 'move':
                    shutil.move(source_path, dest_path)
                elif mode == 'symlink':
                    rel_source = os.path.relpath(source_path, cluster_dir)
                    if os.path.lexists(dest_path):
                        os.remove(dest_path)
                    os.symlink(rel_source, dest_path)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                processed += 1
            except Exception as e:
                logger.error(f"Error processing {source_path}: {e}")

        logger.info(f"Created cluster directory: {cluster_dir} ({len(images)} images)")

    logger.info(f"Reorganization complete! Processed {processed}/{total_files} files")
    return episode_output_dir


def validate_track_integrity(refined_clustering):
    """Validate that all faces from the same track are in the same cluster."""
    track_to_cluster = {}

    for scene_id, faces in refined_clustering.items():
        for face_data in faces:
            unique_face_id = face_data['unique_face_id']
            cluster_id = face_data['cluster_id']

            if unique_face_id in track_to_cluster:
                if track_to_cluster[unique_face_id] != cluster_id:
                    return {
                        'valid': False,
                        'violations': [{
                            'track': unique_face_id,
                            'clusters': [track_to_cluster[unique_face_id], cluster_id]
                        }]
                    }
            else:
                track_to_cluster[unique_face_id] = cluster_id

    return {'valid': True, 'violations': [], 'total_tracks': len(track_to_cluster)}


def consolidate_non_faces(refined_clustering, skip_labels):
    """
    Consolidate all non-face clusters into a single 'cluster-non-face'.

    Non-face labels include: not_human, background, unclear, junk, not face, etc.

    Args:
        refined_clustering: Refined clustering data
        skip_labels: List of labels to treat as non-face

    Returns:
        Updated refined clustering with consolidated non-face cluster
    """
    non_face_cluster_id = 'cluster-non-face'
    non_face_faces = []
    non_face_clusters = set()

    # Find all faces in non-face clusters
    for scene_id, faces in refined_clustering.items():
        for face_data in faces:
            cluster_id = face_data.get('cluster_id')
            if cluster_id and cluster_id.startswith('cluster-'):
                # Extract label from cluster_id
                label = cluster_id.replace('cluster-', '')
                # Check if this is a non-face label
                if label in skip_labels:
                    non_face_clusters.add(cluster_id)
                    non_face_faces.append((scene_id, face_data))

    # Reassign all non-face faces to single cluster
    for scene_id, face_data in non_face_faces:
        face_data['cluster_id'] = non_face_cluster_id

    if non_face_clusters:
        logger.info(f"Consolidated {len(non_face_clusters)} non-face clusters "
                   f"into {non_face_cluster_id} ({len(non_face_faces)} faces)")

    return refined_clustering


def main(episode_id, annotation_file, scratch_dir, dry_run=False, reorganize=False, reorganize_mode='copy',
         config=None):
    """
    Refine clustering using annotations from ClusterMark.

    Args:
        episode_id: Episode identifier (e.g., 'friends_s01e05')
        annotation_file: Path to annotation JSON from ClusterMark
        scratch_dir: SCRATCH_DIR from environment
        dry_run: If True, show what would happen without modifying data
        reorganize: If True, reorganize images into cluster folders
        reorganize_mode: 'copy', 'move', or 'symlink'
        config: Optional configuration overrides
    """
    logger.info("=" * 70)
    logger.info(f"Refining clustering for {episode_id}")
    logger.info("=" * 70)

    # Load annotations
    logger.info(f"Loading annotations from: {annotation_file}")
    annotations = read_json(annotation_file)

    # Validate episode_id matches
    annotation_episode_id = annotations.get('metadata', {}).get('episode_id')
    if annotation_episode_id and annotation_episode_id != episode_id:
        logger.warning(
            f"Episode ID mismatch: annotation={annotation_episode_id}, argument={episode_id}"
        )
        raise ValueError(
            f"Episode ID mismatch: annotation={annotation_episode_id}, argument={episode_id}"
        )

    # Load original clustering data
    clustering_file = utils.get_output_path(
        scratch_dir, utils.OUTPUT_DIR_FACE_CLUSTERING,
        f"{episode_id}_matched_faces_with_clusters.json"
    )
    logger.info(f"Loading clustering data from: {clustering_file}")
    clustering_data = read_json(clustering_file)

    # Create refiner and run refinement
    refiner = ClusterRefiner(annotations, clustering_data, config)

    result = refiner.refine(dry_run=dry_run)

    refined_clustering = result['refined_clustering']
    statistics = result['statistics']

    if dry_run:
        logger.info("\n" + "=" * 70)
        logger.info("DRY RUN SUMMARY")
        logger.info("=" * 70)
        logger.info("No files were modified. Above shows what would happen.")
        logger.info("=" * 70)
        return

    # Validate track integrity
    logger.info("Validating track integrity...")
    track_validation = validate_track_integrity(refined_clustering)
    if not track_validation['valid']:
        logger.error("Track integrity validation failed!")
        logger.error(f"Violations: {track_validation['violations']}")
        raise ValueError("Track integrity violation detected")
    logger.info(f"Track integrity validated for {track_validation['total_tracks']} tracks")

    # Consolidate non-face clusters into single cluster
    logger.info("Consolidating non-face clusters...")
    refined_clustering = consolidate_non_faces(refined_clustering, refiner.SKIP_LABELS)

    # Extract cluster information from refined clustering
    # Cluster IDs are now strings like "cluster-rachel", "cluster-001", etc.
    cluster_info = {}
    for scene_id, faces in refined_clustering.items():
        for face_data in faces:
            cluster_id = face_data.get('cluster_id')
            if cluster_id and cluster_id not in cluster_info:
                # Extract character name from cluster_id if it's a named cluster
                if cluster_id.startswith('cluster-'):
                    name_part = cluster_id.replace('cluster-', '')
                    # Check if it's a numeric cluster (e.g., "001") or a name
                    if name_part and name_part[0].isdigit():
                        cluster_info[cluster_id] = {'label': f'other_{name_part}', 'cluster_id': cluster_id}
                    else:
                        cluster_info[cluster_id] = {'label': name_part, 'cluster_id': cluster_id}

    # Save refined clustering
    output_file = utils.get_output_path(
        scratch_dir, utils.OUTPUT_DIR_FACE_CLUSTERING,
        f"{episode_id}_matched_faces_with_clusters_refined.json"
    )

    output_data = {
        'metadata': {
            'episode_id': episode_id,
            'refined': True,
            'main_characters': list(refiner.MAIN_CHARACTERS)
        },
        'cluster_info': cluster_info,
    }

    # Add clustering data at top level for compatibility
    output_data.update(refined_clustering)

    save_json(output_data, output_file)
    logger.info(f"Refined clustering saved to: {output_file}")

    # Reorganize images by cluster if requested
    if reorganize:
        logger.info("=" * 70)
        logger.info("REORGANIZING IMAGES BY REFINED CLUSTER")
        logger.info("=" * 70)

        source_dir = utils.get_output_path(scratch_dir, utils.OUTPUT_DIR_FACE_TRACKING, episode_id)
        output_base_dir = utils.get_output_path(scratch_dir, utils.OUTPUT_DIR_FACE_TRACKING_REFINED)

        reorganized_dir = reorganize_by_refined_cluster(
            refined_clustering, episode_id, source_dir, output_base_dir,
            reorganize_mode
        )
        logger.info(f"Reorganized images saved to: {reorganized_dir}")

    # Print summary
    logger.info("=" * 70)
    logger.info("REFINEMENT SUMMARY")
    logger.info("=" * 70)

    # Handle both old and new workflow statistics
    if 'ground_truth_tracks' in statistics:
        # New workflow statistics
        logger.info(f"Ground truth tracks:          {statistics['ground_truth_tracks']}")
        logger.info(f"Ground truth faces:           {statistics['ground_truth_faces']}")
        logger.info(f"DK groups linked:             {statistics['dk_groups_linked']}")
        logger.info(f"Constraint violations:        {statistics['constraint_violations']}")
    else:
        # Old workflow statistics
        logger.info(f"Merges performed:              {statistics.get('merges_performed', 0)}")
        logger.info(f"Splits performed:              {statistics.get('splits_performed', 0)}")
        logger.info(f"\nPropagation Statistics:")
        logger.info(f"Tracks mapped:                 {statistics.get('tracks_mapped', 0)}")
        logger.info(f"Conflicts resolved:            {statistics.get('conflicts_resolved', 0)}")
        logger.info(f"Faces propagated via tracks:   {statistics.get('faces_propagated', 0)}")
        logger.info(f"Unannotated clusters matched:  {statistics.get('clusters_matched', 0)}")
        logger.info(f"DK faces converted:            {statistics.get('dk_converted', 0)}")

    logger.info(f"\nInitial clusters:              {statistics.get('initial_clusters', 0)}")
    logger.info(f"Final clusters:                {statistics.get('final_clusters', 0)}")
    logger.info(f"Cluster change:                {statistics.get('cluster_change', 0):+d}")
    logger.info("=" * 70)

    logger.info("\nNext step:")
    logger.info(f"  python ./scripts/08_generate_character_timestamps.py {episode_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Refine face clustering using annotations from ClusterMark'
    )
    parser.add_argument('episode_id', type=str,
                       help='Episode ID (e.g., friends_s01e05)')
    parser.add_argument('annotation_file', type=str,
                       help='Path to annotation JSON from ClusterMark')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would happen without modifying data')
    parser.add_argument('--reorganize', action='store_true',
                       help='Reorganize images into cluster folders after refinement')
    parser.add_argument('--reorganize-mode', type=str, default='copy',
                       choices=['copy', 'move', 'symlink'],
                       help='How to organize files when --reorganize is used (default: copy)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Similarity threshold for matching unannotated clusters (default: 0.55)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')

    args = parser.parse_args()

    # Configure logging level
    logger.setLevel(getattr(logging, args.log_level))

    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")

    if not scratch_dir:
        logger.error("SCRATCH_DIR not found in environment")
        sys.exit(1)

    # Build config
    config = {}
    if args.threshold is not None:
        config['unannotated_cluster_threshold'] = args.threshold

    try:
        main(
            args.episode_id,
            args.annotation_file,
            scratch_dir,
            args.dry_run,
            args.reorganize,
            args.reorganize_mode,
            config
        )
    except Exception as e:
        logger.error(f"Error during refinement: {e}", exc_info=True)
        sys.exit(1)
