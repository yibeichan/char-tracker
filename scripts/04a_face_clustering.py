import os
import sys
import argparse
from dotenv import load_dotenv
from collections import defaultdict
import numpy as np
import pandas as pd
import json

# Add src directory to path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from face_clusterer import FaceEmbedder, FaceClusterer
import utils

def match_clusters_with_unique_faces(clustered_faces, unique_faces_per_scene):
    # Build a mapping from unique_face_id to cluster_id
    unique_face_to_cluster = {}
    for cluster_id, faces_in_cluster in clustered_faces.items():
        for face_data in faces_in_cluster:
            unique_face_id = face_data['unique_face_id']
            unique_face_to_cluster[unique_face_id] = cluster_id  # Overwrites if duplicate, which is acceptable here

    matched_results = {}

    for scene_id, unique_faces in unique_faces_per_scene.items():
        matched_results[scene_id] = []

        for unique_face in unique_faces:
            unique_face_id = unique_face['unique_face_id']
            matched_data = {
                "unique_face_id": unique_face_id,
                "global_face_id": unique_face.get('global_face_id', None),
                "cluster_id": None,
                "embeddings": [],
                "image_paths": []
            }

            cluster_id = unique_face_to_cluster.get(unique_face_id)
            if cluster_id is not None:
                matched_data["cluster_id"] = cluster_id
                # Collect embeddings and image paths from the cluster for this unique_face_id
                embeddings = []
                image_paths = []
                for face_data in clustered_faces[cluster_id]:
                    if face_data['unique_face_id'] == unique_face_id:
                        embeddings.append(face_data['embedding'].tolist())
                        image_paths.append(face_data['image_path'])
                matched_data["embeddings"] = embeddings
                matched_data["image_paths"] = image_paths
            else:
                # Handle unmatched unique_face_id if necessary
                print(f"Warning: unique_face_id {unique_face_id} not found in any cluster.")

            matched_results[scene_id].append(matched_data)

    return matched_results

def read_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, output_file):
    """Save the selected frames to a JSON file."""
    def convert_np(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4, default=convert_np)
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        sys.exit(1)


def validate_track_integrity(clustering_data):
    """
    Check that no track is split across multiple clusters.

    Args:
        clustering_data: Output from clustering (matched_faces dict)

    Returns:
        tuple: (is_valid, violations_dict)
            - is_valid: True if no violations found
            - violations_dict: dict with track_id -> set of cluster_ids
    """
    track_to_clusters = defaultdict(set)

    for scene_id, faces in clustering_data.items():
        for face_data in faces:
            unique_face_id = face_data['unique_face_id']
            cluster_id = face_data.get('cluster_id')
            if cluster_id is not None:
                track_to_clusters[unique_face_id].add(cluster_id)

    # Find violations (tracks in multiple clusters)
    violations = {t: cs for t, cs in track_to_clusters.items() if len(cs) > 1}

    if violations:
        print(f"Warning: Found {len(violations)} tracks split across clusters")
        for track, clusters in list(violations.items())[:5]:  # Show first 5
            print(f"  {track}: {clusters}")
        if len(violations) > 5:
            print(f"  ... and {len(violations) - 5} more")

    return len(violations) == 0, violations


def log_cluster_statistics(clustering_data):
    """
    Log statistics about clusters for quality assessment.

    Args:
        clustering_data: Output from clustering (matched_faces dict)
    """
    cluster_stats = defaultdict(lambda: {
        'faces': 0,
        'tracks': set(),
        'scenes': set()
    })

    for scene_id, faces in clustering_data.items():
        for face_data in faces:
            cluster_id = face_data.get('cluster_id', 'unassigned')
            cluster_stats[cluster_id]['faces'] += len(face_data.get('image_paths', []))
            cluster_stats[cluster_id]['tracks'].add(face_data['unique_face_id'])
            cluster_stats[cluster_id]['scenes'].add(scene_id)

    print(f"\nCluster Statistics: {len(cluster_stats)} clusters")

    # Sort by face count descending
    sorted_clusters = sorted(cluster_stats.items(),
                           key=lambda x: x[1]['faces'],
                           reverse=True)

    for cluster_id, stats in sorted_clusters:
        print(f"  {cluster_id}: {stats['faces']} faces, "
              f"{len(stats['tracks'])} tracks, {len(stats['scenes'])} scenes")



def main(video_name, face_selection_file, output_dir,
         similarity_threshold=0.6, min_scenes=2, same_scene_different_track_threshold=0.75):
    """
    Run face clustering using buffalo_l model.

    Args:
        video_name: Name of the input video file without extension
        face_selection_file: Path to face selection JSON
        output_dir: Directory for output files
        similarity_threshold: Cosine similarity threshold for clustering (default: 0.6)
        min_scenes: Minimum scenes required for a valid cluster (default: 2)
        same_scene_different_track_threshold: Higher threshold for same-scene different-track
            pairs (default: 0.75). Prevents false merges of different characters in same scene.
    """
    face_embedder = FaceEmbedder()
    face_clusterer = FaceClusterer(
        similarity_threshold=similarity_threshold,
        max_iterations=100,
        min_scenes=min_scenes,
        same_scene_different_track_threshold=same_scene_different_track_threshold
    )
    print(f"Clustering with similarity_threshold={similarity_threshold}, "
          f"min_scenes={min_scenes}, same_scene_different_track_threshold={same_scene_different_track_threshold}")

    selected_faces = read_json(face_selection_file)
    image_dir = os.path.dirname(face_selection_file)

    # Extract embeddings
    print("Extracting face embeddings...")
    face_embeddings = face_embedder.get_face_embeddings(selected_faces, image_dir)

    # Cluster faces
    print("Clustering faces...")
    consolidated_clusters = face_clusterer.cluster_faces(face_embeddings)

    # Match clusters with unique faces
    matched_faces = match_clusters_with_unique_faces(consolidated_clusters, selected_faces)

    # Validate track integrity
    print("\nValidating track integrity...")
    is_valid, violations = validate_track_integrity(matched_faces)
    if is_valid:
        print("Track integrity validated: no tracks split across clusters")
    else:
        print(f"Track integrity: {len(violations)} tracks split across clusters (may be expected with top-n > 1)")

    # Log cluster statistics
    log_cluster_statistics(matched_faces)

    output_file = os.path.join(output_dir, f'{video_name}_matched_faces_with_clusters.json')
    save_json(matched_faces, output_file)

    print(f"\nSaved results to: {output_file}")
    print("Processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Clustering with Embeddings (buffalo_l model)')
    parser.add_argument('video_name', type=str, help='Name of the input video file without extension.')
    parser.add_argument('--similarity-threshold', type=float, default=0.6,
                       help='Cosine similarity threshold for clustering (default: 0.6). Lower values create fewer, larger clusters.')
    parser.add_argument('--min-scenes', type=int, default=2,
                       help='Minimum scenes required for a valid cluster (default: 2). '
                            'Clusters with fewer scenes are merged into nearest valid cluster. '
                            'Set to 1 to disable cross-scene validation.')
    parser.add_argument('--same-scene-track-threshold', type=float, default=0.75,
                       help='Higher similarity threshold for same-scene different-track pairs (default: 0.75). '
                            'Prevents false merges of different characters in the same scene.')

    args = parser.parse_args()
    video_name = args.video_name

    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")

    face_selection_file = utils.get_output_path(
        scratch_dir, utils.OUTPUT_DIR_FACE_TRACKING,
        f"{video_name}", f"{video_name}_selected_frames_per_face.json"
    )
    output_dir = utils.get_output_path(scratch_dir, utils.OUTPUT_DIR_FACE_CLUSTERING)
    os.makedirs(output_dir, exist_ok=True)

    main(video_name, face_selection_file, output_dir,
         similarity_threshold=args.similarity_threshold,
         min_scenes=args.min_scenes,
         same_scene_different_track_threshold=args.same_scene_track_threshold)
