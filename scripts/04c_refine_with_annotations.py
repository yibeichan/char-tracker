import os
import sys
import json
import argparse
import logging
import numpy as np
from dotenv import load_dotenv
from collections import defaultdict

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
MAX_CANNOT_LINK_SAMPLES = 5

def read_json(file_path):
    """Read JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, output_file):
    """Save JSON file."""
    def convert_np(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, default=convert_np)

def extract_track_from_filename(filename):
    """
    Extract track identifier from filename.

    Supports two formats:
    - scene_X_face_Y_frame_Z.jpg (tracking output)
    - scene_X_track_Y_frame_Z.jpg (annotation format)

    Args:
        filename: Image filename (basename or full path)

    Returns:
        tuple: (scene_id, track_id) or (None, None) if not parseable
    """
    basename = os.path.basename(filename)
    # Remove extension
    name_without_ext = os.path.splitext(basename)[0]

    # Parse format: scene_X_(face|track)_Y_frame_Z
    parts = name_without_ext.split('_')

    if len(parts) >= 4 and parts[0] == 'scene':
        # parts[0] = 'scene', parts[1] = scene number, parts[2] = 'face' or 'track', parts[3] = track number
        if parts[2] in ('face', 'track') and parts[3].isdigit():
            return parts[1], parts[3]

    return None, None

def build_image_to_face_mapping(clustering_data):
    """
    Build mapping from image_path to face metadata.

    Args:
        clustering_data: Output from 04_face_clustering.py

    Returns:
        dict: {image_path: {unique_face_id, global_face_id, scene_id, cluster_id, embeddings}}
    """
    mapping = {}
    path_collisions = defaultdict(list)

    for scene_id, faces in clustering_data.items():
        for face_data in faces:
            unique_face_id = face_data['unique_face_id']
            global_face_id = face_data.get('global_face_id')
            cluster_id = face_data.get('cluster_id')

            for img_path, embedding in zip(face_data['image_paths'], face_data['embeddings']):
                # Extract filename for matching (clustering data stores relative paths without cluster dirs)
                filename = os.path.basename(img_path)
                
                # Track collisions for validation
                if filename in mapping:
                    path_collisions[filename].append((unique_face_id, cluster_id, scene_id))
                
                mapping[filename] = {
                    'unique_face_id': unique_face_id,
                    'global_face_id': global_face_id,
                    'scene_id': scene_id,
                    'cluster_id': cluster_id,
                    'embedding': np.array(embedding),
                    'full_image_path': img_path
                }

    # Raise error if collisions are found (ensures data integrity)
    if path_collisions:
        error_msg = [f"Found {len(path_collisions)} filename collisions across clusters."]
        error_msg.append("This causes ambiguous constraint mapping and must be resolved.")
        error_msg.append("\nCollisions:")
        for filename, occurrences in list(path_collisions.items())[:10]:
            cluster_info = ", ".join([f"cluster_{c}" for _, c, _ in occurrences])
            error_msg.append(f"  {filename}: appears in clusters {cluster_info}")
        error_msg.append("\nPlease ensure all image filenames are unique across clusters/scenes.")
        raise ValueError("\n".join(error_msg))

    return mapping

def extract_constraints_from_annotations(annotations, image_mapping):
    """
    Convert cluster-level annotations into pairwise must-link/cannot-link constraints.

    Key changes for "dk" (don't know) labels:
    - "dk" faces only get must-link constraints within the same track
    - Named characters get cross-track must-link constraints

    Args:
        annotations: Annotation JSON from ClusterMark
        image_mapping: Mapping from build_image_to_face_mapping()

    Returns:
        dict: {'must_link': [(face_a, face_b), ...], 'cannot_link': [(face_a, face_b), ...]}
    """
    must_link = []
    cannot_link = []

    # Group faces by assigned label
    label_to_faces = defaultdict(list)
    track_to_dk_faces = defaultdict(list)  # (scene_id, track_id) -> [face_ids]
    unmatched_images = []

    # Labels to skip entirely
    skip_labels = ['not_human', 'background', 'unclear', 'junk', 'not face', 'not clear']

    for cluster_id, cluster_info in annotations['cluster_annotations'].items():
        label = cluster_info['label']

        # Skip non-human clusters
        if label in skip_labels:
            continue

        # Add main cluster images
        for img_path in cluster_info.get('image_paths', []):
            filename = os.path.basename(img_path)
            if filename in image_mapping:
                face_id = image_mapping[filename]['unique_face_id']

                if label == 'dk':
                    # For dk, group by track only
                    scene_id, track_id = extract_track_from_filename(filename)
                    if scene_id and track_id:
                        track_key = (scene_id, track_id)
                        track_to_dk_faces[track_key].append(face_id)
                    else:
                        logger.debug(f"Could not extract track from {filename}, skipping dk face")
                else:
                    # For named characters, group by label (cross-track linking)
                    label_to_faces[label].append(face_id)
            else:
                unmatched_images.append(filename)

        # Add outliers with their correct labels
        for outlier in cluster_info.get('outliers', []):
            outlier_label = outlier['label']
            img_path = outlier['image_path']
            filename = os.path.basename(img_path)

            if filename in image_mapping:
                face_id = image_mapping[filename]['unique_face_id']

                if outlier_label == 'dk':
                    scene_id, track_id = extract_track_from_filename(filename)
                    if scene_id and track_id:
                        track_key = (scene_id, track_id)
                        track_to_dk_faces[track_key].append(face_id)
                    else:
                        logger.debug(f"Could not extract track from {filename}, skipping dk face")
                elif outlier_label not in skip_labels:
                    label_to_faces[outlier_label].append(face_id)
            else:
                unmatched_images.append(filename)

    # Report unmatched images
    if unmatched_images:
        logger.warning(f"Found {len(unmatched_images)} unmatched images in annotations")
        logger.warning(f"First few: {unmatched_images[:5]}")

    # Generate must-link constraints for named characters (cross-track)
    main_char_constraints = 0
    for label, face_ids in label_to_faces.items():
        unique_faces = list(set(face_ids))
        for i in range(len(unique_faces)):
            for j in range(i + 1, len(unique_faces)):
                must_link.append((unique_faces[i], unique_faces[j]))
                main_char_constraints += 1

    # Generate must-link constraints for dk faces (within-track only)
    dk_constraints = 0
    dk_tracks_with_constraints = 0
    for track_key, face_ids in track_to_dk_faces.items():
        unique_faces = list(set(face_ids))
        if len(unique_faces) > 1:
            dk_tracks_with_constraints += 1
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    must_link.append((unique_faces[i], unique_faces[j]))
                    dk_constraints += 1

    logger.info(f"Generated {main_char_constraints} must-link constraints for named characters")
    logger.info(f"Generated {dk_constraints} must-link constraints for dk faces within {dk_tracks_with_constraints} tracks")

    # Generate cannot-link constraints (different labels)
    # Collect all valid labels (including dk as a special case)
    valid_labels = list(label_to_faces.keys())

    # Note: dk faces do NOT participate in cannot-link constraints
    # They only get must-link constraints within the same track (applied earlier)
    # This avoids incorrectly constraining dk faces with named characters

    for i in range(len(valid_labels)):
        for j in range(i + 1, len(valid_labels)):
            faces_a = label_to_faces[valid_labels[i]]
            faces_b = label_to_faces[valid_labels[j]]

            # Sample a few cannot-link pairs (don't need all combinations)
            for face_a in faces_a[:MAX_CANNOT_LINK_SAMPLES]:
                for face_b in faces_b[:MAX_CANNOT_LINK_SAMPLES]:
                    cannot_link.append((face_a, face_b))

    logger.info(f"Generated {len(cannot_link)} cannot-link constraints")

    return {'must_link': must_link, 'cannot_link': cannot_link}

def find_root_cluster(cluster_id, cluster_mapping):
    """
    Follow mapping chain to find root cluster (union-find path compression).
    
    Args:
        cluster_id: Cluster to find root for
        cluster_mapping: Dictionary mapping old_cluster -> new_cluster
    
    Returns:
        Root cluster ID
    """
    root = cluster_id
    while root in cluster_mapping:
        root = cluster_mapping[root]
    return root

def apply_constraints_to_clustering(clustering_data, constraints, image_mapping):
    """
    Apply must-link/cannot-link constraints to refine clusters.

    Strategy:
    1. Build constraint graph
    2. Merge clusters that have must-link constraints (union-find)
    3. Split clusters that have cannot-link constraints (graph-based)

    Args:
        clustering_data: Original clustering output
        constraints: {'must_link': [...], 'cannot_link': [...]}
        image_mapping: Image to face mapping

    Returns:
        dict: Refined clustering data with statistics
    """
    # Step 1: Build face_id to cluster mapping
    face_to_cluster = {}
    cluster_to_faces = defaultdict(list)

    for scene_id, faces in clustering_data.items():
        for face_data in faces:
            unique_face_id = face_data['unique_face_id']
            cluster_id = face_data.get('cluster_id')

            face_to_cluster[unique_face_id] = cluster_id
            cluster_to_faces[cluster_id].append(face_data)

    initial_cluster_count = len(cluster_to_faces)
    logger.info(f"Initial cluster count: {initial_cluster_count}")

    # Step 2: Apply must-link constraints (merge clusters)
    logger.info("Applying must-link constraints...")
    cluster_mapping = {}  # old_cluster -> new_cluster
    merges_performed = 0

    for face_a, face_b in constraints['must_link']:
        if face_a not in face_to_cluster or face_b not in face_to_cluster:
            continue

        # Find root clusters (handle transitivity)
        cluster_a = find_root_cluster(face_to_cluster[face_a], cluster_mapping)
        cluster_b = find_root_cluster(face_to_cluster[face_b], cluster_mapping)

        # If different clusters, merge them
        if cluster_a != cluster_b:
            # Map both to the smaller cluster ID
            target_cluster = min(cluster_a, cluster_b)
            source_cluster = max(cluster_a, cluster_b)

            cluster_mapping[source_cluster] = target_cluster
            merges_performed += 1

            # Update face_to_cluster mapping for all faces in source cluster
            for face_data in cluster_to_faces[source_cluster]:
                face_to_cluster[face_data['unique_face_id']] = target_cluster

    logger.info(f"Performed {merges_performed} cluster merges")

    # Step 3: Apply cannot-link constraints (split clusters)
    logger.info("Applying cannot-link constraints...")
    violations = []

    for face_a, face_b in constraints['cannot_link']:
        if face_a not in face_to_cluster or face_b not in face_to_cluster:
            continue

        # Find root clusters
        cluster_a = find_root_cluster(face_to_cluster[face_a], cluster_mapping)
        cluster_b = find_root_cluster(face_to_cluster[face_b], cluster_mapping)

        # If in same cluster, record violation
        if cluster_a == cluster_b:
            violations.append((face_a, face_b, cluster_a))

    splits_performed = 0
    if violations:
        logger.warning(f"Found {len(violations)} cannot-link violations")
        
        # Get next available cluster ID
        all_cluster_ids = set(cluster_to_faces.keys()) | set(cluster_mapping.values())
        next_cluster_id = max(all_cluster_ids) + 1 if all_cluster_ids else 0

        # Group violations by cluster
        violations_by_cluster = defaultdict(list)
        for face_a, face_b, cluster_id in violations:
            violations_by_cluster[cluster_id].append((face_a, face_b))

        # For each violated cluster, move all conflicting faces to ONE new cluster
        for cluster_id, violation_pairs in violations_by_cluster.items():
            # Collect all faces that need to be separated from this cluster
            faces_to_move = set()
            for _, face_b in violation_pairs:
                # Only move faces if they are still in the cluster being violated
                if face_to_cluster.get(face_b) == cluster_id:
                    faces_to_move.add(face_b)

            if not faces_to_move:
                continue

            # Create ONE new cluster for all faces being moved out of the current cluster
            new_cluster_id = next_cluster_id
            for face_id in faces_to_move:
                face_to_cluster[face_id] = new_cluster_id

            next_cluster_id += 1
            splits_performed += 1
            logger.debug(f"Split cluster {cluster_id}: moved {len(faces_to_move)} faces to cluster {new_cluster_id}")

        logger.info(f"Performed {splits_performed} cluster splits to resolve violations")

    # Step 4: Rebuild clustering data with merged/split clusters
    refined_clustering = {}

    for scene_id, faces in clustering_data.items():
        refined_clustering[scene_id] = []

        for face_data in faces:
            updated_face = face_data.copy()
            old_cluster = face_data.get('cluster_id')

            # Apply cluster mapping with transitivity handling
            new_cluster = find_root_cluster(old_cluster, cluster_mapping)
            
            # Check if this face was directly reassigned (for cannot-link splits)
            unique_face_id = face_data['unique_face_id']
            if unique_face_id in face_to_cluster:
                new_cluster = face_to_cluster[unique_face_id]
            
            updated_face['cluster_id'] = new_cluster
            refined_clustering[scene_id].append(updated_face)

    # Step 5: Calculate final statistics
    final_clusters = set()
    for scene_id, faces in refined_clustering.items():
        for face_data in faces:
            final_clusters.add(face_data['cluster_id'])

    final_cluster_count = len(final_clusters)
    
    statistics = {
        'initial_clusters': initial_cluster_count,
        'final_clusters': final_cluster_count,
        'merges_performed': merges_performed,
        'splits_performed': splits_performed,
        'must_link_constraints': len(constraints['must_link']),
        'cannot_link_constraints': len(constraints['cannot_link']),
        'cannot_link_violations': len(violations)
    }

    return refined_clustering, statistics

def validate_constraints(refined_clustering, constraints):
    """
    Validate that constraints are satisfied in the refined clustering.
    
    Args:
        refined_clustering: Refined clustering data
        constraints: Original constraints
    
    Returns:
        dict: Validation statistics
    """
    # Build face_id to cluster mapping
    face_to_cluster = {}
    for scene_id, faces in refined_clustering.items():
        for face_data in faces:
            face_to_cluster[face_data['unique_face_id']] = face_data['cluster_id']
    
    # Check must-link constraints
    must_link_satisfied = 0
    must_link_violated = 0
    for face_a, face_b in constraints['must_link']:
        if face_a in face_to_cluster and face_b in face_to_cluster:
            if face_to_cluster[face_a] == face_to_cluster[face_b]:
                must_link_satisfied += 1
            else:
                must_link_violated += 1
    
    # Check cannot-link constraints
    cannot_link_satisfied = 0
    cannot_link_violated = 0
    for face_a, face_b in constraints['cannot_link']:
        if face_a in face_to_cluster and face_b in face_to_cluster:
            if face_to_cluster[face_a] != face_to_cluster[face_b]:
                cannot_link_satisfied += 1
            else:
                cannot_link_violated += 1
    
    return {
        'must_link_satisfied': must_link_satisfied,
        'must_link_violated': must_link_violated,
        'cannot_link_satisfied': cannot_link_satisfied,
        'cannot_link_violated': cannot_link_violated
    }

def validate_track_integrity(refined_clustering):
    """
    Validate that all faces from the same track (unique_face_id) are in the same cluster.

    This should always be true by construction since the refinement operates at the
    track level (unique_face_id), but this provides an explicit safety check.

    Args:
        refined_clustering: Refined clustering data

    Returns:
        dict: Validation result with 'valid' boolean and any violations found
    """
    # Build mapping from unique_face_id to cluster_id
    track_to_cluster = {}

    for scene_id, faces in refined_clustering.items():
        for face_data in faces:
            unique_face_id = face_data['unique_face_id']
            cluster_id = face_data['cluster_id']

            if unique_face_id in track_to_cluster:
                if track_to_cluster[unique_face_id] != cluster_id:
                    return {
                        'valid': False,
                        'violations': [
                            {
                                'track': unique_face_id,
                                'clusters': [track_to_cluster[unique_face_id], cluster_id]
                            }
                        ]
                    }
            else:
                track_to_cluster[unique_face_id] = cluster_id

    return {'valid': True, 'violations': [], 'total_tracks': len(track_to_cluster)}

def main(episode_id, annotation_file, scratch_dir):
    """
    Refine clustering using annotations from ClusterMark.

    Args:
        episode_id: Episode identifier (e.g., 'friends_s01e05')
        annotation_file: Path to annotation JSON from ClusterMark
        scratch_dir: SCRATCH_DIR from environment
    """
    logger.info(f"=" * 70)
    logger.info(f"Refining clustering for {episode_id}")
    logger.info(f"=" * 70)

    # Load annotations
    logger.info(f"Loading annotations from: {annotation_file}")
    annotations = read_json(annotation_file)

    # Validate episode_id matches
    annotation_episode_id = annotations.get('metadata', {}).get('episode_id')
    if annotation_episode_id and annotation_episode_id != episode_id:
        raise ValueError(
            f"Episode ID mismatch: annotation={annotation_episode_id}, argument={episode_id}"
        )

    # Load original clustering data
    clustering_file = os.path.join(
        scratch_dir, "output", "face_clustering",
        f"{episode_id}_matched_faces_with_clusters.json"
    )
    logger.info(f"Loading clustering data from: {clustering_file}")
    clustering_data = read_json(clustering_file)

    # Build image to face mapping
    logger.info("Building image-to-face mapping...")
    image_mapping = build_image_to_face_mapping(clustering_data)
    logger.info(f"Mapped {len(image_mapping)} images to faces")

    # Extract constraints from annotations
    logger.info("Extracting constraints from annotations...")
    constraints = extract_constraints_from_annotations(annotations, image_mapping)

    # Apply constraints to refine clustering
    logger.info("Refining clusters...")
    refined_clustering, statistics = apply_constraints_to_clustering(
        clustering_data, constraints, image_mapping
    )

    # Validate constraints
    logger.info("Validating constraint satisfaction...")
    validation = validate_constraints(refined_clustering, constraints)

    # Validate track integrity
    logger.info("Validating track integrity...")
    track_validation = validate_track_integrity(refined_clustering)
    if not track_validation['valid']:
        logger.error("Track integrity validation failed!")
        logger.error(f"Violations: {track_validation['violations']}")
        raise ValueError("Track integrity violation detected - this should never happen")
    logger.info(f"✓ Track integrity validated for {track_validation['total_tracks']} tracks")

    # Save refined clustering
    output_file = os.path.join(
        scratch_dir, "output", "face_clustering",
        f"{episode_id}_matched_faces_with_clusters_refined.json"
    )
    save_json(refined_clustering, output_file)

    # Report statistics
    logger.info(f"\n" + "=" * 70)
    logger.info("REFINEMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Initial clusters:              {statistics['initial_clusters']}")
    logger.info(f"Final clusters:                {statistics['final_clusters']}")
    logger.info(f"Cluster change:                {statistics['final_clusters'] - statistics['initial_clusters']:+d}")
    logger.info(f"Merges performed:              {statistics['merges_performed']}")
    logger.info(f"Splits performed:              {statistics['splits_performed']}")
    logger.info(f"\nConstraint Statistics:")
    logger.info(f"Must-link constraints:         {statistics['must_link_constraints']}")
    logger.info(f"  Satisfied:                   {validation['must_link_satisfied']}")
    logger.info(f"  Violated:                    {validation['must_link_violated']}")
    logger.info(f"Cannot-link constraints:       {statistics['cannot_link_constraints']}")
    logger.info(f"  Satisfied:                   {validation['cannot_link_satisfied']}")
    logger.info(f"  Violated:                    {validation['cannot_link_violated']}")
    logger.info("=" * 70)

    logger.info(f"\n✓ Refined clustering saved to: {output_file}")
    logger.info("\nNext step:")
    logger.info(f"  python ./scripts/08_generate_character_timestamps.py {episode_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Refine face clustering using human annotations from ClusterMark'
    )
    parser.add_argument('episode_id', type=str,
                       help='Episode ID (e.g., friends_s01e05)')
    parser.add_argument('annotation_file', type=str,
                       help='Path to annotation JSON from ClusterMark')
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

    try:
        main(args.episode_id, args.annotation_file, scratch_dir)
    except Exception as e:
        logger.error(f"Error during refinement: {e}", exc_info=True)
        sys.exit(1)
