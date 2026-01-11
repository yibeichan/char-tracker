"""
Cluster Refiner Module

This module provides intelligent label propagation for face clustering refinement.
It handles:
- Ground truth extraction from annotations (small clusters, labeled outliers)
- Outlier separation from parent clusters
- Track-based label propagation for unannotated faces
- Cross-cluster DK linking via track and embedding similarity
- Constrained Chinese Whispers re-clustering
- Conflict resolution for tracks with multiple labels

Author: Pipeline stage 04c refinement
Date: 2026-01-09
"""

import os
import logging
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

import constants as constants

logger = logging.getLogger(__name__)


class ClusterRefiner:
    """
    Refines face clustering using annotations with intelligent label propagation.

    The new refiner workflow:
    1. Extract ground truth from small clusters and labeled outliers
    2. Separate outliers from parent clusters
    3. Link DK groups across clusters via track and embedding
    4. Re-run Chinese Whispers with constraints
    5. Validate track integrity and merge

    Ground truth rules:
    - Small clusters (n < 15) labeled with main character = absolute GT
    - Outliers labeled with main characters = absolute GT
    - Large clusters (n >= 15) labeled with main character = GT for main images
    """

    # Expose constants as class attributes for external access
    MAIN_CHARACTERS = constants.MAIN_CHARACTERS
    SKIP_LABELS = constants.SKIP_LABELS
    DK_LABEL_PREFIX = constants.DK_LABEL_PREFIX
    QUALITY_MODIFIERS = constants.QUALITY_MODIFIERS

    # Default configuration
    DEFAULT_CONFIG = {
        'unannotated_cluster_threshold': 0.55,
        'dk_propagation_enabled': True,
        'max_cannot_link_samples': 5,
        # New config options
        'small_cluster_threshold': 15,      # Below this = absolute ground truth
        'dk_linking_threshold': 0.6,        # For DK embedding-based linking (conservative)
        'must_link_track_threshold': 0.5,   # Min faces in track for must-link
        'chinese_whispers_iterations': 100,
        'similarity_threshold': 0.6,        # For graph construction
        'large_cluster_auto_trust': True,   # Auto-trust all faces in large labeled clusters
        'small_residual_threshold': 5,      # Below this = try to merge
        'small_residual_similarity': 0.7,   # Threshold for merging small clusters
        # Scene proximity and track consistency options
        'scene_proximity_bonus': 0.03,      # Max bonus for adjacent scenes in DK linking
        'scene_proximity_decay': 0.015,     # Bonus decay per additional scene distance
        'track_consistency_auto_correct': True,  # Auto-correct track inconsistencies
        'scene_proximity_enabled': True,    # Enable scene proximity weighting in DK linking
    }

    def __init__(self, annotations: dict, clustering_data: dict, config: Optional[dict] = None):
        """
        Initialize the ClusterRefiner.

        Args:
            annotations: Annotation JSON from ClusterMark
            clustering_data: Clustering output from stage 04
            config: Optional configuration overrides
        """
        self.annotations = annotations
        self.clustering_data = clustering_data
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # State for tracking propagation
        self.track_to_label: Dict[str, str] = {}
        self.propagated_labels: Dict[str, str] = {}  # face_id -> label
        self.cluster_matches: Dict[int, str] = {}  # cluster_id -> label
        self.dk_propagated: Dict[str, str] = {}  # face_id -> label
        self.conflicts_resolved: int = 0

        # New state for ground truth and DK linking
        self.ground_truth_tracks: Dict[str, str] = {}  # {(scene_id, track_id) -> character}
        self.ground_truth_faces: Set[str] = set()  # face_ids confirmed as ground truth
        self.dk_track_mapping: Dict[Tuple[str, int], str] = {}  # {(scene_id, track_id) -> dk_label}
        self.dk_by_cluster: Dict[int, Dict[str, List[str]]] = {}  # {cluster_id: {dk_label: [face_ids]}}
        self.locked_clusters: Dict[str, List[str]] = {}  # {character: [face_ids]} - cannot be modified
        self.separated_outliers: Dict[int, List[Dict]] = {}  # {cluster_id: [outlier_infos]}

        # Statistics
        self.stats = {
            'tracks_mapped': 0,
            'tracks_with_conflicts': 0,
            'faces_propagated': 0,
            'clusters_matched': 0,
            'dk_converted': 0,
            # New stats
            'ground_truth_tracks': 0,
            'ground_truth_faces': 0,
            'dk_groups_linked': 0,
            'track_based_merges': 0,
            'constraint_violations': 0,
        }

        # Build image mapping once
        self.image_mapping = self._build_image_to_face_mapping(clustering_data)

        # Build normalized lookup for matching with annotation file (which uses 'track' format)
        self.image_mapping_norm = {}
        for filename, data in self.image_mapping.items():
            norm_key = self._normalize_filename_key(filename)
            self.image_mapping_norm[norm_key] = data

    @staticmethod
    def _normalize_filename_key(filename: str) -> str:
        """
        Normalize filename for consistent lookup.

        Handles both formats:
        - scene_X_face_Y_frame_Z.jpg (original tracking)
        - scene_X_track_Y_frame_Z.jpg (reorganized 04b format)

        Both normalize to: scene_X_Y_Z (without face/track/frame keywords)

        Args:
            filename: Image filename

        Returns:
            Normalized key for matching
        """
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')

        # Rebuild as scene_X_Y_Z (removing face/track keywords, keeping frame number)
        # Frame number is REQUIRED for unique keys per image
        if len(parts) >= 6 and parts[0] == 'scene' and parts[2] in ('face', 'track') and parts[4] == 'frame':
            return f"{parts[0]}_{parts[1]}_{parts[3]}_{parts[5]}"

        return filename

    @staticmethod
    def extract_track_from_filename(filename: str) -> Optional[str]:
        """
        Extract track identifier from filename.

        Supports formats:
        - scene_X_face_Y_frame_Z.jpg (tracking output)
        - scene_X_track_Y_frame_Z.jpg (annotation format)

        Args:
            filename: Image filename (basename or full path)

        Returns:
            Combined track ID "scene_X_track_Y" or None if not parseable
        """
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]

        parts = name_without_ext.split('_')

        if len(parts) >= 4 and parts[0] == 'scene':
            if parts[2] in ('face', 'track') and parts[3].isdigit():
                return f"scene_{parts[1]}_track_{parts[3]}"

        return None

    def _build_image_to_face_mapping(self, clustering_data: dict) -> dict:
        """
        Build mapping from image_path to face metadata.

        Args:
            clustering_data: Output from 04_face_clustering.py

        Returns:
            dict: {image_path: {unique_face_id, global_face_id, scene_id, cluster_id, embedding}}
        """
        mapping = {}
        path_collisions = defaultdict(list)

        for scene_id, faces in clustering_data.items():
            for face_data in faces:
                unique_face_id = face_data['unique_face_id']
                global_face_id = face_data.get('global_face_id')
                cluster_id = face_data.get('cluster_id')

                for img_path, embedding in zip(face_data['image_paths'], face_data['embeddings']):
                    filename = os.path.basename(img_path)

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

        if path_collisions:
            logger.warning(f"Found {len(path_collisions)} filename collisions across clusters.")

        return mapping

    def _extract_ground_truth(self) -> Dict[str, Any]:
        """
        Extract ground truth from annotations based on rules.

        Rules:
        1. Small clusters (n < 15) labeled with main character = absolute GT
        2. Outliers labeled with main characters = absolute GT
        3. Large clusters (n >= 15) labeled with main character = GT for main images

        Also extracts DK groups with their track mappings.

        Quality modifiers (@poor, @blurry, @dark, @profile, @back) are stripped from labels
        and the face weights are stored for use during centroid computation.

        Returns:
            dict: Ground truth data with keys:
                - ground_truth_tracks: {(scene_id, track_id) -> character}
                - ground_truth_faces: set of face_ids
                - dk_track_mapping: {(scene_id, track_id) -> dk_label}
                - dk_by_cluster: {cluster_id: {dk_label: [face_ids]}}
                - locked_clusters: {character: [face_ids]}
        """
        small_threshold = self.config['small_cluster_threshold']
        gt_tracks: Dict[Tuple[str, int], str] = {}
        gt_faces: Set[str] = set()
        dk_tracks: Dict[Tuple[str, int], str] = {}
        dk_by_cluster: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        locked_clusters: Dict[str, Set[str]] = {char: set() for char in self.MAIN_CHARACTERS}
        face_weights: Dict[str, float] = {}  # Store weight per face_id

        logger.info(f"Extracting ground truth (small cluster threshold: {small_threshold})...")

        for cluster_key, cluster_info in self.annotations['cluster_annotations'].items():
            # Extract numeric cluster ID
            if '-' in cluster_key:
                cluster_id = int(cluster_key.split('-')[-1])
            else:
                cluster_id = int(cluster_key)

            # Parse label to extract base label and quality modifiers
            raw_label = cluster_info['label'].lower()
            main_label, main_modifiers = self._parse_label_with_modifiers(raw_label)
            image_count = cluster_info.get('image_count', len(cluster_info.get('image_paths', [])))

            # Check if this is a main character cluster
            is_main_char = main_label in self.MAIN_CHARACTERS
            is_small_cluster = image_count < small_threshold

            # Process main cluster images
            for img_path in cluster_info.get('image_paths', []):
                filename = os.path.basename(img_path)
                norm_key = self._normalize_filename_key(filename)

                if norm_key not in self.image_mapping_norm:
                    continue

                face_data = self.image_mapping_norm[norm_key]
                face_id = face_data['unique_face_id']
                track_id = self.extract_track_from_filename(filename)

                # Extract scene and track numbers for track-based mapping
                if track_id:
                    parts = track_id.split('_')
                    scene_id = parts[1]
                    track_num = int(parts[3])
                    track_key = (scene_id, track_num)
                else:
                    track_key = None

                # Ground truth rules
                if is_main_char:
                    # Rule 1: Small clusters = absolute GT
                    # Rule 3: Large clusters = GT for main images
                    gt_faces.add(face_id)
                    locked_clusters[main_label].add(face_id)

                    if track_key:
                        gt_tracks[track_key] = main_label

            # Process outliers
            for outlier in cluster_info.get('outliers', []):
                # Parse label to extract base label and quality modifiers
                raw_label = outlier['label'].lower()
                outlier_label, outlier_modifiers = self._parse_label_with_modifiers(raw_label)

                if outlier_label in self.SKIP_LABELS:
                    continue

                img_path = outlier['image_path']
                filename = os.path.basename(img_path)
                norm_key = self._normalize_filename_key(filename)

                if norm_key not in self.image_mapping_norm:
                    continue

                face_data = self.image_mapping_norm[norm_key]
                face_id = face_data['unique_face_id']
                track_id = self.extract_track_from_filename(filename)

                # Store face weight based on quality modifiers
                if outlier_modifiers & constants.QUALITY_MODIFIERS:
                    face_weights[face_id] = 0.5

                if track_id:
                    parts = track_id.split('_')
                    scene_id = parts[1]
                    track_num = int(parts[3])
                    track_key = (scene_id, track_num)
                else:
                    track_key = None

                # Rule 2: Outliers labeled with main characters = absolute GT
                if outlier_label in self.MAIN_CHARACTERS:
                    gt_faces.add(face_id)
                    locked_clusters[outlier_label].add(face_id)

                    if track_key:
                        gt_tracks[track_key] = outlier_label

                # Handle DK labels
                elif outlier_label.startswith(self.DK_LABEL_PREFIX):
                    if track_key:
                        dk_tracks[track_key] = outlier_label

                    dk_by_cluster[cluster_id][outlier_label].append(face_id)

        # Store in instance variables
        self.ground_truth_tracks = {f"{k[0]}_track_{k[1]}": v for k, v in gt_tracks.items()}
        self.ground_truth_faces = gt_faces
        self.dk_track_mapping = dk_tracks
        self.dk_by_cluster = dict(dk_by_cluster)
        self.locked_clusters = {k: list(v) for k, v in locked_clusters.items()}
        self.face_weights = face_weights  # Store face weights for quality modifiers

        # Update statistics
        self.stats['ground_truth_tracks'] = len(self.ground_truth_tracks)
        self.stats['ground_truth_faces'] = len(gt_faces)

        # Count faces per locked character
        for char, faces in locked_clusters.items():
            if faces:
                logger.info(f"  Locked cluster for {char}: {len(faces)} ground truth faces")

        # Count DK groups
        total_dk_groups = sum(len(groups) for groups in dk_by_cluster.values())
        logger.info(f"  Found {total_dk_groups} DK groups across {len(dk_by_cluster)} clusters")

        return {
            'ground_truth_tracks': self.ground_truth_tracks,
            'ground_truth_faces': gt_faces,
            'dk_track_mapping': dk_tracks,
            'dk_by_cluster': dict(dk_by_cluster),
            'locked_clusters': self.locked_clusters,
        }

    def _separate_outliers(self) -> Tuple[Dict[int, List[str]], Dict[str, List[str]]]:
        """
        Separate outliers from parent clusters into their own groups.

        Now includes track consistency validation before separation.

        Returns:
            tuple: (remaining_cluster_faces, outlier_groups)
                - remaining_cluster_faces: {cluster_id: [remaining face_ids]}
                - outlier_groups: {label: [face_ids]}
        """
        logger.info("Separating outliers from parent clusters...")

        # Track which faces are in each cluster
        cluster_to_faces: Dict[int, Set[str]] = defaultdict(set)
        for scene_id, faces in self.clustering_data.items():
            for face_data in faces:
                cluster_id = face_data.get('cluster_id')
                if cluster_id is not None:
                    cluster_to_faces[cluster_id].add(face_data['unique_face_id'])

        # Step 0: Validate track consistency within each cluster before separating
        # Find all tracks with inconsistent labels and remove them from ground truth
        tracks_to_remove: Set[str] = set()  # Track IDs to remove from GT

        if self.config.get('track_consistency_auto_correct', True):
            logger.info("  Validating track consistency within clusters...")
            for cluster_id in cluster_to_faces.keys():
                inconsistent_tracks = self._find_inconsistent_tracks_in_cluster(cluster_id)
                if inconsistent_tracks:
                    tracks_to_remove.update(inconsistent_tracks)

            if tracks_to_remove:
                # Remove faces from inconsistent tracks from ground truth
                faces_to_remove = set()
                for track_id in tracks_to_remove:
                    # Find all faces in this track
                    for scene_id, faces in self.clustering_data.items():
                        for face_data in faces:
                            for img_path in face_data.get('image_paths', []):
                                if self.extract_track_from_filename(img_path) == track_id:
                                    faces_to_remove.add(face_data['unique_face_id'])
                                    break

                # Remove from ground truth
                removed_count = 0
                for face_id in faces_to_remove:
                    if face_id in self.ground_truth_faces:
                        self.ground_truth_faces.remove(face_id)
                        removed_count += 1
                        # Also remove from locked clusters
                        for char_faces in self.locked_clusters.values():
                            if face_id in char_faces:
                                char_faces.remove(face_id)

                logger.info(f"  Found {len(tracks_to_remove)} tracks with label inconsistencies")
                logger.info(f"  Removed {removed_count} faces from ground truth due to track inconsistencies")

        # Track outlier faces to remove from clusters
        outlier_faces: Set[str] = set()
        outlier_groups: Dict[str, List[str]] = defaultdict(list)

        for cluster_key, cluster_info in self.annotations['cluster_annotations'].items():
            if '-' in cluster_key:
                cluster_id = int(cluster_key.split('-')[-1])
            else:
                cluster_id = int(cluster_key)

            # Process outliers
            for outlier in cluster_info.get('outliers', []):
                outlier_label = outlier['label'].lower()

                if outlier_label in self.SKIP_LABELS:
                    continue

                img_path = outlier['image_path']
                filename = os.path.basename(img_path)
                norm_key = self._normalize_filename_key(filename)

                if norm_key not in self.image_mapping_norm:
                    continue

                face_id = self.image_mapping_norm[norm_key]['unique_face_id']

                # Skip ground truth faces - they stay in locked clusters
                if face_id in self.ground_truth_faces:
                    continue

                outlier_faces.add(face_id)
                outlier_groups[outlier_label].append(face_id)

            # Store separated outliers for this cluster
            if cluster_id in cluster_to_faces:
                cluster_outliers = cluster_to_faces[cluster_id] & outlier_faces
                self.separated_outliers[cluster_id] = [
                    {'face_id': f, 'label': self._get_face_label(f)}
                    for f in cluster_outliers
                ]

        # Remove outliers from their original clusters
        remaining_cluster_faces: Dict[int, List[str]] = {}
        for cluster_id, faces in cluster_to_faces.items():
            remaining = faces - outlier_faces
            remaining_cluster_faces[cluster_id] = list(remaining)

        # Also separate DK faces from locked clusters
        # They shouldn't be re-clustered with the main character
        for char, face_ids in self.locked_clusters.items():
            for face_id in face_ids:
                if face_id in outlier_faces:
                    outlier_faces.remove(face_id)

        logger.info(f"  Separated {len(outlier_faces)} outlier faces from clusters")
        logger.info(f"  Created {len(outlier_groups)} outlier groups")

        return remaining_cluster_faces, dict(outlier_groups)

    def _get_face_label(self, face_id: str) -> str:
        """Get the label for a face_id from various sources."""
        # Check ground truth
        for char, faces in self.locked_clusters.items():
            if face_id in faces:
                return char

        # Check propagated labels
        if face_id in self.propagated_labels:
            return self.propagated_labels[face_id]

        return 'unknown'

    @staticmethod
    def _get_scene_distance(scene_a: str, scene_b: str) -> int:
        """
        Calculate distance between two scene IDs.

        Args:
            scene_a: Scene ID like '001' or full 'scene_001'
            scene_b: Scene ID like '003' or full 'scene_003'

        Returns:
            int: Absolute difference in scene numbers (e.g., '001' and '003' -> 2)
        """
        # Extract numeric part if full scene ID
        if '_' in scene_a:
            scene_a = scene_a.split('_')[1]
        if '_' in scene_b:
            scene_b = scene_b.split('_')[1]

        try:
            return abs(int(scene_a) - int(scene_b))
        except (ValueError, IndexError):
            return 999  # Large distance for unparseable scene IDs

    def _compute_scene_weighted_similarity(
        self,
        similarity: float,
        scene_distance: int
    ) -> float:
        """
        Apply small scene proximity bonus to similarity score.

        The bonus is small (max +0.03) so it only helps when embeddings
        are already very close. This prevents scene proximity from
        overpowering actual face similarity.

        Args:
            similarity: Base cosine similarity
            scene_distance: Distance between scenes (from _get_scene_distance)

        Returns:
            float: Similarity score with proximity bonus applied
        """
        if not self.config.get('scene_proximity_enabled', True):
            return similarity

        bonus = self.config.get('scene_proximity_bonus', 0.03)
        decay = self.config.get('scene_proximity_decay', 0.015)

        if scene_distance == 1:
            return similarity + bonus
        elif scene_distance == 2:
            return similarity + (bonus - decay)
        else:
            return similarity  # No bonus for distant scenes

    @staticmethod
    def _parse_label_with_modifiers(label: str) -> Tuple[str, Set[str]]:
        """
        Parse a label to extract the base label and any quality modifiers.

        This implements the "simpler approach" for quality modifiers without
        requiring ClusterMark UI changes. Quality attributes are encoded directly
        in the label string using @-prefixed modifiers (e.g., "rachel @poor").

        The future ClusterMark format (see docs/CLUSTERMARK_QUALITY_MODIFIERS.md)
        would use a separate 'quality' field in the annotation JSON, but this
        label-based approach works immediately with the existing UI.

        Examples:
            "rachel" -> ("rachel", set())
            "rachel @poor" -> ("rachel", {"@poor"})
            "dk1 @blurry @dark" -> ("dk1", {"@blurry", "@dark"})

        Args:
            label: The label string to parse

        Returns:
            tuple: (base_label, set of modifiers)
        """
        if not label:
            return label, set()

        parts = label.lower().split()
        base_label = parts[0]
        modifiers = set()

        for part in parts[1:]:
            if part.startswith('@'):
                modifiers.add(part)

        return base_label, modifiers

    def _get_face_weight(self, label: str) -> float:
        """
        Get the weight to use for a face based on quality modifiers.

        Faces with quality modifiers (@poor, @blurry, @dark, @profile, @back)
        get lower weight during clustering since their embeddings may be less reliable.

        Args:
            label: The face label (may include quality modifiers)

        Returns:
            float: Weight multiplier (0.5 for poor quality, 1.0 for normal)
        """
        base_label, modifiers = self._parse_label_with_modifiers(label)

        # Check for quality modifiers
        if modifiers & constants.QUALITY_MODIFIERS:
            return 0.5  # Down-weight poor quality faces

        return 1.0  # Full weight for normal quality faces

    def _find_inconsistent_tracks_in_cluster(self, cluster_id: int) -> Set[str]:
        """
        Find tracks with inconsistent labels within a cluster.

        A track is inconsistent if it has faces labeled with different
        characters (e.g., some "rachel", some "dk1").

        Args:
            cluster_id: The cluster ID to validate

        Returns:
            set: Track IDs with label inconsistencies
        """
        cluster_key = None

        # Find the cluster key for this cluster_id
        for key in self.annotations['cluster_annotations'].keys():
            if '-' in key:
                numeric_id = int(key.split('-')[-1])
            else:
                numeric_id = int(key)

            if numeric_id == cluster_id:
                cluster_key = key
                break

        if cluster_key is None:
            return set()

        cluster_info = self.annotations['cluster_annotations'][cluster_key]
        # Parse label to extract base label and quality modifiers
        raw_label = cluster_info['label'].lower()
        main_label, main_modifiers = self._parse_label_with_modifiers(raw_label)

        # Skip if this is a DK or skip label cluster
        if main_label in self.SKIP_LABELS or main_label.startswith(self.DK_LABEL_PREFIX):
            return set()

        # Build track to labels mapping for faces in this cluster
        track_labels: Dict[str, Set[str]] = defaultdict(set)

        # Check main cluster images
        for img_path in cluster_info.get('image_paths', []):
            filename = os.path.basename(img_path)
            track_id = self.extract_track_from_filename(filename)
            if track_id:
                track_labels[track_id].add(main_label)

        # Check outliers
        for outlier in cluster_info.get('outliers', []):
            # Parse label to extract base label and quality modifiers
            raw_label = outlier['label'].lower()
            outlier_label, outlier_modifiers = self._parse_label_with_modifiers(raw_label)

            if outlier_label in self.SKIP_LABELS:
                continue

            img_path = outlier['image_path']
            filename = os.path.basename(img_path)
            track_id = self.extract_track_from_filename(filename)

            if track_id:
                track_labels[track_id].add(outlier_label)

        # Find tracks with multiple different labels
        inconsistent_tracks = {
            track_id for track_id, labels in track_labels.items()
            if len(labels) > 1
        }

        return inconsistent_tracks

    def _link_dk_groups(self) -> Dict[str, List[str]]:
        """
        Link DK groups across clusters via track and embedding similarity.

        Process:
        1. Track-based linking: same track = same person
        2. Embedding-based linking: similar embeddings (threshold: dk_linking_threshold)

        Returns:
            dict: {dk_label: [linked_face_ids]}
        """
        logger.info("Linking DK groups across clusters...")

        dk_threshold = self.config['dk_linking_threshold']
        scene_proximity_enabled = self.config.get('scene_proximity_enabled', True)
        logger.info(f"  DK linking threshold: {dk_threshold:.3f}")
        if scene_proximity_enabled:
            bonus = self.config.get('scene_proximity_bonus', 0.03)
            logger.info(f"  Scene proximity bonus: +{bonus:.3f} for adjacent scenes")

        # Step 1: Build DK groups from annotations
        # IMPORTANT: DK labels (dk1, dk2, etc.) are cluster-specific!
        # We use (cluster_id, dk_label) as the unique key for each group
        # Track-based and embedding-based linking will merge them later if appropriate
        dk_groups: Dict[str, Set[str]] = defaultdict(set)

        # Add DK faces from annotations with cluster-specific keys
        for cluster_id, dk_label_dict in self.dk_by_cluster.items():
            for dk_label, face_ids in dk_label_dict.items():
                # Use compound key to make each DK group unique per cluster
                unique_key = f"cluster{cluster_id}_{dk_label}"
                dk_groups[unique_key].update(face_ids)

        initial_count = len(dk_groups)
        logger.info(f"  Initial DK groups (cluster-specific): {initial_count}")

        # Step 2: Track-based linking across clusters
        # If same track appears in different cluster-specific DK groups, merge them
        # Build mapping from track to cluster-specific DK group keys
        track_to_dk_groups: Dict[str, Set[str]] = defaultdict(set)

        for dk_group_key, face_ids in dk_groups.items():
            for face_id in face_ids:
                # Find which track this face belongs to
                for scene_id, faces in self.clustering_data.items():
                    for face_data in faces:
                        if face_data['unique_face_id'] == face_id:
                            for img_path in face_data.get('image_paths', []):
                                track_id = self.extract_track_from_filename(img_path)
                                if track_id:
                                    track_to_dk_groups[track_id].add(dk_group_key)
                                break
                            break

        # Merge DK groups that share tracks
        # If the same track has faces from multiple DK groups, they're the same person
        groups_to_merge: List[Set[str]] = []

        for track_id, group_keys in track_to_dk_groups.items():
            if len(group_keys) > 1:
                # This track has faces from multiple DK groups - merge them
                groups_to_merge.append(group_keys)

        # Perform merges
        track_merges = 0
        for merge_set in groups_to_merge:
            # Use the smallest key as the target
            target_key = min(merge_set)
            faces_to_add = set()

            for group_key in merge_set:
                if group_key != target_key:
                    faces_to_add.update(dk_groups.get(group_key, set()))

            dk_groups[target_key].update(faces_to_add)

            # Remove old keys
            for group_key in merge_set:
                if group_key != target_key and group_key in dk_groups:
                    del dk_groups[group_key]

            track_merges += len(merge_set) - 1

        if track_merges > 0:
            logger.info(f"  Track-based linking: merged {track_merges} DK groups")

        # Step 3: Embedding-based linking for DK groups with no track overlap
        # Compute centroids and scene info for each DK group
        # Use face weights to down-weight poor quality faces marked with @poor, @blurry, etc.
        dk_centroids: Dict[str, np.ndarray] = {}
        dk_scene_info: Dict[str, List[str]] = defaultdict(list)  # {dk_label: [scene_ids]}

        for dk_label, face_ids in dk_groups.items():
            weighted_embeddings = []
            weights = []
            for face_id in face_ids:
                # Get weight for this face (0.5 for poor quality, 1.0 for normal)
                weight = self.face_weights.get(face_id, 1.0)

                # Find embedding and scene for this face
                for scene_id, faces in self.clustering_data.items():
                    for face_data in faces:
                        if face_data['unique_face_id'] == face_id:
                            if face_data.get('embeddings'):
                                for emb in face_data['embeddings']:
                                    weighted_embeddings.append(np.array(emb) * weight)
                                    weights.append(weight)
                            dk_scene_info[dk_label].append(scene_id)
                            break

            if weighted_embeddings and weights:
                # Weighted average: sum(weighted_embeddings) / sum(weights)
                total_weight = sum(weights)
                dk_centroids[dk_label] = np.sum(weighted_embeddings, axis=0) / total_weight

        # Link DK groups via embedding similarity with scene proximity weighting
        dk_labels = list(dk_centroids.keys())
        merged = set()

        for i, label_a in enumerate(dk_labels):
            if label_a in merged:
                continue

            for label_b in dk_labels[i + 1:]:
                if label_b in merged:
                    continue

                centroid_a = dk_centroids[label_a]
                centroid_b = dk_centroids[label_b]

                # Compute cosine similarity with zero-norm guard
                norm_a = np.linalg.norm(centroid_a)
                norm_b = np.linalg.norm(centroid_b)
                if norm_a == 0 or norm_b == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(centroid_a, centroid_b) / (norm_a * norm_b)

                # Apply scene proximity weighting
                # Calculate minimum scene distance between the two DK groups
                scenes_a = dk_scene_info.get(label_a, [])
                scenes_b = dk_scene_info.get(label_b, [])

                if scenes_a and scenes_b:
                    # Find minimum distance between any scene in group A and any scene in group B
                    min_distance = min(
                        self._get_scene_distance(s_a, s_b)
                        for s_a in scenes_a
                        for s_b in scenes_b
                    )
                    weighted_similarity = self._compute_scene_weighted_similarity(similarity, min_distance)

                    if weighted_similarity > similarity:
                        logger.debug(
                            f"    Scene proximity bonus: {label_a} vs {label_b}, "
                            f"distance={min_distance}, {similarity:.4f} -> {weighted_similarity:.4f}"
                        )

                    similarity = weighted_similarity

                if similarity > dk_threshold:
                    # Merge the groups
                    dk_groups[label_a].update(dk_groups[label_b])
                    del dk_groups[label_b]
                    merged.add(label_b)

        # Update statistics
        embedding_merges = len(merged)
        if embedding_merges > 0:
            logger.info(f"  Embedding-based linking: merged {embedding_merges} DK groups")

        self.stats['dk_groups_linked'] = len(dk_groups)

        logger.info(f"  Linked into {len(dk_groups)} DK groups")

        return {k: list(v) for k, v in dk_groups.items()}

    def _build_constrained_graph(
        self,
        remaining_faces: Dict[int, List[str]],
        dk_groups: Dict[str, List[str]],
    ) -> nx.Graph:
        """
        Build similarity graph with must-link/cannot-link constraints.

        Args:
            remaining_faces: {cluster_id: [face_ids]} - non-ground-truth faces
            dk_groups: {dk_label: [face_ids]} - linked DK groups

        Returns:
            nx.Graph: Similarity graph with constraints applied
        """
        logger.info("Building constrained similarity graph...")

        threshold = self.config['similarity_threshold']
        graph = nx.Graph()

        # Collect all faces to include in graph
        all_faces = set()
        for faces in remaining_faces.values():
            all_faces.update(faces)
        for faces in dk_groups.values():
            all_faces.update(faces)

        # Remove ground truth faces - they're in locked clusters
        all_faces -= self.ground_truth_faces

        logger.info(f"  Building graph for {len(all_faces)} non-GT faces")

        # Get embeddings for all faces
        face_embeddings: Dict[str, np.ndarray] = {}
        for face_id in all_faces:
            for scene_id, faces in self.clustering_data.items():
                for face_data in faces:
                    if face_data['unique_face_id'] == face_id:
                        if face_data.get('embeddings'):
                            # Use average embedding
                            embs = [np.array(e) for e in face_data['embeddings']]
                            face_embeddings[face_id] = np.mean(embs, axis=0)
                        break

        # Add nodes
        for face_id in all_faces:
            graph.add_node(face_id)

        # Build edges based on similarity
        face_ids = list(face_embeddings.keys())
        must_link_count = 0

        for i, face_a in enumerate(face_ids):
            for face_b in face_ids[i + 1:]:
                emb_a = face_embeddings.get(face_a)
                emb_b = face_embeddings.get(face_b)

                if emb_a is None or emb_b is None:
                    continue

                # Compute cosine similarity with zero-norm guard
                norm_a = np.linalg.norm(emb_a)
                norm_b = np.linalg.norm(emb_b)
                if norm_a == 0 or norm_b == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(emb_a, emb_b) / (norm_a * norm_b)

                if similarity > threshold:
                    graph.add_edge(face_a, face_b, weight=similarity)

        logger.info(f"  Added {graph.number_of_edges()} similarity edges")

        # Apply must-link constraints
        # 1. Faces in same track → must-link
        track_to_faces: Dict[str, Set[str]] = defaultdict(set)
        for face_id in all_faces:
            for scene_id, faces in self.clustering_data.items():
                for face_data in faces:
                    if face_data['unique_face_id'] == face_id:
                        for img_path in face_data.get('image_paths', []):
                            track_id = self.extract_track_from_filename(img_path)
                            if track_id:
                                track_to_faces[track_id].add(face_id)
                                break
                        break

        # Merge faces in same track
        for track_id, track_faces in track_to_faces.items():
            if len(track_faces) > 1:
                face_list = list(track_faces)
                # Merge all faces in track into first face
                main_face = face_list[0]
                for other_face in face_list[1:]:
                    if other_face in graph.nodes:
                        # Merge by moving all edges from other_face to main_face
                        for neighbor in list(graph.neighbors(other_face)):
                            if neighbor != main_face and neighbor in graph.nodes:
                                weight = graph[other_face][neighbor].get('weight', 1.0)
                                if graph.has_edge(main_face, neighbor):
                                    # Keep max weight
                                    existing_weight = graph[main_face][neighbor].get('weight', 0)
                                    graph[main_face][neighbor]['weight'] = max(weight, existing_weight)
                                else:
                                    graph.add_edge(main_face, neighbor, weight=weight)
                        graph.remove_node(other_face)
                        # Update references
                        all_faces.discard(other_face)
                        must_link_count += 1

        # 2. Faces in same DK group → must-link
        for dk_label, dk_faces in dk_groups.items():
            dk_face_list = [f for f in dk_faces if f in graph.nodes]
            if len(dk_face_list) > 1:
                main_face = dk_face_list[0]
                for other_face in dk_face_list[1:]:
                    if other_face in graph.nodes:
                        for neighbor in list(graph.neighbors(other_face)):
                            if neighbor != main_face and neighbor in graph.nodes:
                                weight = graph[other_face][neighbor].get('weight', 1.0)
                                if graph.has_edge(main_face, neighbor):
                                    existing_weight = graph[main_face][neighbor].get('weight', 0)
                                    graph[main_face][neighbor]['weight'] = max(weight, existing_weight)
                                else:
                                    graph.add_edge(main_face, neighbor, weight=weight)
                        graph.remove_node(other_face)
                        all_faces.discard(other_face)
                        must_link_count += 1

        logger.info(f"  Applied {must_link_count} must-link constraints")

        # Apply cannot-link constraints
        # Faces with different main character labels → cannot-link
        cannot_link_count = 0
        locked_char_faces = {
            char: set(faces) for char, faces in self.locked_clusters.items()
        }

        for char_a, faces_a in locked_char_faces.items():
            for char_b, faces_b in locked_char_faces.items():
                if char_a >= char_b:
                    continue

                # Remove edges between different character groups
                for face_a in faces_a:
                    for face_b in faces_b:
                        if face_a in graph.nodes and face_b in graph.nodes:
                            if graph.has_edge(face_a, face_b):
                                graph.remove_edge(face_a, face_b)
                                cannot_link_count += 1

        logger.info(f"  Applied {cannot_link_count} cannot-link constraints")

        return graph

    def _run_constrained_chinese_whispers(
        self,
        graph: nx.Graph,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Run Chinese Whispers clustering on constrained graph.

        Args:
            graph: Similarity graph with constraints applied
            max_iterations: Maximum iterations (uses config if not specified)

        Returns:
            dict: {face_id: cluster_id}
        """
        if max_iterations is None:
            max_iterations = self.config['chinese_whispers_iterations']

        logger.info(f"Running Chinese Whispers (max {max_iterations} iterations)...")

        # Initialize each node with unique label
        labels = {node: i for i, node in enumerate(graph.nodes())}

        # Chinese Whispers algorithm
        for iteration in range(max_iterations):
            nodes = list(graph.nodes())
            changed = 0

            # Randomize order (optional, but helps with convergence)
            # np.random.shuffle(nodes)  # Not using to keep results deterministic

            for node in nodes:
                if not list(graph.neighbors(node)):
                    continue

                # Count labels among neighbors
                label_counts: Dict[int, float] = defaultdict(float)
                for neighbor in graph.neighbors(node):
                    neighbor_label = labels[neighbor]
                    # Weight by edge weight
                    weight = graph[node][neighbor].get('weight', 1.0)
                    label_counts[neighbor_label] += weight

                if label_counts:
                    # Choose most common label
                    best_label = max(label_counts.items(), key=lambda x: x[1])[0]
                    if labels[node] != best_label:
                        labels[node] = best_label
                        changed += 1

            if changed == 0:
                logger.info(f"  Converged after {iteration + 1} iterations")
                break

        # Count unique clusters
        unique_clusters = set(labels.values())
        logger.info(f"  Found {len(unique_clusters)} clusters from Chinese Whispers")

        return labels

    def _extract_named_characters(self) -> Set[str]:
        """
        Extract named characters from annotations.

        Named characters are labels that:
        - Are not in SKIP_LABELS
        - Don't start with 'dk'
        - Are not generic labels like 'other', 'guy', 'woman', 'kid', 'man', 'person'
        - Are not numeric labels

        Returns:
            set: Named character labels
        """
        generic_labels = {
            'other', 'guy', 'woman', 'kid', 'man', 'person', 'male', 'female',
            'boy', 'girl', 'unknown', 'not_human', 'background', 'unclear',
            'junk', 'not face', 'not clear', 'dk'
        }

        named_characters = set()

        for cluster_key, cluster_info in self.annotations['cluster_annotations'].items():
            main_label = cluster_info['label'].lower()

            # Check if main label is a named character
            if main_label not in generic_labels and not main_label.startswith('dk'):
                # Check if it's not purely numeric (like "guy10" -> treat as generic)
                if not main_label.replace('guy', '').replace('woman', '').replace('man', '').replace('kid', '').isdigit():
                    named_characters.add(main_label)

            # Check outliers for named characters
            for outlier in cluster_info.get('outliers', []):
                outlier_label = outlier['label'].lower()
                if outlier_label not in generic_labels and not outlier_label.startswith('dk'):
                    named_characters.add(outlier_label)

        return named_characters

    def _validate_and_merge(
        self,
        cw_labels: Dict[str, int],
        dk_groups: Dict[str, List[str]],
    ) -> dict:
        """
        Validate track integrity and perform final merging.

        Uses STRING cluster IDs:
        - Named characters: "cluster-{character_name}" (e.g., "cluster-rachel")
        - Others: "cluster-{number:03d}" (e.g., "cluster-001", "cluster-002")

        Args:
            cw_labels: {face_id: cluster_id} from Chinese Whispers
            dk_groups: {dk_label: [face_ids]}

        Returns:
            dict: Refined clustering data with string cluster IDs
        """
        logger.info("Validating track integrity and performing final merge...")

        # Extract named characters from annotations
        named_characters = self._extract_named_characters()
        logger.info(f"  Found {len(named_characters)} named characters: {sorted(named_characters)}")

        # Start with locked clusters (ground truth) - use named cluster IDs
        char_to_cluster_id: Dict[str, str] = {}
        next_numeric_id = 1

        for char in sorted(self.locked_clusters.keys()):
            if self.locked_clusters[char]:
                cluster_id = f"cluster-{char}"
                char_to_cluster_id[char] = cluster_id
                logger.info(f"  Cluster {cluster_id}: {char} (locked, {len(self.locked_clusters[char])} faces)")

        # Assign Chinese Whisper clusters
        # Map CW cluster IDs to new cluster IDs
        cw_to_new_cluster: Dict[int, str] = {}
        face_to_cluster: Dict[str, str] = {}

        # First, assign all ground truth faces to their character clusters
        for char, faces in self.locked_clusters.items():
            cluster_id = char_to_cluster_id.get(char)
            if cluster_id:
                for face_id in faces:
                    face_to_cluster[face_id] = cluster_id

        # Track-based validation: ensure all faces from same track are in same cluster
        track_to_faces: Dict[str, Set[str]] = defaultdict(set)
        for face_id in cw_labels.keys():
            for scene_id, faces in self.clustering_data.items():
                for face_data in faces:
                    if face_data['unique_face_id'] == face_id:
                        for img_path in face_data.get('image_paths', []):
                            track_id = self.extract_track_from_filename(img_path)
                            if track_id:
                                track_to_faces[track_id].add(face_id)
                                break
                        break

        # Assign DK groups to clusters
        for dk_label, dk_faces in dk_groups.items():
            # Find which CW cluster most of these faces belong to
            cw_cluster_counts: Dict[int, int] = defaultdict(int)
            for face_id in dk_faces:
                if face_id in cw_labels:
                    cw_cluster_counts[cw_labels[face_id]] += 1

            if cw_cluster_counts:
                best_cw_cluster = max(cw_cluster_counts.items(), key=lambda x: x[1])[0]

                # Assign new cluster ID or use existing
                if best_cw_cluster not in cw_to_new_cluster:
                    cluster_id = f"cluster-{next_numeric_id:03d}"
                    cw_to_new_cluster[best_cw_cluster] = cluster_id
                    next_numeric_id += 1

                cluster_id = cw_to_new_cluster[best_cw_cluster]

                # Assign all faces in this DK group to the same cluster
                for face_id in dk_faces:
                    face_to_cluster[face_id] = cluster_id

        # Assign remaining faces from Chinese Whispers
        for face_id, cw_cluster in cw_labels.items():
            if face_id not in face_to_cluster:
                if cw_cluster not in cw_to_new_cluster:
                    cluster_id = f"cluster-{next_numeric_id:03d}"
                    cw_to_new_cluster[cw_cluster] = cluster_id
                    next_numeric_id += 1
                face_to_cluster[face_id] = cw_to_new_cluster[cw_cluster]

        # Validate track integrity
        violations = 0
        for track_id, track_faces in track_to_faces.items():
            track_clusters = set()
            for face_id in track_faces:
                if face_id in face_to_cluster:
                    track_clusters.add(face_to_cluster[face_id])

            if len(track_clusters) > 1:
                # Violation - faces from same track in different clusters
                violations += 1
                # Resolve by preferring cluster with ground truth
                gt_cluster = None
                for face_id in track_faces:
                    for char, cluster_id in char_to_cluster_id.items():
                        if cluster_id == face_to_cluster.get(face_id):
                            gt_cluster = cluster_id
                            break
                    if gt_cluster is not None:
                        break

                if gt_cluster is not None:
                    # Move all faces in track to GT cluster
                    for face_id in track_faces:
                        face_to_cluster[face_id] = gt_cluster

        if violations > 0:
            logger.warning(f"  Fixed {violations} track integrity violations")
            self.stats['constraint_violations'] = violations

        # Rebuild clustering data
        refined_clustering = {}

        for scene_id, faces in self.clustering_data.items():
            refined_clustering[scene_id] = []

            for face_data in faces:
                updated_face = face_data.copy()
                face_id = face_data['unique_face_id']

                if face_id in face_to_cluster:
                    updated_face['cluster_id'] = face_to_cluster[face_id]
                else:
                    # Face not in mapping - assign to unassigned/unknown cluster
                    updated_face['cluster_id'] = 'cluster-unassigned'

                refined_clustering[scene_id].append(updated_face)

        # Print cluster distribution
        cluster_distribution: Dict[str, int] = defaultdict(int)
        for cluster_id in face_to_cluster.values():
            cluster_distribution[cluster_id] += 1

        # Log final clusters grouped by type
        named_clusters = {k: v for k, v in cluster_distribution.items() if any(k == f"cluster-{c}" for c in named_characters)}
        numeric_clusters = {k: v for k, v in cluster_distribution.items() if k not in named_clusters}

        logger.info(f"  Final cluster count: {len(cluster_distribution)}")
        logger.info(f"    Named clusters: {len(named_clusters)}")
        for cluster_id, count in sorted(named_clusters.items()):
            char = cluster_id.replace('cluster-', '')
            logger.info(f"      {cluster_id}: {count} faces")
        logger.info(f"    Other clusters: {len(numeric_clusters)}")
        for cluster_id, count in sorted(numeric_clusters.items()):
            logger.info(f"      {cluster_id}: {count} faces")

        return refined_clustering

    def _build_track_to_label_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Build mapping from track_id to label with source information.

        Scans all annotated faces (main cluster images + outliers) to build:
        {track_id: {'label': char, 'sources': [(cluster_id, type), ...]}}

        Returns:
            dict: Track to label mapping with source information
        """
        track_info = {}  # {track_id: {'label': {char: count}, 'sources': [...], 'main_labels': set}}

        for cluster_key, cluster_info in self.annotations['cluster_annotations'].items():
            label = cluster_info['label'].lower()

            if label in self.SKIP_LABELS:
                continue

            # Extract numeric cluster ID
            if '-' in cluster_key:
                numeric_id = int(cluster_key.split('-')[-1])
            else:
                numeric_id = int(cluster_key)

            # Process main cluster images
            for img_path in cluster_info.get('image_paths', []):
                filename = os.path.basename(img_path)
                norm_key = self._normalize_filename_key(filename)
                if norm_key not in self.image_mapping_norm:
                    continue

                track_id = self.extract_track_from_filename(filename)
                if not track_id:
                    continue

                if track_id not in track_info:
                    track_info[track_id] = {
                        'label': {label: 1},
                        'sources': [(numeric_id, 'main')],
                        'main_labels': set()
                    }

                if label not in track_info[track_id]['label']:
                    track_info[track_id]['label'][label] = 0
                track_info[track_id]['label'][label] += 1
                track_info[track_id]['sources'].append((numeric_id, 'main'))
                track_info[track_id]['main_labels'].add(label)

            # Process outliers
            for outlier in cluster_info.get('outliers', []):
                outlier_label = outlier['label'].lower()
                if outlier_label in self.SKIP_LABELS:
                    continue

                img_path = outlier['image_path']
                filename = os.path.basename(img_path)
                norm_key = self._normalize_filename_key(filename)

                if norm_key not in self.image_mapping_norm:
                    continue

                track_id = self.extract_track_from_filename(filename)
                if not track_id:
                    continue

                if track_id not in track_info:
                    track_info[track_id] = {
                        'label': {outlier_label: 1},
                        'sources': [(numeric_id, 'outlier')],
                        'main_labels': set()
                    }

                if outlier_label not in track_info[track_id]['label']:
                    track_info[track_id]['label'][outlier_label] = 0
                track_info[track_id]['label'][outlier_label] += 1
                track_info[track_id]['sources'].append((numeric_id, 'outlier'))

        self.stats['tracks_mapped'] = len(track_info)

        # Count tracks with conflicts (multiple labels)
        for track_id, info in track_info.items():
            if len(info['label']) > 1:
                self.stats['tracks_with_conflicts'] += 1

        return track_info

    def _resolve_track_conflicts(self, track_info: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Resolve conflicts where a track has multiple labels.

        Rule c: Prefer cluster main label over outlier label
        Rule a: Use majority vote if rule c doesn't resolve

        Args:
            track_info: Track info from _build_track_to_label_mapping

        Returns:
            dict: Cleaned {track_id: character_label}
        """
        resolved = {}

        for track_id, info in track_info.items():
            labels = info['label']
            main_labels = info['main_labels']

            if len(labels) == 1:
                # No conflict
                resolved[track_id] = list(labels.keys())[0]
            else:
                # Conflict exists - apply resolution rules
                self.stats['tracks_with_conflicts'] += 1

                # Rule c: Prefer cluster main label over outlier label
                if main_labels:
                    # Use the first main label (should be only one if annotations are consistent)
                    resolved[track_id] = list(main_labels)[0]
                    self.conflicts_resolved += 1
                else:
                    # Rule a: Use majority vote
                    resolved[track_id] = max(labels.items(), key=lambda x: x[1])[0]
                    self.conflicts_resolved += 1

        self.track_to_label = resolved
        return resolved

    def _propagate_labels_to_unannotated(self, track_to_label: Dict[str, str], dry_run: bool = False) -> Dict[str, str]:
        """
        Propagate labels to unannotated outliers via track matching.

        For each cluster:
          For each unannotated outlier:
            If track_id exists in track_to_label → assign label

        Args:
            track_to_label: Resolved track to label mapping
            dry_run: If True, don't modify state, just return what would be propagated

        Returns:
            dict: {face_id: propagated_label}
        """
        propagated = {}

        for cluster_key, cluster_info in self.annotations['cluster_annotations'].items():
            # Extract numeric cluster ID
            if '-' in cluster_key:
                cluster_id = int(cluster_key.split('-')[-1])
            else:
                cluster_id = int(cluster_key)

            # Check unannotated outliers
            # An outlier is "unannotated" if it doesn't have a label or label is empty
            for outlier in cluster_info.get('outliers', []):
                outlier_label = outlier.get('label', '').lower()
                img_path = outlier['image_path']
                filename = os.path.basename(img_path)
                norm_key = self._normalize_filename_key(filename)

                if norm_key not in self.image_mapping_norm:
                    continue

                # Skip if already annotated with a valid label
                if outlier_label and outlier_label not in ['dk', ''] + self.SKIP_LABELS:
                    continue

                track_id = self.extract_track_from_filename(filename)
                if not track_id:
                    continue

                if track_id in track_to_label:
                    face_id = self.image_mapping_norm[norm_key]['unique_face_id']
                    propagated[face_id] = track_to_label[track_id]

        if not dry_run:
            self.propagated_labels = propagated
            self.stats['faces_propagated'] = len(propagated)

        return propagated

    def _match_unannotated_clusters(self, dry_run: bool = False) -> Dict[int, str]:
        """
        Match entirely unannotated clusters via embedding similarity.

        Finds clusters with NO annotations and matches them to annotated clusters.

        Args:
            dry_run: If True, don't modify state

        Returns:
            dict: {cluster_id: assigned_label}
        """
        threshold = self.config['unannotated_cluster_threshold']

        # Find unannotated clusters (no main label, no annotated outliers)
        unannotated_clusters = set()
        annotated_clusters = set()

        for cluster_key, cluster_info in self.annotations['cluster_annotations'].items():
            if '-' in cluster_key:
                cluster_id = int(cluster_key.split('-')[-1])
            else:
                cluster_id = int(cluster_key)

            main_label = cluster_info['label'].lower()

            # Check if cluster has any valid annotation
            has_annotation = (
                main_label not in ['', 'dk'] + self.SKIP_LABELS or
                any(o.get('label', '').lower() not in ['', 'dk'] + self.SKIP_LABELS
                    for o in cluster_info.get('outliers', []))
            )

            if has_annotation:
                annotated_clusters.add(cluster_id)
            else:
                unannotated_clusters.add(cluster_id)

        # Build cluster centroids for annotated clusters
        cluster_centroids = {}  # {cluster_id: {'embedding': centroid, 'label': label}}
        cluster_labels = {}  # {cluster_id: label}

        for cluster_key, cluster_info in self.annotations['cluster_annotations'].items():
            if '-' in cluster_key:
                cluster_id = int(cluster_key.split('-')[-1])
            else:
                cluster_id = int(cluster_key)

            if cluster_id not in annotated_clusters:
                continue

            label = cluster_info['label'].lower()
            if label in self.SKIP_LABELS:
                continue

            cluster_labels[cluster_id] = label

            # Get all embeddings for this cluster
            embeddings = []

            # Main cluster images
            for img_path in cluster_info.get('image_paths', []):
                filename = os.path.basename(img_path)
                norm_key = self._normalize_filename_key(filename)
                if norm_key in self.image_mapping_norm:
                    embeddings.append(self.image_mapping_norm[norm_key]['embedding'])

            # Outliers with same label
            for outlier in cluster_info.get('outliers', []):
                if outlier['label'].lower() == label:
                    filename = os.path.basename(outlier['image_path'])
                    norm_key = self._normalize_filename_key(filename)
                    if norm_key in self.image_mapping_norm:
                        embeddings.append(self.image_mapping_norm[norm_key]['embedding'])

            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                cluster_centroids[cluster_id] = {'embedding': centroid, 'label': label}

        # Match unannotated clusters
        matches = {}

        for cluster_id in unannotated_clusters:
            # Get centroid for unannotated cluster
            embeddings = []

            # Find faces in this cluster from clustering_data
            for scene_id, faces in self.clustering_data.items():
                for face_data in faces:
                    if face_data.get('cluster_id') == cluster_id:
                        for emb in face_data.get('embeddings', []):
                            embeddings.append(np.array(emb))

            if not embeddings:
                continue

            unannotated_centroid = np.mean(embeddings, axis=0)

            # Find most similar annotated cluster
            best_match = None
            best_similarity = -1

            for ann_cluster_id, ann_data in cluster_centroids.items():
                # Compute cosine similarity with zero-norm guard
                norm_a = np.linalg.norm(unannotated_centroid)
                norm_b = np.linalg.norm(ann_data['embedding'])
                if norm_a == 0 or norm_b == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(unannotated_centroid, ann_data['embedding']) / (norm_a * norm_b)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = ann_cluster_id

            if best_match and best_similarity > threshold:
                matches[cluster_id] = cluster_centroids[best_match]['label']

        if not dry_run:
            self.cluster_matches = matches
            self.stats['clusters_matched'] = len(matches)

        return matches

    def _handle_dk_propagation(self, track_to_label: Dict[str, str], dry_run: bool = False) -> Dict[str, str]:
        """
        Convert dk faces to characters when track evidence is strong.

        Args:
            track_to_label: Resolved track to label mapping
            dry_run: If True, don't modify state

        Returns:
            dict: {face_id: character_label} for dk faces that were converted
        """
        if not self.config.get('dk_propagation_enabled', True):
            return {}

        dk_converted = {}

        for cluster_key, cluster_info in self.annotations['cluster_annotations'].items():
            main_label = cluster_info['label'].lower()

            # Process outliers with dk label
            for outlier in cluster_info.get('outliers', []):
                outlier_label = outlier['label'].lower()

                if not outlier_label.startswith('dk'):
                    continue

                img_path = outlier['image_path']
                filename = os.path.basename(img_path)
                norm_key = self._normalize_filename_key(filename)

                if norm_key not in self.image_mapping_norm:
                    continue

                track_id = self.extract_track_from_filename(filename)
                if not track_id:
                    continue

                if track_id in track_to_label:
                    face_id = self.image_mapping_norm[norm_key]['unique_face_id']
                    character_label = track_to_label[track_id]

                    # Don't convert to dk
                    if character_label != 'dk':
                        dk_converted[face_id] = character_label

        if not dry_run:
            self.dk_propagated = dk_converted
            self.stats['dk_converted'] = len(dk_converted)

        return dk_converted

    def extract_constraints(self, propagated_labels: Optional[Dict[str, str]] = None,
                           dk_converted: Optional[Dict[str, str]] = None,
                           cluster_matches: Optional[Dict[int, str]] = None) -> Dict[str, List[Tuple[str, str]]]:
        """
        Convert annotations into pairwise must-link/cannot-link constraints.

        Enhanced to include propagated labels and dk conversions.

        Args:
            propagated_labels: Faces that got labels via track matching
            dk_converted: DK faces that were converted to characters
            cluster_matches: Unannotated clusters that got matched

        Returns:
            dict: {'must_link': [(face_a, face_b), ...], 'cannot_link': [(face_a, face_b), ...]}
        """
        propagated_labels = propagated_labels or {}
        dk_converted = dk_converted or {}
        cluster_matches = cluster_matches or {}

        must_link = []
        cannot_link = []

        # Group faces by assigned label
        label_to_faces = defaultdict(list)
        track_to_dk_faces = defaultdict(list)  # track_id -> [face_ids]
        unmatched_images = []

        for cluster_key, cluster_info in self.annotations['cluster_annotations'].items():
            label = cluster_info['label'].lower()

            if label in self.SKIP_LABELS:
                continue

            # Extract numeric cluster ID
            if '-' in cluster_key:
                cluster_id = int(cluster_key.split('-')[-1])
            else:
                cluster_id = int(cluster_key)

            # Check if this cluster was matched via similarity
            effective_label = cluster_matches.get(cluster_id, label)

            # Add main cluster images
            for img_path in cluster_info.get('image_paths', []):
                filename = os.path.basename(img_path)
                norm_key = self._normalize_filename_key(filename)
                if norm_key in self.image_mapping_norm:
                    face_id = self.image_mapping_norm[norm_key]['unique_face_id']

                    if effective_label == 'dk' or effective_label.startswith('dk'):
                        track_id = self.extract_track_from_filename(filename)
                        if track_id:
                            track_to_dk_faces[track_id].append(face_id)
                    else:
                        label_to_faces[effective_label].append(face_id)
                else:
                    unmatched_images.append(filename)

            # Add outliers
            for outlier in cluster_info.get('outliers', []):
                outlier_label = outlier['label'].lower()
                img_path = outlier['image_path']
                filename = os.path.basename(img_path)
                norm_key = self._normalize_filename_key(filename)

                if norm_key not in self.image_mapping_norm:
                    continue

                face_id = self.image_mapping_norm[norm_key]['unique_face_id']

                # Check if this face was propagated or converted
                if face_id in propagated_labels:
                    final_label = propagated_labels[face_id]
                    label_to_faces[final_label].append(face_id)
                elif face_id in dk_converted:
                    final_label = dk_converted[face_id]
                    label_to_faces[final_label].append(face_id)
                elif outlier_label in self.SKIP_LABELS:
                    continue
                elif outlier_label == 'dk' or outlier_label.startswith('dk'):
                    track_id = self.extract_track_from_filename(filename)
                    if track_id:
                        track_to_dk_faces[track_id].append(face_id)
                else:
                    label_to_faces[outlier_label].append(face_id)

        # Generate must-link constraints for named characters
        main_char_constraints = 0
        for label, face_ids in label_to_faces.items():
            unique_faces = list(set(face_ids))
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    must_link.append((unique_faces[i], unique_faces[j]))
                    main_char_constraints += 1

        # Generate must-link constraints for dk faces (within-track only)
        dk_constraints = 0
        for track_key, face_ids in track_to_dk_faces.items():
            unique_faces = list(set(face_ids))
            if len(unique_faces) > 1:
                for i in range(len(unique_faces)):
                    for j in range(i + 1, len(unique_faces)):
                        must_link.append((unique_faces[i], unique_faces[j]))
                        dk_constraints += 1

        logger.info(f"Generated {main_char_constraints} must-link constraints for named characters")
        logger.info(f"Generated {dk_constraints} must-link constraints for dk faces")

        # Generate cannot-link constraints (different labels)
        valid_labels = list(label_to_faces.keys())

        for i in range(len(valid_labels)):
            for j in range(i + 1, len(valid_labels)):
                faces_a = label_to_faces[valid_labels[i]][:self.config['max_cannot_link_samples']]
                faces_b = label_to_faces[valid_labels[j]][:self.config['max_cannot_link_samples']]

                for face_a in faces_a:
                    for face_b in faces_b:
                        cannot_link.append((face_a, face_b))

        logger.info(f"Generated {len(cannot_link)} cannot-link constraints")

        return {'must_link': must_link, 'cannot_link': cannot_link}

    @staticmethod
    def find_root_cluster(cluster_id: int, cluster_mapping: Dict[int, int]) -> int:
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

    def apply_constraints_to_clustering(self, constraints: Dict[str, List[Tuple[str, str]]],
                                       dry_run: bool = False) -> Tuple[dict, dict]:
        """
        Apply must-link/cannot-link constraints to refine clusters.

        Args:
            constraints: {'must_link': [...], 'cannot_link': [...]}
            dry_run: If True, return what would happen without modifying

        Returns:
            tuple: (refined_clustering, statistics)
        """
        # Build face_id to cluster mapping
        face_to_cluster = {}
        cluster_to_faces = defaultdict(list)

        for scene_id, faces in self.clustering_data.items():
            for face_data in faces:
                unique_face_id = face_data['unique_face_id']
                cluster_id = face_data.get('cluster_id')

                face_to_cluster[unique_face_id] = cluster_id
                cluster_to_faces[cluster_id].append(face_data)

        initial_cluster_count = len(cluster_to_faces)
        logger.info(f"Initial cluster count: {initial_cluster_count}")

        if dry_run:
            # Return what would happen
            return self.clustering_data, {
                'initial_clusters': initial_cluster_count,
                'final_clusters': initial_cluster_count,
                'merges_performed': 0,
                'splits_performed': 0,
            }

        # Apply must-link constraints (merge clusters)
        cluster_mapping = {}
        merges_performed = 0

        for face_a, face_b in constraints['must_link']:
            if face_a not in face_to_cluster or face_b not in face_to_cluster:
                continue

            cluster_a = self.find_root_cluster(face_to_cluster[face_a], cluster_mapping)
            cluster_b = self.find_root_cluster(face_to_cluster[face_b], cluster_mapping)

            if cluster_a != cluster_b:
                target_cluster = min(cluster_a, cluster_b)
                source_cluster = max(cluster_a, cluster_b)

                cluster_mapping[source_cluster] = target_cluster
                merges_performed += 1

                for face_data in cluster_to_faces[source_cluster]:
                    face_to_cluster[face_data['unique_face_id']] = target_cluster

        logger.info(f"Performed {merges_performed} cluster merges")

        # Apply cannot-link constraints (split clusters)
        violations = []
        for face_a, face_b in constraints['cannot_link']:
            if face_a not in face_to_cluster or face_b not in face_to_cluster:
                continue

            cluster_a = self.find_root_cluster(face_to_cluster[face_a], cluster_mapping)
            cluster_b = self.find_root_cluster(face_to_cluster[face_b], cluster_mapping)

            if cluster_a == cluster_b:
                violations.append((face_a, face_b, cluster_a))

        splits_performed = 0
        if violations:
            logger.warning(f"Found {len(violations)} cannot-link violations")

            all_cluster_ids = set(cluster_to_faces.keys()) | set(cluster_mapping.values())
            next_cluster_id = max(all_cluster_ids) + 1 if all_cluster_ids else 0

            violations_by_cluster = defaultdict(list)
            for face_a, face_b, cluster_id in violations:
                violations_by_cluster[cluster_id].append((face_a, face_b))

            for cluster_id, violation_pairs in violations_by_cluster.items():
                faces_to_move = set()
                for _, face_b in violation_pairs:
                    if face_to_cluster.get(face_b) == cluster_id:
                        faces_to_move.add(face_b)

                if not faces_to_move:
                    continue

                new_cluster_id = next_cluster_id
                for face_id in faces_to_move:
                    face_to_cluster[face_id] = new_cluster_id

                next_cluster_id += 1
                splits_performed += 1

            logger.info(f"Performed {splits_performed} cluster splits")

        # Rebuild clustering data with merged/split clusters
        refined_clustering = {}

        for scene_id, faces in self.clustering_data.items():
            refined_clustering[scene_id] = []

            for face_data in faces:
                updated_face = face_data.copy()
                old_cluster = face_data.get('cluster_id')

                new_cluster = self.find_root_cluster(old_cluster, cluster_mapping)

                unique_face_id = face_data['unique_face_id']
                if unique_face_id in face_to_cluster:
                    new_cluster = face_to_cluster[unique_face_id]

                updated_face['cluster_id'] = new_cluster
                refined_clustering[scene_id].append(updated_face)

        # Calculate final statistics
        final_clusters = set()
        for scene_id, faces in refined_clustering.items():
            for face_data in faces:
                final_clusters.add(face_data['cluster_id'])

        statistics = {
            'initial_clusters': initial_cluster_count,
            'final_clusters': len(final_clusters),
            'merges_performed': merges_performed,
            'splits_performed': splits_performed,
            'must_link_constraints': len(constraints['must_link']),
            'cannot_link_constraints': len(constraints['cannot_link']),
            'cannot_link_violations': len(violations),
        }

        return refined_clustering, statistics

    def refine(self, dry_run: bool = False, use_new_workflow: bool = True) -> Dict[str, Any]:
        """
        Main refinement orchestration.

        New workflow (when use_new_workflow=True):
        1. Extract ground truth from annotations
        2. Separate outliers from parent clusters
        3. Link DK groups across clusters
        4. Build constrained similarity graph
        5. Run Chinese Whispers with constraints
        6. Validate track integrity and merge

        Old workflow (when use_new_workflow=False):
        1. Build track-to-label mapping
        2. Resolve conflicts
        3. Propagate labels to unannotated
        4. Match unannotated clusters
        5. Handle DK propagation
        6. Extract and apply constraints

        Args:
            dry_run: If True, return statistics without modifying data
            use_new_workflow: If True, use new ground-truth-based workflow

        Returns:
            dict: {
                'refined_clustering': dict or None if dry_run,
                'statistics': dict,
                'propagated_labels': dict,
                'cluster_matches': dict,
                'dk_converted': dict,
            }
        """
        if use_new_workflow:
            return self._refine_new_workflow(dry_run)
        else:
            return self._refine_old_workflow(dry_run)

    def _refine_new_workflow(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        New refinement workflow with ground truth extraction and re-clustering.

        Args:
            dry_run: If True, return statistics without modifying data

        Returns:
            dict: Result with refined_clustering and statistics
        """
        logger.info("=" * 70)
        logger.info("CLUSTER REFINEMENT - NEW WORKFLOW")
        logger.info("=" * 70)

        # Phase 1: Extract ground truth
        logger.info("Phase 1: Extracting ground truth from annotations...")
        gt_data = self._extract_ground_truth()

        # Phase 2: Separate outliers
        logger.info("Phase 2: Separating outliers from parent clusters...")
        remaining_faces, outlier_groups = self._separate_outliers()

        # Phase 3: Link DK groups
        logger.info("Phase 3: Linking DK groups across clusters...")
        dk_groups = self._link_dk_groups()

        if dry_run:
            logger.info("DRY RUN - No data modified")
            return {
                'refined_clustering': None,
                'statistics': {**self.stats, 'dry_run': True},
                'propagated_labels': {},
                'cluster_matches': {},
                'dk_converted': {},
            }

        # Phase 4: Build constrained graph
        logger.info("Phase 4: Building constrained similarity graph...")
        graph = self._build_constrained_graph(remaining_faces, dk_groups)

        # Phase 5: Run Chinese Whispers
        logger.info("Phase 5: Running Chinese Whispers with constraints...")
        cw_labels = self._run_constrained_chinese_whispers(graph)

        # Phase 6: Validate and merge
        logger.info("Phase 6: Validating and merging...")
        refined_clustering = self._validate_and_merge(cw_labels, dk_groups)

        # Calculate initial cluster count for comparison
        initial_clusters = set()
        for scene_id, faces in self.clustering_data.items():
            for face_data in faces:
                initial_clusters.add(face_data.get('cluster_id'))

        # Calculate final cluster count
        final_clusters = set()
        for scene_id, faces in refined_clustering.items():
            for face_data in faces:
                final_clusters.add(face_data.get('cluster_id'))

        # Compile statistics
        statistics = {
            **self.stats,
            'initial_clusters': len(initial_clusters),
            'final_clusters': len(final_clusters),
            'cluster_change': len(final_clusters) - len(initial_clusters),
        }

        logger.info("=" * 70)
        logger.info("REFINEMENT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Ground truth tracks:     {statistics['ground_truth_tracks']}")
        logger.info(f"  Ground truth faces:      {statistics['ground_truth_faces']}")
        logger.info(f"  DK groups linked:        {statistics['dk_groups_linked']}")
        logger.info(f"  Constraint violations:   {statistics['constraint_violations']}")
        logger.info(f"  Initial clusters:        {statistics['initial_clusters']}")
        logger.info(f"  Final clusters:          {statistics['final_clusters']}")
        logger.info(f"  Cluster change:          {statistics['cluster_change']:+d}")
        logger.info("=" * 70)

        return {
            'refined_clustering': refined_clustering,
            'statistics': statistics,
            'propagated_labels': {},
            'cluster_matches': {},
            'dk_converted': {},
        }

    def _refine_old_workflow(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Original refinement workflow (for backward compatibility).

        Args:
            dry_run: If True, return statistics without modifying data

        Returns:
            dict: Result with refined_clustering and statistics
        """
        logger.info("=" * 70)
        logger.info("CLUSTER REFINEMENT - OLD WORKFLOW")
        logger.info("=" * 70)

        # Step 1: Build track-to-label mapping
        logger.info("Step 1: Building track-to-label mapping...")
        track_info = self._build_track_to_label_mapping()
        logger.info(f"  Mapped {self.stats['tracks_mapped']} tracks")
        logger.info(f"  Found {self.stats['tracks_with_conflicts']} tracks with conflicts")

        # Step 2: Resolve conflicts
        logger.info("Step 2: Resolving track conflicts...")
        track_to_label = self._resolve_track_conflicts(track_info)
        logger.info(f"  Resolved {self.conflicts_resolved} conflicts")
        logger.info(f"  Final: {len(track_to_label)} tracks with labels")

        # Step 3: Propagate to unannotated outliers
        logger.info("Step 3: Propagating labels to unannotated outliers...")
        propagated = self._propagate_labels_to_unannotated(track_to_label, dry_run)
        logger.info(f"  Propagated {len(propagated)} faces via track matching")

        # Step 4: Match unannotated clusters
        logger.info("Step 4: Matching unannotated clusters...")
        matches = self._match_unannotated_clusters(dry_run)
        logger.info(f"  Matched {len(matches)} unannotated clusters")

        # Step 5: Handle dk propagation
        logger.info("Step 5: Converting dk faces via track matching...")
        dk_converted = self._handle_dk_propagation(track_to_label, dry_run)
        logger.info(f"  Converted {len(dk_converted)} dk faces to characters")

        # Step 6: Extract constraints
        logger.info("Step 6: Extracting constraints...")
        constraints = self.extract_constraints(propagated, dk_converted, matches)

        # Step 7: Apply constraints
        logger.info("Step 7: Applying constraints...")
        refined_clustering, constraint_stats = self.apply_constraints_to_clustering(
            constraints, dry_run
        )

        # Compile statistics
        statistics = {
            **constraint_stats,
            'tracks_mapped': self.stats['tracks_mapped'],
            'tracks_with_conflicts': self.stats['tracks_with_conflicts'],
            'conflicts_resolved': self.conflicts_resolved,
            'faces_propagated': len(propagated),
            'clusters_matched': len(matches),
            'dk_converted': len(dk_converted),
        }

        logger.info("=" * 70)
        logger.info("REFINEMENT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Tracks mapped:          {statistics['tracks_mapped']}")
        logger.info(f"  Faces propagated:       {statistics['faces_propagated']}")
        logger.info(f"  Clusters matched:       {statistics['clusters_matched']}")
        logger.info(f"  DK faces converted:     {statistics['dk_converted']}")
        logger.info(f"  Cluster change:         {statistics['final_clusters'] - statistics['initial_clusters']:+d}")
        logger.info("=" * 70)

        if dry_run:
            logger.info("DRY RUN - No data modified")

        return {
            'refined_clustering': refined_clustering if not dry_run else None,
            'statistics': statistics,
            'propagated_labels': propagated,
            'cluster_matches': matches,
            'dk_converted': dk_converted,
        }
