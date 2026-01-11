"""
Face clustering module for character tracking.

This module provides:
- FaceEmbedder: Extracts face embeddings using InsightFace buffalo_l model
- FaceClusterer: Clusters faces using Chinese Whispers algorithm

Note: This module now exclusively uses the InsightFace buffalo_l model
for better accuracy and performance.
"""

import os
import cv2
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.spatial.distance import cosine, cdist


class FaceEmbedder:
    """
    Extracts face embeddings using InsightFace buffalo_l model.

    The buffalo_l model uses ArcFace loss and produces 512-dimensional embeddings.
    Input images should be 112x112 BGR format (uint8, 0-255).
    """

    def __init__(self, device=None):
        """
        Initialize FaceEmbedder with InsightFace buffalo_l model.

        Args:
            device: 'cuda' for GPU, 'cpu' for CPU (auto-detected if None)
        """
        # Determine device
        if device is None:
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = device

        # Import insightface and initialize model
        import insightface
        from insightface.app import FaceAnalysis

        # Determine available providers with graceful fallback
        if self.device == 'cuda':
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    print("Warning: CUDA requested but CUDAExecutionProvider not available. "
                          "Falling back to CPU. Install onnxruntime-gpu for GPU acceleration.")
                    providers = ['CPUExecutionProvider']
            except ImportError:
                providers = ['CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Initialize FaceAnalysis with buffalo_l
        self.model = FaceAnalysis(name='buffalo_l', providers=providers)
        self.model.prepare(
            ctx_id=0 if self.device == 'cuda' and 'CUDAExecutionProvider' in providers else -1,
            det_size=(640, 640)
        )

        # Store recognition model reference for direct embedding extraction
        # (bypasses face detection which fails on pre-cropped faces)
        self.rec_model = self.model.models.get('recognition')

        # buffalo_l configuration
        self.input_size = (112, 112)
        self.embedding_dim = 512

    def load_image(self, image_path):
        """Load an image from disk."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        return self.preprocess_face(image)

    def preprocess_face(self, face_image):
        """
        Preprocess face image for embedding extraction.

        Args:
            face_image: BGR image from cv2.imread

        Returns:
            Preprocessed image (112x112 BGR uint8)
        """
        return cv2.resize(face_image, self.input_size)

    def get_face_embeddings(self, selected_frames, image_dir):
        """
        Get embeddings for each cropped face image (sequential processing).

        Args:
            selected_frames: Dictionary of selected frames per scene
            image_dir: Directory containing face images

        Returns:
            List of face embeddings with metadata
        """
        face_embeddings = []

        with tqdm(total=len(selected_frames), desc="Extracting Face Embeddings", unit="face") as pbar:
            for scene_id, faces in selected_frames.items():
                for face_data in faces:
                    embeddings = []
                    for frame_info in face_data['top_frames']:
                        image_path = os.path.join(image_dir, frame_info['image_path'])
                        face_tensor = self.load_image(image_path)

                        # Use recognition model directly (bypasses face detection)
                        try:
                            embedding = self.rec_model.get_feat(face_tensor).flatten()
                        except Exception:
                            # Return NaN embeddings on failure
                            embedding = np.full((self.embedding_dim,), np.nan, dtype=np.float32)

                        embeddings.append({
                            "frame_idx": frame_info['frame_idx'],
                            "embedding": embedding,
                            "image_path": frame_info['image_path']
                        })

                    face_embeddings.append({
                        "scene_id": scene_id,
                        "unique_face_id": face_data['unique_face_id'],
                        "global_face_id": face_data['global_face_id'],
                        "embeddings": embeddings
                    })

                    pbar.update(1)

        return face_embeddings


class FaceClusterer:
    """
    Clusters faces using Chinese Whispers algorithm on a similarity graph.

    Uses cosine similarity threshold to build edges between similar face embeddings.
    """

    def __init__(self, similarity_threshold: float = 0.6, max_iterations: int = 100, min_scenes: int = 2):
        """
        Initialize FaceClusterer.

        Args:
            similarity_threshold: Minimum cosine similarity to create edge (default: 0.6)
            max_iterations: Maximum iterations for Chinese Whispers (default: 100)
            min_scenes: Minimum unique scenes required for a valid cluster.
                       Clusters with fewer scenes are merged into nearest valid cluster.
                       Set to 1 to disable merging.
        """
        self.similarity_threshold = similarity_threshold
        self.max_iterations = max_iterations
        self.min_scenes = min_scenes

    def build_graph(self, face_embeddings):
        """
        Build similarity graph where nodes are embeddings and edges represent similarities.

        Uses vectorized cdist for efficient pairwise distance computation.
        """
        G = nx.Graph()

        # Flatten embeddings with identifiers into node_data
        node_data = [
            (i, emb_info['embedding'], face_data, emb_info['frame_idx'], emb_info['image_path'])
            for i, face_data in enumerate(face_embeddings)
            for emb_info in face_data['embeddings']
        ]

        # Add nodes to the graph
        for i, (face_idx, embedding, face_data, frame_idx, image_path) in enumerate(node_data):
            G.add_node(i, face_idx=face_idx, embedding=embedding, face_data=face_data,
                      frame_idx=frame_idx, image_path=image_path)

        # Guard against empty embeddings
        if len(node_data) == 0:
            print("No face embeddings to cluster. Returning empty graph.")
            return G, node_data

        # Vectorized similarity computation using cdist
        print(f"Computing pairwise similarities for {len(node_data)} embeddings...")
        embeddings_matrix = np.array([n[1].flatten() for n in node_data])

        # Compute all pairwise cosine distances at once
        distances = cdist(embeddings_matrix, embeddings_matrix, 'cosine')
        similarities = 1 - distances

        # Vectorized edge addition: find all pairs above threshold and add in bulk
        rows, cols = np.where(np.triu(similarities, k=1) > self.similarity_threshold)
        edges = [(r, c, similarities[r, c]) for r, c in zip(rows, cols)]
        G.add_weighted_edges_from(edges)

        print(f"Added {len(edges)} edges based on similarity threshold {self.similarity_threshold}")

        return G, node_data

    def apply_chinese_whispers(self, G: nx.Graph) -> dict:
        """
        Run Chinese Whispers clustering algorithm on the similarity graph.

        Uses convergence detection to stop early if labels stop changing.
        """
        labels = {node: i for i, node in enumerate(G.nodes())}

        with tqdm(total=self.max_iterations, desc="Running Chinese Whispers", unit="iteration") as pbar:
            for iteration in range(self.max_iterations):
                nodes = list(G.nodes())
                np.random.shuffle(nodes)

                labels_changed = False
                for node in nodes:
                    neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node)]
                    if neighbor_labels:
                        label_counts = np.bincount(neighbor_labels)
                        most_common_label = np.argmax(label_counts)
                        if labels[node] != most_common_label:
                            labels[node] = most_common_label
                            labels_changed = True

                # Check for convergence
                if not labels_changed:
                    print(f"Converged after {iteration + 1} iterations.")
                    break

                pbar.update(1)

        return labels

    def consolidate_clusters(self, initial_clusters: dict) -> dict:
        """
        Consolidate clusters by assigning each face to the best cluster.

        For each unique_face_id, considers only the clusters it was assigned to
        and assigns it to the cluster with highest average similarity.
        """
        # Map to hold the best cluster assignment for each unique_face_id
        face_best_assignment = {}
        face_embeddings = {}
        face_data_map = {}

        # Step 1: Collect all clusters and embeddings for each unique_face_id
        for cluster_id, face_list in initial_clusters.items():
            for face_data in face_list:
                unique_face_id = face_data['unique_face_id']
                embedding = face_data['embedding']

                if unique_face_id not in face_embeddings:
                    face_embeddings[unique_face_id] = []
                    face_data_map[unique_face_id] = []
                face_embeddings[unique_face_id].append(embedding)
                face_data_map[unique_face_id].append((cluster_id, face_data))

        # Step 2: For each unique_face_id, find the best cluster
        for unique_face_id, embeddings in face_embeddings.items():
            assigned_clusters = set(cluster_id for cluster_id, _ in face_data_map[unique_face_id])

            if len(assigned_clusters) == 1:
                face_best_assignment[unique_face_id] = next(iter(assigned_clusters))
            else:
                embeddings = np.array(embeddings)
                if embeddings.ndim == 3 and embeddings.shape[1] == 1:
                    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])

                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)

                max_avg_similarity = -1
                best_cluster_id = None

                for cluster_id in assigned_clusters:
                    cluster_embeddings = []
                    for face_data in initial_clusters[cluster_id]:
                        cluster_embeddings.append(face_data['embedding'])
                    cluster_embeddings = np.array(cluster_embeddings)

                    if cluster_embeddings.ndim == 3 and cluster_embeddings.shape[1] == 1:
                        cluster_embeddings = cluster_embeddings.reshape(cluster_embeddings.shape[0], cluster_embeddings.shape[2])

                    if cluster_embeddings.ndim == 1:
                        cluster_embeddings = cluster_embeddings.reshape(1, -1)

                    similarities = 1 - cdist(embeddings, cluster_embeddings, 'cosine')
                    avg_similarity = np.mean(similarities)

                    if avg_similarity > max_avg_similarity:
                        max_avg_similarity = avg_similarity
                        best_cluster_id = cluster_id

                face_best_assignment[unique_face_id] = best_cluster_id

        # Step 3: Build consolidated clusters
        consolidated_clusters = {}
        for unique_face_id, best_cluster_id in face_best_assignment.items():
            if best_cluster_id not in consolidated_clusters:
                consolidated_clusters[best_cluster_id] = []

            for cluster_id, face_data in face_data_map[unique_face_id]:
                if face_data not in consolidated_clusters[best_cluster_id]:
                    consolidated_clusters[best_cluster_id].append(face_data)

        return consolidated_clusters

    def merge_isolated_clusters(self, clusters: dict) -> dict:
        """
        Merge single-scene clusters into the nearest valid multi-scene cluster.

        Clusters with fewer than min_scenes unique scenes are merged into
        the most similar valid cluster based on centroid similarity.
        """
        if self.min_scenes <= 1:
            return clusters

        # Categorize clusters by scene count
        valid_clusters = {}
        isolated_clusters = {}

        for cluster_id, face_list in clusters.items():
            unique_scenes = set(face['scene_id'] for face in face_list)
            if len(unique_scenes) >= self.min_scenes:
                valid_clusters[cluster_id] = face_list
            else:
                isolated_clusters[cluster_id] = face_list

        print(f"Cross-scene validation: {len(valid_clusters)} valid clusters, "
              f"{len(isolated_clusters)} isolated clusters to merge")

        if not valid_clusters:
            print("Warning: No valid clusters found. Keeping all clusters unchanged.")
            return clusters

        # Compute centroids for valid clusters
        valid_centroids = {}
        for cluster_id, face_list in valid_clusters.items():
            embeddings = np.array([face['embedding'].flatten() for face in face_list])
            valid_centroids[cluster_id] = np.mean(embeddings, axis=0)

        # Merge each isolated cluster into nearest valid cluster
        for isolated_id, isolated_faces in isolated_clusters.items():
            isolated_embeddings = np.array([face['embedding'].flatten() for face in isolated_faces])
            isolated_centroid = np.mean(isolated_embeddings, axis=0)

            best_cluster_id = None
            best_similarity = -1

            for valid_id, valid_centroid in valid_centroids.items():
                similarity = 1 - cosine(isolated_centroid, valid_centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_id = valid_id

            if best_cluster_id is not None:
                valid_clusters[best_cluster_id].extend(isolated_faces)
                isolated_scenes = set(face['scene_id'] for face in isolated_faces)
                print(f"Merged isolated cluster {isolated_id} ({list(isolated_scenes)}) "
                      f"into cluster {best_cluster_id} (similarity={best_similarity:.3f})")

        return valid_clusters

    def cluster_faces(self, face_embeddings: list) -> dict:
        """
        Cluster faces based on their embeddings using Chinese Whispers.

        Args:
            face_embeddings: List of face embeddings with metadata

        Returns:
            Dictionary mapping cluster_id to list of face data
        """
        G, node_data = self.build_graph(face_embeddings)
        labels = self.apply_chinese_whispers(G)

        # Build initial clusters
        initial_clusters = {}
        for node_idx, label in labels.items():
            face_data = node_data[node_idx][2]
            frame_idx = node_data[node_idx][3]
            image_path = node_data[node_idx][4]

            if label not in initial_clusters:
                initial_clusters[label] = []

            initial_clusters[label].append({
                "scene_id": face_data['scene_id'],
                "unique_face_id": face_data['unique_face_id'],
                "global_face_id": face_data['global_face_id'],
                "frame_idx": frame_idx,
                "image_path": image_path,
                "embedding": node_data[node_idx][1]
            })

        # Consolidate and merge
        consolidated = self.consolidate_clusters(initial_clusters)
        return self.merge_isolated_clusters(consolidated)
