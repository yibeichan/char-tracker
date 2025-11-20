import os
import cv2
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.spatial.distance import cosine, cdist
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import Dataset, DataLoader

class FaceEmbedder:
    def __init__(self, model=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model or InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def load_image(self, image_path):
        """Load an image from disk and convert it to a tensor."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        
        # Optionally preprocess image if needed (resize, etc.)
        face_tensor = self.preprocess_face(image)
        return face_tensor

    def preprocess_face(self, face_image):
        """Preprocess the face image for embedding extraction (resize, normalize, etc.)."""
        face_image = cv2.resize(face_image, (160, 160))  # Resize to 160x160 pixels if required
        face_tensor = torch.tensor(face_image).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        face_tensor = (face_tensor - 127.5) / 128.0  # Normalize
        return face_tensor

    def get_face_embeddings(self, selected_frames, image_dir):
        """Get embeddings for each cropped face image."""
        face_embeddings = []

        # Initialize progress bar
        with tqdm(total=len(selected_frames), desc="Extracting Face Embeddings", unit="face") as pbar:
            for scene_id, faces in selected_frames.items():
                for face_data in faces:
                    embeddings = []
                    for frame_info in face_data['top_frames']:
                        image_path = os.path.join(image_dir, frame_info['image_path'])  # Construct full path to image
                        face_tensor = self.load_image(image_path)
                        with torch.no_grad():
                            embedding = self.model(face_tensor).cpu().numpy()
                        embeddings.append({
                            "frame_idx": frame_info['frame_idx'],
                            "embedding": embedding,
                            "image_path": frame_info['image_path']
                        })

                    face_embeddings.append({
                        "scene_id": scene_id,
                        "unique_face_id": face_data['unique_face_id'],
                        "global_face_id": face_data['global_face_id'],
                        "embeddings": embeddings  # Each embedding with its image path
                    })

                    # Update progress bar
                    pbar.update(1)

        return face_embeddings

    def get_face_embeddings_batch(self, selected_frames, image_dir, batch_size=32, num_workers=4):
        """
        Get embeddings for each cropped face image using batch processing.

        This method is optimized for GPU utilization by processing multiple images
        at once instead of one-by-one.

        Args:
            selected_frames: Dictionary of selected frames per scene
            image_dir: Directory containing face images
            batch_size: Number of images to process in each batch (default: 32)
            num_workers: Number of worker threads for data loading (default: 4)

        Returns:
            List of face embeddings with metadata
        """
        # Create dataset and dataloader
        dataset = FaceImageDataset(selected_frames, image_dir, self.preprocess_face, self.device)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        # Process batches
        all_embeddings = []

        with torch.no_grad():
            for batch_tensors, batch_metadata in tqdm(dataloader, desc="Extracting Face Embeddings (Batched)", unit="batch"):
                # Move batch to device
                batch_tensors = batch_tensors.to(self.device)

                # Get embeddings for entire batch
                batch_embeddings = self.model(batch_tensors).cpu().numpy()

                # Store embeddings with metadata
                for i, embedding in enumerate(batch_embeddings):
                    metadata = batch_metadata[i]
                    all_embeddings.append({
                        'embedding': embedding,
                        'metadata': metadata
                    })

        # Reconstruct the hierarchical structure
        return self._reconstruct_embedding_structure(all_embeddings, selected_frames)

    def _reconstruct_embedding_structure(self, flat_embeddings, selected_frames):
        """
        Reconstruct the hierarchical structure of embeddings from flat list.

        Args:
            flat_embeddings: Flat list of embeddings with metadata
            selected_frames: Original selected_frames dictionary for structure

        Returns:
            Embeddings in the same structure as get_face_embeddings()
        """
        face_embeddings = []
        embedding_idx = 0

        for scene_id, faces in selected_frames.items():
            for face_data in faces:
                embeddings = []
                for frame_info in face_data['top_frames']:
                    emb_data = flat_embeddings[embedding_idx]
                    embeddings.append({
                        "frame_idx": frame_info['frame_idx'],
                        "embedding": emb_data['embedding'],
                        "image_path": frame_info['image_path']
                    })
                    embedding_idx += 1

                face_embeddings.append({
                    "scene_id": scene_id,
                    "unique_face_id": face_data['unique_face_id'],
                    "global_face_id": face_data['global_face_id'],
                    "embeddings": embeddings
                })

        return face_embeddings

class FaceImageDataset(Dataset):
    """
    PyTorch Dataset for loading face images in batches.

    This dataset flattens the hierarchical structure of faces/frames into a single
    list for efficient batch processing.
    """
    def __init__(self, selected_frames, image_dir, preprocess_fn, device):
        """
        Args:
            selected_frames: Dictionary of selected frames per scene
            image_dir: Directory containing face images
            preprocess_fn: Function to preprocess face images
            device: Device for tensor operations
        """
        self.image_dir = image_dir
        self.preprocess_fn = preprocess_fn
        self.device = device

        # Flatten all image paths with metadata
        self.image_items = []
        for scene_id, faces in selected_frames.items():
            for face_data in faces:
                for frame_info in face_data['top_frames']:
                    self.image_items.append({
                        'image_path': os.path.join(image_dir, frame_info['image_path']),
                        'scene_id': scene_id,
                        'unique_face_id': face_data['unique_face_id'],
                        'frame_idx': frame_info['frame_idx']
                    })

    def __len__(self):
        return len(self.image_items)

    def __getitem__(self, idx):
        """
        Load and preprocess a single image.

        Returns:
            tuple: (preprocessed_tensor, metadata_dict)
        """
        item = self.image_items[idx]

        # Load image
        image = cv2.imread(item['image_path'])
        if image is None:
            raise ValueError(f"Error loading image: {item['image_path']}")

        # Preprocess (resize, normalize)
        face_tensor = self.preprocess_fn(image)

        # Remove batch dimension added by preprocess_fn
        face_tensor = face_tensor.squeeze(0)

        # Return tensor and metadata
        metadata = {
            'scene_id': item['scene_id'],
            'unique_face_id': item['unique_face_id'],
            'frame_idx': item['frame_idx'],
            'image_path': item['image_path']
        }

        return face_tensor, metadata

class FaceClusterer:
    def __init__(self, similarity_threshold: float = 0.6, max_iterations: int = 100):
        self.similarity_threshold = similarity_threshold
        self.max_iterations = max_iterations

    def build_graph(self, face_embeddings):
        """Builds a graph where nodes represent embeddings, and edges represent similarities (vectorized)."""
        G = nx.Graph()

        # Flatten embeddings with identifiers into node_data
        node_data = [
            (i, emb_info['embedding'], face_data, emb_info['frame_idx'], emb_info['image_path'])
            for i, face_data in enumerate(face_embeddings)
            for emb_info in face_data['embeddings']
        ]

        # Add nodes to the graph
        for i, (face_idx, embedding, face_data, frame_idx, image_path) in enumerate(node_data):
            G.add_node(i, face_idx=face_idx, embedding=embedding, face_data=face_data, frame_idx=frame_idx, image_path=image_path)

        # Guard against empty embeddings (no faces detected)
        if len(node_data) == 0:
            print("No face embeddings to cluster. Returning empty graph.")
            return G, node_data

        # Vectorized similarity computation using cdist (MUCH faster)
        print(f"Computing pairwise similarities for {len(node_data)} embeddings...")
        embeddings_matrix = np.array([n[1].flatten() for n in node_data])

        # Compute all pairwise cosine distances at once
        distances = cdist(embeddings_matrix, embeddings_matrix, 'cosine')
        similarities = 1 - distances

        # Vectorized edge addition: find all pairs above threshold and add in bulk
        rows, cols = np.where(np.triu(similarities, k=1) > self.similarity_threshold)
        edges = [(r, c, similarities[r, c]) for r, c in zip(rows, cols)]
        G.add_weighted_edges_from(edges)
        edge_count = len(edges)

        print(f"Added {edge_count} edges based on similarity threshold {self.similarity_threshold}")

        return G, node_data

    def apply_chinese_whispers(self, G: nx.Graph) -> dict:
        """Runs the Chinese Whispers algorithm on the similarity graph with convergence check."""
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
        """Consolidate clusters by assigning each face to the best cluster among its initial assignments."""
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
        
        # Step 2: For each unique_face_id, consider only the clusters it was assigned to
        for unique_face_id, embeddings in face_embeddings.items():
            assigned_clusters = set(cluster_id for cluster_id, _ in face_data_map[unique_face_id])
            print(f"Unique Face ID: {unique_face_id}, Assigned Clusters: {assigned_clusters}")
            if len(assigned_clusters) == 1:
                # All embeddings assigned to the same cluster, no need to compute similarities
                best_cluster_id = next(iter(assigned_clusters))
            else:
                embeddings = np.array(embeddings)
                if embeddings.ndim == 3 and embeddings.shape[1] == 1:
                    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])
                
                # Ensure embeddings is a 2D array
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                max_avg_similarity = -1
                best_cluster_id = None
                
                for cluster_id in assigned_clusters:
                    # Get embeddings of the cluster
                    cluster_embeddings = []
                    for face_data in initial_clusters[cluster_id]:
                        cluster_embeddings.append(face_data['embedding'])
                    cluster_embeddings = np.array(cluster_embeddings)
                    
                    # Reshape cluster_embeddings if necessary
                    if cluster_embeddings.ndim == 3 and cluster_embeddings.shape[1] == 1:
                        cluster_embeddings = cluster_embeddings.reshape(cluster_embeddings.shape[0], cluster_embeddings.shape[2])
                    
                    # Ensure cluster_embeddings is a 2D array
                    if cluster_embeddings.ndim == 1:
                        cluster_embeddings = cluster_embeddings.reshape(1, -1)
                    
                    # Step 3: Compute average similarity between face embeddings and cluster embeddings
                    similarities = 1 - cdist(embeddings, cluster_embeddings, 'cosine')
                    avg_similarity = np.mean(similarities)
                    
                    if avg_similarity > max_avg_similarity:
                        max_avg_similarity = avg_similarity
                        best_cluster_id = cluster_id
            
            # Assign the face to the best cluster
            face_best_assignment[unique_face_id] = best_cluster_id
        
        # Step 4: Build consolidated clusters based on best assignments
        consolidated_clusters = {}
        for unique_face_id, best_cluster_id in face_best_assignment.items():
            if best_cluster_id not in consolidated_clusters:
                consolidated_clusters[best_cluster_id] = []
            
            # Add all face_data instances for this unique_face_id to the best cluster
            for cluster_id, face_data in face_data_map[unique_face_id]:
                # Avoid duplicates
                if face_data not in consolidated_clusters[best_cluster_id]:
                    consolidated_clusters[best_cluster_id].append(face_data)
        
        return consolidated_clusters


    def _max_similarity(self, face_list: list, embedding: np.ndarray) -> float:
        """Helper function to calculate the maximum similarity of an embedding with a list of faces."""
        similarities = similarities = [1 - cosine(face['embedding'].flatten(), embedding.flatten()) for face in face_list]
        return max(similarities)

    def cluster_faces(self, face_embeddings: list) -> dict:
        """Cluster faces based on their embeddings using Chinese Whispers."""
        G, node_data = self.build_graph(face_embeddings)
        labels = self.apply_chinese_whispers(G)

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
                "image_path": image_path,  # Include image path in the cluster data
                "embedding": node_data[node_idx][1]
            })

        return self.consolidate_clusters(initial_clusters)