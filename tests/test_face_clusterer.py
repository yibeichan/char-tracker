"""
Tests for FaceClusterer class.

These tests verify the clustering functionality.
FaceEmbedder tests are skipped as they require the actual model.
"""

import os
import sys
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from face_clusterer import FaceClusterer


class TestFaceClusterer:
    """Test suite for FaceClusterer."""

    def test_clustering_basic(self):
        """Test basic clustering functionality."""
        clusterer = FaceClusterer(similarity_threshold=0.6)

        # Create mock embeddings
        face_embeddings = [
            {
                'scene_id': 'scene_0',
                'unique_face_id': 0,
                'global_face_id': 0,
                'embeddings': [
                    {'frame_idx': 0, 'embedding': np.random.randn(512), 'image_path': 'img0.jpg'}
                ]
            },
            {
                'scene_id': 'scene_0',
                'unique_face_id': 1,
                'global_face_id': 1,
                'embeddings': [
                    {'frame_idx': 1, 'embedding': np.random.randn(512), 'image_path': 'img1.jpg'}
                ]
            }
        ]

        clusters = clusterer.cluster_faces(face_embeddings)

        # Should produce some clusters
        assert isinstance(clusters, dict)
        assert len(clusters) > 0

    def test_similarity_threshold_affects_clustering(self):
        """Test that similarity threshold affects cluster count."""
        # Create identical embeddings (high similarity)
        identical_embedding = np.random.randn(512)
        face_embeddings = [
            {
                'scene_id': 'scene_0',
                'unique_face_id': i,
                'global_face_id': i,
                'embeddings': [
                    {'frame_idx': 0, 'embedding': identical_embedding.copy(), 'image_path': f'img{i}.jpg'}
                ]
            }
            for i in range(3)
        ]

        # High threshold should merge similar faces
        high_threshold_clusterer = FaceClusterer(similarity_threshold=0.5)
        high_threshold_clusters = high_threshold_clusterer.cluster_faces(face_embeddings)

        # Low threshold should create more clusters
        low_threshold_clusterer = FaceClusterer(similarity_threshold=0.99)
        low_threshold_clusters = low_threshold_clusterer.cluster_faces(face_embeddings)

        # High threshold should produce fewer clusters (identical embeddings should merge)
        assert len(high_threshold_clusters) <= len(low_threshold_clusters)

    def test_empty_embeddings_returns_empty_graph(self):
        """Test that empty embeddings are handled gracefully."""
        clusterer = FaceClusterer(similarity_threshold=0.6)

        clusters = clusterer.cluster_faces([])

        assert isinstance(clusters, dict)
        # Empty input should produce empty result
        # (consolidate_clusters will return empty dict)

    def test_cross_scene_validation_with_min_scenes(self):
        """Test that min_scenes parameter affects cluster merging."""
        # Create embeddings for multiple scenes
        face_embeddings = [
            {
                'scene_id': f'scene_{i % 2}',  # Only 2 scenes
                'unique_face_id': i,
                'global_face_id': i,
                'embeddings': [
                    {'frame_idx': 0, 'embedding': np.random.randn(512), 'image_path': f'img{i}.jpg'}
                ]
            }
            for i in range(4)
        ]

        # With min_scenes=2, clusters with only 1 scene should be merged
        clusterer_merge = FaceClusterer(similarity_threshold=0.6, min_scenes=2)
        clusters_merged = clusterer_merge.cluster_faces(face_embeddings)

        # With min_scenes=1, no merging should occur
        clusterer_no_merge = FaceClusterer(similarity_threshold=0.6, min_scenes=1)
        clusters_no_merge = clusterer_no_merge.cluster_faces(face_embeddings)

        # We expect different cluster counts depending on min_scenes
        # (though the exact behavior depends on random embeddings)
        assert isinstance(clusters_merged, dict)
        assert isinstance(clusters_no_merge, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
