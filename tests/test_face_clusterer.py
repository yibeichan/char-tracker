"""
Tests for FaceEmbedder and FaceClusterer classes.

These tests verify the correct behavior of both embedding models (vggface2 and buffalo_l),
including input preprocessing, embedding dimensions, and error handling.
"""

import os
import sys
import pytest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from face_clusterer import FaceEmbedder, FaceClusterer


class TestFaceEmbedderBuffaloL:
    """Test suite for FaceEmbedder with buffalo_l model."""

    @pytest.fixture
    def mock_insightface(self):
        """Mock InsightFace FaceAnalysis for buffalo_l tests."""
        with patch('face_clusterer.insightface') as mock_if:
            # Mock FaceAnalysis
            mock_app = MagicMock()
            mock_if.app.FaceAnalysis.return_value = mock_app

            # Mock face detection result
            mock_face = MagicMock()
            mock_face.normed_embedding = np.random.randn(512).astype(np.float32)
            mock_app.get.return_value = [mock_face]

            yield mock_app

    @pytest.fixture
    def mock_onnxruntime(self):
        """Mock onnxruntime for provider checking."""
        with patch('face_clusterer.ort') as mock_ort:
            mock_ort.get_available_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            yield mock_ort

    def test_buffalo_l_input_size(self, mock_insightface, mock_onnxruntime):
        """Test that buffalo_l uses correct input size (112x112)."""
        embedder = FaceEmbedder(model_name='buffalo_l')
        assert embedder.input_size == (112, 112), "buffalo_l should use 112x112 input size"

    def test_buffalo_l_embedding_dimension(self, mock_insightface, mock_onnxruntime):
        """Test that buffalo_l produces 512-dimensional embeddings."""
        embedder = FaceEmbedder(model_name='buffalo_l')
        assert embedder.embedding_dim == 512, "buffalo_l should produce 512-dim embeddings"

    def test_vggface2_input_size(self):
        """Test that vggface2 uses correct input size (160x160)."""
        with patch('face_clusterer.InceptionResnetV1') as mock_model:
            mock_model.return_value.eval.return_value.to.return_value = MagicMock()
            embedder = FaceEmbedder(model_name='vggface2')
            assert embedder.input_size == (160, 160), "vggface2 should use 160x160 input size"

    def test_vggface2_embedding_dimension(self):
        """Test that vggface2 produces 512-dimensional embeddings."""
        with patch('face_clusterer.InceptionResnetV1') as mock_model:
            mock_model.return_value.eval.return_value.to.return_value = MagicMock()
            embedder = FaceEmbedder(model_name='vggface2')
            assert embedder.embedding_dim == 512, "vggface2 should produce 512-dim embeddings"

    def test_buffalo_l_preprocess_returns_uint8_bgr(self, mock_insightface, mock_onnxruntime):
        """Test that buffalo_l preprocessing returns uint8 BGR image (no normalization)."""
        embedder = FaceEmbedder(model_name='buffalo_l')

        # Create a test BGR image
        test_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

        preprocessed = embedder.preprocess_face(test_image)

        # Should be uint8
        assert preprocessed.dtype == np.uint8, "buffalo_l preprocessed image should be uint8"
        # Should be resized to 112x112
        assert preprocessed.shape == (112, 112, 3), "buffalo_l should resize to 112x112"
        # Should be in valid range [0, 255]
        assert preprocessed.min() >= 0 and preprocessed.max() <= 255

    @patch('face_clusterer.cv2')
    def test_vggface2_preprocess_converts_to_rgb(self, mock_cv2):
        """Test that vggface2 preprocessing converts BGR to RGB."""
        with patch('face_clusterer.InceptionResnetV1') as mock_model:
            mock_model.return_value.eval.return_value.to.return_value = MagicMock()
            embedder = FaceEmbedder(model_name='vggface2')

            # Create a test BGR image
            test_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = test_image
            mock_cv2.cvtColor.return_value = test_image

            _ = embedder.preprocess_face(test_image)

            # Verify BGR to RGB conversion was called
            mock_cv2.cvtColor.assert_called_once()
            args = mock_cv2.cvtColor.call_args[0]
            assert args[1] == mock_cv2.COLOR_BGR2RGB, "vggface2 should convert BGR to RGB"

    def test_buffalo_l_no_face_detected_returns_nan(self, mock_insightface, mock_onnxruntime):
        """Test that buffalo_l returns NaN when no face is detected."""
        # Mock no face detected
        mock_insightface.get.return_value = []

        embedder = FaceEmbedder(model_name='buffalo_l')

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(tmp.name, test_image)
            tmp_path = tmp.name

        try:
            # Mock load_image to return the test image
            with patch.object(embedder, 'load_image', return_value=test_image):
                # Create mock selected_frames
                selected_frames = {
                    'scene_0': [{
                        'unique_face_id': 0,
                        'global_face_id': 0,
                        'top_frames': [{'image_path': 'test.jpg', 'frame_idx': 0}]
                    }]
                }

                embeddings = embedder.get_face_embeddings(selected_frames, '/tmp')

                # Check that embedding contains NaN
                embedding_array = embeddings[0]['embeddings'][0]['embedding']
                assert np.all(np.isnan(embedding_array)), "Should return NaN when no face detected"
        finally:
            os.unlink(tmp_path)

    def test_buffalo_l_provider_fallback_without_cuda(self, mock_insightface):
        """Test that buffalo_l gracefully falls back to CPU when CUDA provider unavailable."""
        with patch('face_clusterer.ort') as mock_ort:
            # Simulate CUDA provider not available
            mock_ort.get_available_providers.return_value = ['CPUExecutionProvider']

            embedder = FaceEmbedder(model_name='buffalo_l')

            # Verify FaceAnalysis was initialized with CPU provider only
            call_args = mock_insightface.app.FaceAnalysis.call_args
            providers = call_args[1]['providers']
            assert providers == ['CPUExecutionProvider'], "Should fallback to CPU when CUDA unavailable"

    def test_buffalo_l_uses_cuda_when_available(self, mock_insightface, mock_onnxruntime):
        """Test that buffalo_l uses CUDA provider when available."""
        import torch
        with patch('face_clusterer.torch.cuda.is_available', return_value=True):
            FaceEmbedder(model_name='buffalo_l')

            # Verify FaceAnalysis was initialized with CUDA provider
            call_args = mock_insightface.app.FaceAnalysis.call_args
            providers = call_args[1]['providers']
            assert 'CUDAExecutionProvider' in providers, "Should use CUDA when available"

    def test_model_switching(self, mock_insightface, mock_onnxruntime):
        """Test that switching models changes input size and embedding behavior."""
        # Create buffalo_l embedder
        buffalo_embedder = FaceEmbedder(model_name='buffalo_l')
        assert buffalo_embedder.input_size == (112, 112)
        assert buffalo_embedder.embedding_dim == 512

        # Create vggface2 embedder
        with patch('face_clusterer.InceptionResnetV1') as mock_model:
            mock_model.return_value.eval.return_value.to.return_value = MagicMock()
            vggface2_embedder = FaceEmbedder(model_name='vggface2')
            assert vggface2_embedder.input_size == (160, 160)
            assert vggface2_embedder.embedding_dim == 512


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
