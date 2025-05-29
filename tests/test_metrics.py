import pytest
import numpy as np
from unittest.mock import patch
from src.metrics import EmbeddingMetrics, average_pairwise_cosine_similarity, mev, intra_inter_similarity


class TestEmbeddingMetrics:
    """Test suite for EmbeddingMetrics class."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        np.random.seed(42)  # For reproducible tests
        return np.random.randn(50, 10)  # 50 samples, 10 dimensions
    
    @pytest.fixture
    def sample_labels(self):
        """Sample labels for testing (5 categories, 10 samples each)."""
        return np.repeat(range(5), 10)
    
    @pytest.fixture
    def anisotropic_embeddings(self):
        """Larger set of embeddings for anisotropic baseline."""
        np.random.seed(123)
        return np.random.randn(200, 10)  # 200 samples, 10 dimensions
    
    @pytest.fixture
    def metrics_no_aniso(self, sample_embeddings, sample_labels):
        """EmbeddingMetrics without anisotropic correction."""
        return EmbeddingMetrics(embeddings=sample_embeddings, labels=sample_labels)
    
    @pytest.fixture 
    def metrics_with_aniso(self, sample_embeddings, sample_labels, anisotropic_embeddings):
        """EmbeddingMetrics with anisotropic correction."""
        return EmbeddingMetrics(
            embeddings=sample_embeddings,
            labels=sample_labels,
            anisotropic_embeddings=anisotropic_embeddings
        )

    def test_initialization_basic(self, sample_embeddings):
        """Test basic initialization."""
        metrics = EmbeddingMetrics(embeddings=sample_embeddings)
        
        assert np.array_equal(metrics.embeddings, sample_embeddings)
        assert metrics.labels is None
        assert metrics.anisotropic_embeddings is None
        assert metrics.sample_size is None

    def test_initialization_with_labels(self, sample_embeddings, sample_labels):
        """Test initialization with labels."""
        metrics = EmbeddingMetrics(embeddings=sample_embeddings, labels=sample_labels)
        
        assert np.array_equal(metrics.embeddings, sample_embeddings)
        assert np.array_equal(metrics.labels, sample_labels)

    def test_initialization_preserves_original_data(self, sample_embeddings, sample_labels):
        """Test that initialization doesn't modify the original data."""
        original_embeddings = sample_embeddings.copy()
        original_labels = sample_labels.copy()
        
        metrics = EmbeddingMetrics(embeddings=sample_embeddings, labels=sample_labels)
        
        # Original data should be unchanged
        assert np.array_equal(sample_embeddings, original_embeddings)
        assert np.array_equal(sample_labels, original_labels)
        # But metrics should have the data
        assert np.array_equal(metrics.embeddings, original_embeddings)
        assert np.array_equal(metrics.labels, original_labels)

    # Test raw metrics
    def test_mev_raw(self, metrics_no_aniso):
        """Test MEV calculation."""
        mev_result = metrics_no_aniso.mev_raw
        
        assert isinstance(mev_result, float)
        assert 0 <= mev_result <= 1  # MEV should be between 0 and 1
        
        # Test caching - should return same value
        assert metrics_no_aniso.mev_raw == mev_result

    def test_average_similarity_raw(self, metrics_no_aniso):
        """Test average similarity calculation."""
        avg_sim, std_sim = metrics_no_aniso.average_similarity_raw
        
        assert isinstance(avg_sim, float)
        assert isinstance(std_sim, float)
        assert -1 <= avg_sim <= 1  # Cosine similarity bounds
        assert std_sim >= 0  # Standard deviation is non-negative
        
        # Test caching
        assert metrics_no_aniso.average_similarity_raw == (avg_sim, std_sim)

    def test_similarity_matrix_caching(self, metrics_no_aniso):
        """Test that similarity matrix is computed once and cached."""
        matrix1 = metrics_no_aniso.similarity_matrix
        matrix2 = metrics_no_aniso.similarity_matrix
        
        # Should be the same object (cached)
        assert matrix1 is matrix2
        assert matrix1.shape == (50, 50)

    def test_intra_inter_similarity_raw(self, metrics_no_aniso):
        """Test intra/inter similarity calculation."""
        intra, inter = metrics_no_aniso.intra_inter_similarity_raw
        
        assert isinstance(intra, float)
        assert isinstance(inter, float)
        # Note: intra isn't always >= inter with random data, so just check types

    def test_intra_inter_similarity_without_labels_raises_error(self, sample_embeddings):
        """Test that intra/inter similarity raises error without labels."""
        metrics = EmbeddingMetrics(embeddings=sample_embeddings)
        
        with pytest.raises(ValueError, match="Labels required"):
            _ = metrics.intra_inter_similarity_raw

    # Test anisotropic correction
    def test_anisotropic_baselines_computation(self, metrics_with_aniso):
        """Test anisotropic baseline computation."""
        aniso_sim, aniso_mev = metrics_with_aniso.anisotropic_baselines
        
        assert isinstance(aniso_sim, float)
        assert isinstance(aniso_mev, float)
        assert 0 <= aniso_mev <= 1
        assert -1 <= aniso_sim <= 1

    def test_anisotropic_baselines_caching(self, metrics_with_aniso):
        """Test that anisotropic baselines are cached."""
        baselines1 = metrics_with_aniso.anisotropic_baselines
        baselines2 = metrics_with_aniso.anisotropic_baselines
        
        assert baselines1 is baselines2  # Same object (cached)

    def test_anisotropic_baselines_without_embeddings_raises_error(self, sample_embeddings):
        """Test that accessing corrected metrics without anisotropic embeddings raises error."""
        metrics = EmbeddingMetrics(embeddings=sample_embeddings)
        
        with pytest.raises(ValueError, match="anisotropic_embeddings must be provided"):
            _ = metrics.anisotropic_baselines

    def test_mev_corrected(self, metrics_with_aniso):
        """Test corrected MEV calculation."""
        mev_corrected = metrics_with_aniso.mev_corrected
        mev_raw = metrics_with_aniso.mev_raw
        _, aniso_mev = metrics_with_aniso.anisotropic_baselines
        
        assert isinstance(mev_corrected, float)
        assert abs(mev_corrected - (mev_raw - aniso_mev)) < 1e-10

    def test_average_similarity_corrected(self, metrics_with_aniso):
        """Test corrected average similarity."""
        avg_sim_corrected = metrics_with_aniso.average_similarity_corrected
        avg_sim_raw, _ = metrics_with_aniso.average_similarity_raw
        aniso_sim, _ = metrics_with_aniso.anisotropic_baselines
        
        assert isinstance(avg_sim_corrected, float)
        assert abs(avg_sim_corrected - (avg_sim_raw - aniso_sim)) < 1e-10

    def test_intra_inter_similarity_corrected(self, metrics_with_aniso):
        """Test corrected intra/inter similarity."""
        intra_corr, inter_corr = metrics_with_aniso.intra_inter_similarity_corrected
        intra_raw, inter_raw = metrics_with_aniso.intra_inter_similarity_raw
        aniso_sim, _ = metrics_with_aniso.anisotropic_baselines
        
        assert isinstance(intra_corr, float)
        assert isinstance(inter_corr, float)
        assert abs(intra_corr - (intra_raw - aniso_sim)) < 1e-10
        assert abs(inter_corr - (inter_raw - aniso_sim)) < 1e-10

    # Test sampling functionality
    def test_anisotropic_sampling(self, sample_embeddings, sample_labels):
        """Test that anisotropic embeddings are sampled when sample_size is specified."""
        large_aniso_embeddings = np.random.randn(500, 10)
        
        metrics = EmbeddingMetrics(
            embeddings=sample_embeddings,
            labels=sample_labels,
            anisotropic_embeddings=large_aniso_embeddings,
            sample_size=100
        )
        
        # Should work without error (sampling should happen internally)
        aniso_sim, aniso_mev = metrics.anisotropic_baselines
        assert isinstance(aniso_sim, float)
        assert isinstance(aniso_mev, float)

    def test_no_sampling_when_aniso_embeddings_smaller_than_sample_size(self, sample_embeddings, sample_labels):
        """Test that no sampling occurs when anisotropic embeddings are smaller than sample_size."""
        small_aniso_embeddings = np.random.randn(50, 10)
        
        metrics = EmbeddingMetrics(
            embeddings=sample_embeddings,
            labels=sample_labels,
            anisotropic_embeddings=small_aniso_embeddings,
            sample_size=100  # Larger than aniso embeddings
        )
        
        # Should still work (no sampling needed)
        aniso_sim, aniso_mev = metrics.anisotropic_baselines
        assert isinstance(aniso_sim, float)
        assert isinstance(aniso_mev, float)

    # Test get_metrics method
    def test_get_metrics_raw(self, metrics_with_aniso):
        """Test get_metrics with raw=True."""
        metrics_dict = metrics_with_aniso.get_metrics(corrected=False)
        
        expected_keys = {
            'mev', 'average_similarity', 'similarity_std',
            'intra_similarity', 'inter_similarity'
        }
        assert set(metrics_dict.keys()) == expected_keys
        
        # Check that values match individual property access
        assert metrics_dict['mev'] == metrics_with_aniso.mev_raw
        assert metrics_dict['average_similarity'] == metrics_with_aniso.average_similarity_raw[0]
        assert metrics_dict['similarity_std'] == metrics_with_aniso.average_similarity_raw[1]

    def test_get_metrics_corrected(self, metrics_with_aniso):
        """Test get_metrics with corrected=True."""
        metrics_dict = metrics_with_aniso.get_metrics(corrected=True)
        
        expected_keys = {
            'mev', 'average_similarity', 'similarity_std',
            'intra_similarity', 'inter_similarity'
        }
        assert set(metrics_dict.keys()) == expected_keys
        
        # Check that values match individual property access
        assert metrics_dict['mev'] == metrics_with_aniso.mev_corrected
        assert metrics_dict['average_similarity'] == metrics_with_aniso.average_similarity_corrected

    def test_get_metrics_with_include_filter(self, metrics_with_aniso):
        """Test get_metrics with include filter."""
        metrics_dict = metrics_with_aniso.get_metrics(
            corrected=True, 
            include=['mev']
        )
        
        assert set(metrics_dict.keys()) == {'mev'}
        assert metrics_dict['mev'] == metrics_with_aniso.mev_corrected

    def test_get_metrics_without_labels(self, sample_embeddings, anisotropic_embeddings):
        """Test get_metrics when no labels are provided."""
        metrics = EmbeddingMetrics(
            embeddings=sample_embeddings,
            anisotropic_embeddings=anisotropic_embeddings
        )
        
        metrics_dict = metrics.get_metrics(corrected=False)
        
        # Should only have basic metrics, not intra/inter
        expected_keys = {'mev', 'average_similarity', 'similarity_std'}
        assert set(metrics_dict.keys()) == expected_keys

    # Test edge cases
    def test_identical_embeddings(self):
        """Test behavior with identical embeddings."""
        identical_embeddings = np.ones((10, 5))  # All identical
        metrics = EmbeddingMetrics(embeddings=identical_embeddings)
        
        # Should handle gracefully - suppress the expected RuntimeWarning
        with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
            mev_result = metrics.mev_raw
        
        assert isinstance(mev_result, float)
        # MEV will be NaN for identical embeddings (no variance)
        assert np.isnan(mev_result)
        
        avg_sim, std_sim = metrics.average_similarity_raw
        assert abs(avg_sim - 1.0) < 1e-10  # Should be very close to 1.0
        assert abs(std_sim - 0.0) < 1e-10  # Should be very close to 0.0

    def test_single_embedding(self):
        """Test behavior with single embedding."""
        single_embedding = np.random.randn(1, 5)
        metrics = EmbeddingMetrics(embeddings=single_embedding)
        
        # MEV should handle single sample - suppress the expected RuntimeWarning
        with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
            mev_result = metrics.mev_raw
        
        assert isinstance(mev_result, float)
        # MEV will be NaN for single embedding (can't compute variance)
        assert np.isnan(mev_result)

    def test_sampling_behavior(self, sample_embeddings, sample_labels):
        """Test that sampling works correctly for anisotropic embeddings."""
        # Create embeddings larger than sample size
        large_aniso_embeddings = np.random.randn(500, 10)
        
        metrics = EmbeddingMetrics(
            embeddings=sample_embeddings,
            labels=sample_labels,
            anisotropic_embeddings=large_aniso_embeddings,
            sample_size=100
        )
        
        # Should work without error (sampling should happen internally)
        aniso_sim, aniso_mev = metrics.anisotropic_baselines
        assert isinstance(aniso_sim, float)
        assert isinstance(aniso_mev, float)
        
        # Test that it's cached properly
        aniso_sim2, aniso_mev2 = metrics.anisotropic_baselines
        assert aniso_sim == aniso_sim2
        assert aniso_mev == aniso_mev2

    def test_error_handling_for_corrected_metrics_without_aniso(self, sample_embeddings, sample_labels):
        """Test that corrected metrics raise proper errors without anisotropic embeddings."""
        metrics = EmbeddingMetrics(embeddings=sample_embeddings, labels=sample_labels)
        
        with pytest.raises(ValueError, match="anisotropic_embeddings must be provided"):
            _ = metrics.mev_corrected
            
        with pytest.raises(ValueError, match="anisotropic_embeddings must be provided"):
            _ = metrics.average_similarity_corrected
            
        with pytest.raises(ValueError, match="anisotropic_embeddings must be provided"):
            _ = metrics.intra_inter_similarity_corrected

    def test_get_metrics_handles_corrected_without_aniso(self, sample_embeddings, sample_labels):
        """Test that get_metrics handles requests for corrected metrics without aniso embeddings."""
        metrics = EmbeddingMetrics(embeddings=sample_embeddings, labels=sample_labels)
        
        # This should raise an error when trying to access corrected metrics
        with pytest.raises(ValueError, match="anisotropic_embeddings must be provided"):
            metrics.get_metrics(corrected=True)

    def test_individual_corrected_metrics_return_types(self, metrics_with_aniso):
        """Test that individual corrected metrics return correct types."""
        # These should all return single floats, not tuples
        mev_corr = metrics_with_aniso.mev_corrected
        avg_sim_corr = metrics_with_aniso.average_similarity_corrected
        intra_inter_corr = metrics_with_aniso.intra_inter_similarity_corrected
        
        assert isinstance(mev_corr, float)
        assert isinstance(avg_sim_corr, float)  # Should be float, not tuple
        assert isinstance(intra_inter_corr, tuple)  # This should be tuple
        assert len(intra_inter_corr) == 2

    def test_small_embedding_dimensions(self):
        """Test with very small embedding dimensions."""
        small_embeddings = np.random.randn(5, 2)  # Only 2D
        labels = np.array([0, 0, 1, 1, 2])
        
        metrics = EmbeddingMetrics(embeddings=small_embeddings, labels=labels)
        
        # Should still work
        mev_result = metrics.mev_raw
        assert isinstance(mev_result, float)
        
        intra, inter = metrics.intra_inter_similarity_raw
        assert isinstance(intra, float)
        assert isinstance(inter, float)


class TestOriginalFunctions:
    """Test the original standalone functions."""
    
    @pytest.fixture
    def sample_embeddings(self):
        np.random.seed(42)
        return np.random.randn(20, 10)
    
    @pytest.fixture
    def sample_labels(self):
        return np.repeat(range(4), 5)
    
    def test_mev_function(self, sample_embeddings):
        """Test standalone MEV function."""
        result = mev(sample_embeddings)
        
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_average_pairwise_cosine_similarity_function(self, sample_embeddings):
        """Test standalone average pairwise cosine similarity function."""
        similarities, avg_sim, std_sim = average_pairwise_cosine_similarity(sample_embeddings)
        
        assert similarities.shape == (20, 20)
        assert isinstance(avg_sim, float)
        assert isinstance(std_sim, float)
        assert -1 <= avg_sim <= 1
        assert std_sim >= 0

    def test_intra_inter_similarity_function(self, sample_embeddings, sample_labels):
        """Test standalone intra/inter similarity function."""
        intra, inter = intra_inter_similarity(sample_embeddings, sample_labels)
        
        assert isinstance(intra, float)
        assert isinstance(inter, float)
        # With random data, intra isn't always >= inter, so just check types

    def test_intra_inter_similarity_with_sampling(self, sample_embeddings, sample_labels):
        """Test intra/inter similarity with sampling."""
        intra, inter = intra_inter_similarity(sample_embeddings, sample_labels, sample=10)
        
        assert isinstance(intra, float)
        assert isinstance(inter, float)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])