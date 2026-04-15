import pytest
import torch
import math
from src.manifold_muon import manifold_muon


class TestManifoldMuon:
    @pytest.fixture
    def seed(self):
        torch.manual_seed(42)
        yield
        torch.manual_seed(42)

    def test_manifold_muon_preserves_shape(self):
        """manifold_muon should preserve input shape."""
        for shape in [(4, 4), (4, 3), (3, 4), (5, 2), (2, 5)]:
            W = torch.randn(*shape)
            G = torch.randn(*shape)
            result = manifold_muon(W, G)
            assert result.shape == shape

    def test_manifold_muon_wide_matrix(self):
        """manifold_muon should handle wide matrices."""
        W = torch.randn(3, 5)
        G = torch.randn(3, 5)
        result = manifold_muon(W, G)
        assert result.shape == (3, 5)

    def test_manifold_muon_tall_matrix(self):
        """manifold_muon should handle tall matrices."""
        W = torch.randn(5, 3)
        G = torch.randn(5, 3)
        result = manifold_muon(W, G)
        assert result.shape == (5, 3)

    def test_manifold_muon_no_nan(self):
        """manifold_muon should not produce NaN values."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result = manifold_muon(W, G, steps=10)
        assert not torch.isnan(result).any()

    def test_manifold_muon_no_inf(self):
        """manifold_muon should not produce Inf values."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result = manifold_muon(W, G, steps=10)
        assert not torch.isinf(result).any()

    def test_manifold_muon_convergence(self):
        """manifold_muon should converge to a stationary point."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result = manifold_muon(W, G, steps=100, tol=1e-6)
        # Check that final result is on manifold (W.T @ W = I)
        W_result = result
        metric = W_result.T @ W_result
        identity = torch.eye(W_result.shape[1])
        assert torch.allclose(metric, identity, atol=1e-3)

    def test_manifold_muon_custom_eta(self):
        """manifold_muon should respect eta parameter."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result1 = manifold_muon(W, G, eta=0.01)
        result2 = manifold_muon(W, G, eta=1.0)
        # Different eta should give different results
        assert not torch.allclose(result1, result2)

    def test_manifold_muon_custom_alpha(self):
        """manifold_muon should respect alpha parameter."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result1 = manifold_muon(W, G, alpha=0.001)
        result2 = manifold_muon(W, G, alpha=0.1)
        # Different alpha should give different results
        assert not torch.allclose(result1, result2)

    def test_manifold_muon_custom_steps(self):
        """manifold_muon should respect steps parameter."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result1 = manifold_muon(W, G, steps=5)
        result2 = manifold_muon(W, G, steps=50)
        # More steps should give different results
        assert not torch.allclose(result1, result2)

    def test_manifold_muon_result_is_orthogonal(self):
        """Result should be orthogonal (columns are orthonormal)."""
        W = torch.randn(4, 3)
        G = torch.randn(4, 3)
        result = manifold_muon(W, G, steps=50)
        # Check that columns are orthonormal
        metric = result.T @ result
        assert torch.allclose(metric, torch.eye(3), atol=1e-3)

    def test_manifold_muon_tensor_input(self):
        """manifold_muon should accept torch tensors."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        assert isinstance(W, torch.Tensor) and isinstance(G, torch.Tensor)
        result = manifold_muon(W, G)
        assert isinstance(result, torch.Tensor)

    def test_manifold_muon_square_matrix(self):
        """manifold_muon should work with square matrices."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result = manifold_muon(W, G)
        assert result.shape == (4, 4)