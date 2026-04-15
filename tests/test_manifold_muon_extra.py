"""Additional tests for manifold_muon."""

import pytest
import torch


class TestManifoldMuon:
    """Test manifold muon optimizer."""

    def test_manifold_muon_preserves_shape(self):
        """Should preserve input shape."""
        from src.manifold_muon import manifold_muon
        # Use tall matrices (rows > cols)
        W = torch.randn(8, 4)
        G = torch.randn(8, 4)
        result = manifold_muon(W, G)
        assert result.shape == (8, 4)

    def test_manifold_muon_wide_matrix(self):
        """Should handle wide matrices (cols > rows)."""
        from src.manifold_muon import manifold_muon
        W = torch.randn(4, 8)
        G = torch.randn(4, 8)
        result = manifold_muon(W, G)
        assert result.shape == (4, 8)

    def test_manifold_muon_no_nan(self):
        """Should not produce NaN."""
        from src.manifold_muon import manifold_muon
        W = torch.randn(8, 4)
        G = torch.randn(8, 4)
        result = manifold_muon(W, G)
        assert not torch.isnan(result).any()

    def test_manifold_muon_orthogonality(self):
        """Result columns should be approximately orthonormal."""
        from src.manifold_muon import manifold_muon
        W = torch.randn(8, 4)
        G = torch.randn(8, 4)
        result = manifold_muon(W, G, steps=50)
        # Check orthonormal: Q^T @ Q should be close to identity
        QtQ = result.T @ result
        assert torch.allclose(QtQ, torch.eye(4), atol=0.1)