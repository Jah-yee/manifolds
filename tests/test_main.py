"""Tests for main.py entry points."""

import pytest
import torch


class TestHypersphericalDescent:
    """Test hyperspherical descent optimizer."""

    def test_hyperspherical_descent_preserves_shape(self):
        """Should preserve input shape."""
        from src.hyperspherical_descent import hyperspherical_descent
        for shape in [(4,), (8,), (16,)]:
            W = torch.randn(*shape)
            G = torch.randn(*shape)
            result = hyperspherical_descent(W, G)
            assert result.shape == shape

    def test_hyperspherical_descent_unit_norm(self):
        """Result should have unit norm."""
        from src.hyperspherical_descent import hyperspherical_descent
        W = torch.randn(8)
        G = torch.randn(8)
        result = hyperspherical_descent(W, G)
        assert torch.allclose(result.norm(), torch.tensor(1.0), atol=1e-5)

    def test_hyperspherical_descent_no_nan(self):
        """Should not produce NaN."""
        from src.hyperspherical_descent import hyperspherical_descent
        W = torch.randn(8)
        G = torch.randn(8)
        result = hyperspherical_descent(W, G)
        assert not torch.isnan(result).any()

    def test_hyperspherical_descent_2d(self):
        """Should work with 2D tensors."""
        from src.hyperspherical_descent import hyperspherical_descent
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result = hyperspherical_descent(W, G)
        assert result.shape == (4, 4)
        assert not torch.isnan(result).any()