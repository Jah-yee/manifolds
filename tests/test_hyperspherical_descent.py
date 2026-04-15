import pytest
import torch
import math
from src.hyperspherical_descent import hyperspherical_descent


class TestHypersphericalDescent:
    @pytest.fixture
    def seed(self):
        torch.manual_seed(42)
        yield
        torch.manual_seed(42)

    def test_hyperspherical_descent_preserves_shape(self):
        """hyperspherical_descent should preserve input shape."""
        for shape in [(4, 4), (4, 3), (3, 4), (5, 2), (2, 5), (10,)]:
            W = torch.randn(*shape)
            G = torch.randn(*shape)
            result = hyperspherical_descent(W, G)
            assert result.shape == shape

    def test_hyperspherical_descent_1d_input(self):
        """hyperspherical_descent should work with 1D input."""
        W = torch.randn(10)
        G = torch.randn(10)
        result = hyperspherical_descent(W, G)
        assert result.shape == (10,)

    def test_hyperspherical_descent_2d_input(self):
        """hyperspherical_descent should work with 2D input."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result = hyperspherical_descent(W, G)
        assert result.shape == (4, 4)

    def test_hyperspherical_descent_no_nan(self):
        """hyperspherical_descent should not produce NaN values."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result = hyperspherical_descent(W, G)
        assert not torch.isnan(result).any()

    def test_hyperspherical_descent_no_inf(self):
        """hyperspherical_descent should not produce Inf values."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result = hyperspherical_descent(W, G)
        assert not torch.isinf(result).any()

    def test_hyperspherical_descent_on_unit_sphere(self):
        """Result should lie on unit sphere."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        result = hyperspherical_descent(W, G)
        # Check that vector has unit norm (for all elements)
        for row in result:
            norm = row.norm()
            assert math.isclose(norm, 1.0, rel_tol=1e-5)

    def test_hyperspherical_descent_unit_vector(self):
        """hyperspherical_descent should keep unit vectors on unit sphere."""
        W = torch.randn(10)
        W = W / W.norm()  # Make it a unit vector
        G = torch.randn(10)
        result = hyperspherical_descent(W, G)
        # Result should also be a unit vector
        assert math.isclose(result.norm().item(), 1.0, rel_tol=1e-5)

    def test_hyperspherical_descent_custom_eta(self):
        """hyperspherical_descent should respect eta parameter."""
        W = torch.randn(10)
        G = torch.randn(10)
        result1 = hyperspherical_descent(W, G, eta=0.01)
        result2 = hyperspherical_descent(W, G, eta=1.0)
        # Different eta should give different results
        assert not torch.allclose(result1, result2)

    def test_hyperspherical_descent_eta_zero(self):
        """hyperspherical_descent with eta=0 should return normalized input."""
        W = torch.randn(10) * 5  # Non-unit vector
        G = torch.randn(10)
        result = hyperspherical_descent(W, G, eta=0.0)
        # With eta=0, result should just be normalized W
        expected = W / W.norm()
        assert torch.allclose(result, expected, atol=1e-5)

    def test_hyperspherical_descent_deterministic(self):
        """hyperspherical_descent should be deterministic."""
        torch.manual_seed(42)
        W = torch.randn(10)
        G = torch.randn(10)
        result1 = hyperspherical_descent(W, G)

        torch.manual_seed(42)
        W = torch.randn(10)
        G = torch.randn(10)
        result2 = hyperspherical_descent(W, G)

        assert torch.allclose(result1, result2)

    def test_hyperspherical_descent_gradient_descent(self):
        """Result should move in direction of negative gradient."""
        W = torch.randn(10)
        G = torch.randn(10)
        result = hyperspherical_descent(W, G, eta=1.0)
        # Direction should have negative correlation with gradient
        diff = result - W
        correlation = torch.dot(diff, G)
        # If moving in gradient direction, correlation should be positive
        # But we move in opposite direction of projection
        assert correlation < 0  # Moving against gradient

    def test_hyperspherical_descent_tensor_input(self):
        """hyperspherical_descent should accept torch tensors."""
        W = torch.randn(4, 4)
        G = torch.randn(4, 4)
        assert isinstance(W, torch.Tensor) and isinstance(G, torch.Tensor)
        result = hyperspherical_descent(W, G)
        assert isinstance(result, torch.Tensor)

    def test_hyperspherical_descent_large_eta(self):
        """hyperspherical_descent should handle large eta."""
        W = torch.randn(10)
        G = torch.randn(10)
        result = hyperspherical_descent(W, G, eta=10.0)
        assert not torch.isnan(result).any() and not torch.isinf(result).any()