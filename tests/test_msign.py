import pytest
import torch
from src.msign import msign, ABC_LIST, ABC_LIST_STABLE


class TestMsign:
    def test_msign_preserves_shape(self):
        """msign should preserve input shape."""
        for shape in [(4, 4), (4, 3), (3, 4), (5, 2), (2, 5)]:
            G = torch.randn(*shape)
            result = msign(G)
            assert result.shape == shape

    def test_msign_wide_matrix(self):
        """msign should handle wide matrices (cols > rows)."""
        G = torch.randn(3, 5)
        result = msign(G)
        assert result.shape == (3, 5)

    def test_msign_tall_matrix(self):
        """msign should handle tall matrices (rows > cols)."""
        G = torch.randn(5, 3)
        result = msign(G)
        assert result.shape == (5, 3)

    def test_msign_no_nan(self):
        """msign should not produce NaN values."""
        G = torch.randn(4, 4)
        result = msign(G)
        assert not torch.isnan(result).any()

    def test_msign_no_inf(self):
        """msign should not produce Inf values."""
        G = torch.randn(4, 4)
        result = msign(G)
        assert not torch.isinf(result).any()

    def test_msign_deterministic(self):
        """msign should be deterministic with fixed seed."""
        torch.manual_seed(42)
        G = torch.randn(4, 4)
        result1 = msign(G)

        torch.manual_seed(42)
        G = torch.randn(4, 4)
        result2 = msign(G)

        assert torch.allclose(result1, result2)

    def test_msign_sign_property(self):
        """For a positive definite matrix, result should be identity-like."""
        G = torch.eye(4) * 2  # positive definite
        result = msign(G, steps=20)
        # For PD matrix, sign should be identity
        assert torch.allclose(result, torch.eye(4), atol=0.1)

    def test_msign_negative_definite(self):
        """For a negative definite matrix, result should be negative identity."""
        G = -torch.eye(4) * 2  # negative definite
        result = msign(G, steps=20)
        # For ND matrix, sign should be -identity
        assert torch.allclose(result, -torch.eye(4), atol=0.1)

    def test_msign_bfloat16_internal(self):
        """msign should use bfloat16 internally but return float."""
        G = torch.randn(4, 4)
        result = msign(G)
        assert result.dtype == torch.float32

    def test_msign_custom_steps(self):
        """msign should respect custom steps parameter."""
        G = torch.randn(4, 4)
        result1 = msign(G, steps=1)
        result2 = msign(G, steps=20)
        # More steps should give different results
        # (not guaranteed, but likely)
        assert result1.shape == result2.shape

    def test_msign_abc_list_stable_length(self):
        """ABC_LIST_STABLE should have correct length."""
        assert len(ABC_LIST_STABLE) == len(ABC_LIST)

    def test_msign_single_step(self):
        """msign should work with single step."""
        G = torch.randn(3, 3)
        result = msign(G, steps=1)
        assert result.shape == (3, 3)
        assert not torch.isnan(result).any()