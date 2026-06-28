import pytest


def test_int2_quantize_dequantize_preserves_shape_dtype_and_finite_values():
    torch = pytest.importorskip("torch")
    from modules.gofa.cache_quant import dequantize_tensor, quantize_tensor

    torch.manual_seed(0)
    tensor = torch.randn(3, 5, 7, dtype=torch.float32)

    payload = quantize_tensor(tensor, bits=2, channel_axis=-1)
    reconstructed = dequantize_tensor(payload)

    assert payload["bits"] == 2
    assert payload["packed"] is True
    assert payload["pack_bits"] == 2
    assert reconstructed.shape == tensor.shape
    assert reconstructed.dtype == tensor.dtype
    assert torch.isfinite(reconstructed).all()
