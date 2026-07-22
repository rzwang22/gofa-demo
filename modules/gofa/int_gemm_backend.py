SUPPORTED_INT_GEMM_WEIGHT_BITS = {4, 8}


def normalize_weight_bits(bits):
    bits = int(bits)
    if bits not in SUPPORTED_INT_GEMM_WEIGHT_BITS:
        raise ValueError("scheme_b_int_gemm.weight_bits supports {4, 8}.")
    return bits


def weight_qmax(bits):
    return 7 if normalize_weight_bits(bits) == 4 else 127


def execute_int_mm(int_mm, lhs, rhs):
    """Small injectable boundary used by production torch._int_mm and CPU tests."""
    return int_mm(lhs, rhs)
