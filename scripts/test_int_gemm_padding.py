#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.gofa.int_gemm_quant import TORCH_INT_MM_M_ALIGNMENT, pad_int8_rows_to_multiple


def _parse_m_values(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Smoke test GOFA W4A8 torch._int_mm M-dimension padding.")
    parser.add_argument("--m-values", default="200,208", help="Comma-separated M values to test.")
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("torch._int_mm is unavailable in this torch build.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for torch._int_mm padding smoke test.")

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda")
    m_values = _parse_m_values(args.m_values)
    if not m_values:
        raise ValueError("--m-values must contain at least one integer.")

    for m in m_values:
        q_x = torch.randint(-127, 128, (m, args.k), device=device, dtype=torch.int8).contiguous()
        q_w_t = torch.randint(-7, 8, (args.k, args.n), device=device, dtype=torch.int8).contiguous()
        q_x_pad, original_m = pad_int8_rows_to_multiple(q_x, TORCH_INT_MM_M_ALIGNMENT)
        if original_m != m:
            raise AssertionError(f"original_m mismatch: got={original_m}, expected={m}")
        if q_x_pad.size(0) % TORCH_INT_MM_M_ALIGNMENT != 0:
            raise AssertionError(f"padded M is not {TORCH_INT_MM_M_ALIGNMENT}-aligned: {q_x_pad.size(0)}")
        if q_x_pad.size(0) > m and not torch.equal(
                q_x_pad[m:],
                torch.zeros_like(q_x_pad[m:])):
            raise AssertionError("Padded q_x rows are not zero-filled.")
        if q_x_pad.dtype != torch.int8 or q_w_t.dtype != torch.int8:
            raise AssertionError("torch._int_mm inputs must be int8.")
        if q_x_pad.device.type != "cuda" or q_w_t.device.type != "cuda":
            raise AssertionError("torch._int_mm inputs must be CUDA tensors.")
        if not q_x_pad.is_contiguous() or not q_w_t.is_contiguous():
            raise AssertionError("torch._int_mm inputs must be contiguous.")

        y_int = torch._int_mm(q_x_pad, q_w_t)
        y_int = y_int[:m].contiguous()
        if tuple(y_int.shape) != (m, args.n):
            raise AssertionError(f"Unexpected output shape for M={m}: {tuple(y_int.shape)}")
        print(
            "passed padded torch._int_mm: "
            f"M={m}, padded_M={q_x_pad.size(0)}, K={args.k}, N={args.n}, "
            f"output_shape={tuple(y_int.shape)}"
        )


if __name__ == "__main__":
    main()
