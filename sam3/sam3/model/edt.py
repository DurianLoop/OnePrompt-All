# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Euclidean distance transform helper.

The upstream implementation uses a Triton kernel. Triton wheels are generally
not available on native Windows, so this local reproduction keeps the fast
Triton path when present and falls back to OpenCV/CPU otherwise.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        k = 0
        for q in range(1, length):
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            r = tl.load(v + block_start + (k * stride))
            z_k = tl.load(z + block_start + (k * stride))
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride),
                    mask=(k + 1) < length,
                    other=q,
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)


def _edt_triton_impl(data: torch.Tensor) -> torch.Tensor:
    assert triton is not None
    assert data.dim() == 3
    assert data.is_cuda
    B, H, W = data.shape
    data = data.contiguous()
    output = torch.where(data, 1e18, 0.0)
    parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
    parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, H)
    edt_kernel[grid](
        output.clone(),
        output,
        parabola_loc,
        parabola_inter,
        H,
        W,
        horizontal=True,
    )

    parabola_loc.zero_()
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, W)
    edt_kernel[grid](
        output.clone(),
        output,
        parabola_loc,
        parabola_inter,
        H,
        W,
        horizontal=False,
    )
    return output.sqrt()


def _edt_cpu_impl(data: torch.Tensor) -> torch.Tensor:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Triton is unavailable and OpenCV is required for the Windows EDT fallback."
        ) from exc

    assert data.dim() == 3
    device = data.device
    dtype = data.dtype if data.dtype.is_floating_point else torch.float32
    cpu_data = data.detach().to("cpu").bool().numpy()
    outputs = []
    for mask in cpu_data:
        src = mask.astype(np.uint8)
        outputs.append(cv2.distanceTransform(src, cv2.DIST_L2, 0))
    stacked = np.stack(outputs, axis=0)
    return torch.from_numpy(stacked).to(device=device, dtype=dtype)


def edt_triton(data: torch.Tensor) -> torch.Tensor:
    """Compute batched Euclidean distance transform for BxHxW binary tensors."""
    if triton is not None and data.is_cuda:
        return _edt_triton_impl(data)
    return _edt_cpu_impl(data)
