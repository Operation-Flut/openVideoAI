"""Overlap-add tiled inference for image-restoration pipelines.

Large images are split into overlapping tiles, each tile is run through the
model independently, and the results are blended with a 2D tapering window.
That keeps memory bounded and blends away the model's inference border
artifacts instead of hard-cutting tile seams into the output.

A reflected margin of one ``overlap`` is added before tiling so border and
corner pixels always sit inside a neighbouring tile's window interior; the
Hamming default only decays to ~0.08 at the tile borders (instead of the exact
0.0 of a Hann window), which keeps the weight normalisation stable.
"""
import cv2
import numpy as np
from typing import Callable, Optional


def _window(kind: str, height: int, width: int) -> np.ndarray:
    """2D tapering window as an ``(h, w, 1)`` float32 array."""
    if kind == 'hann':
        wy = np.hanning(height)
        wx = np.hanning(width)
    else:
        wy = np.hamming(height)
        wx = np.hamming(width)
    return np.outer(wy, wx)[..., None].astype(np.float32)


def _to_multiple_of_16(tile_size: int) -> int:
    """Tiles are aligned to a 16-pixel grid (the smallest common model stride)."""
    return max(256, (int(tile_size) // 16) * 16)


def process_tiled(
    image: np.ndarray,
    run_model: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int,
    overlap: int,
    window: str = 'hamming',
    progress: Optional[Callable[[float, str], None]] = None,
) -> np.ndarray:
    """Run ``run_model`` over ``image`` in overlapping tiles and blend the result.

    ``run_model`` receives an RGB uint8 tile and must return an RGB uint8 tile of
    the same shape. Returns an RGB uint8 image of the original size.
    """
    if window not in ('hann', 'hamming'):
        raise ValueError(f"window must be 'hann' or 'hamming', got {window!r}")

    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    tile_size = _to_multiple_of_16(tile_size)
    # Keep the stride at least half a tile so tiling always terminates.
    overlap = max(0, min(int(overlap), tile_size // 2))
    stride = tile_size - overlap

    padded = cv2.copyMakeBorder(image, overlap, overlap, overlap, overlap, cv2.BORDER_REFLECT_101)
    padded_h, padded_w = padded.shape[:2]

    output = np.zeros((h, w, channels), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)

    x_coords = list(range(0, padded_w, stride))
    y_coords = list(range(0, padded_h, stride))
    total_tiles = max(1, len(x_coords) * len(y_coords))
    tile_count = 0

    for yp in y_coords:
        for xp in x_coords:
            x_end = min(xp + tile_size, padded_w)
            y_end = min(yp + tile_size, padded_h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)

            tile = padded[y_start:y_end, x_start:x_end]
            tile_h, tile_w = tile.shape[:2]

            out_tile = np.asarray(run_model(tile)).astype(np.float32)
            win = _window(window, tile_h, tile_w)

            out_y0 = max(y_start, overlap)
            out_y1 = min(y_end, overlap + h)
            out_x0 = max(x_start, overlap)
            out_x1 = min(x_end, overlap + w)
            if out_y0 >= out_y1 or out_x0 >= out_x1:
                continue

            output[out_y0 - overlap: out_y1 - overlap,
                   out_x0 - overlap: out_x1 - overlap] += (
                out_tile[out_y0 - y_start: out_y1 - y_start,
                         out_x0 - x_start: out_x1 - x_start]
                * win[out_y0 - y_start: out_y1 - y_start,
                      out_x0 - x_start: out_x1 - x_start]
            )
            weight[out_y0 - overlap: out_y1 - overlap,
                   out_x0 - overlap: out_x1 - overlap] += (
                win[out_y0 - y_start: out_y1 - y_start,
                    out_x0 - x_start: out_x1 - x_start]
            )

            tile_count += 1
            if progress is not None:
                percent = 30 + (tile_count / total_tiles) * 50
                progress(percent, f"Processing tile {tile_count}/{total_tiles}")

    weight = np.maximum(weight, 1e-8)
    output = output / weight
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)
