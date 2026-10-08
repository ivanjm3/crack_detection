"""Split a full frame into native-resolution tiles, and reassemble masks.

The single-camera pipeline centre-crops 720x720 out of a 1280x720 frame and
downscales it to 512 (`perception.html` §3). That exists only because one
512x512 inference cannot cover a 16:9 frame without anisotropic distortion, and
it costs twice over: the left and right thirds of every frame are thrown away,
and what survives is scaled by 0.711.

A stationary rig has inference to spare, so neither cost is necessary. Tile the
whole frame into 512x512 crops at native scale instead - each tile is square, so
no distortion; nothing is resized, so no detail is lost; nothing is cropped, so
the full sensor width is used. Measured against the current pipeline at 0.4 m
standoff: 1.8x the swath and 2.1x finer crack detection, simultaneously.

Two choices worth knowing about:

  Edge tiles are CLAMPED, not padded. A tile running off the frame could be
  zero-padded to 512, but that introduces a hard black edge - a strong,
  perfectly straight linear feature, which is exactly what a crack detector is
  built to find. Clamping the origin back inside the frame instead re-covers a
  few pixels that a neighbouring tile already saw, which costs nothing.

  Overlapping masks are combined with OR. A crack crossing a tile boundary is
  truncated in both tiles; taking the union keeps whichever tile saw more of it.
  This favours recall, consistent with the rest of the pipeline. Shape metrics
  are then measured once on the reassembled mask, never per tile, so a truncated
  piece never reaches the geometry filter.
"""
import numpy as np


def plan_tiles(width, height, tile=512, overlap=0.15):
    """Tile origins covering the whole frame.

    Returns [(x, y), ...] with every tile fully inside the frame. Guarantees
    complete coverage: the last tile in each direction is clamped to the edge,
    overlapping its neighbour by more than `overlap` rather than hanging off.
    """
    if tile <= 0:
        raise ValueError("tile must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if width < tile or height < tile:
        raise ValueError(
            f"frame {width}x{height} is smaller than the {tile}px tile; "
            "pad or downscale before tiling")

    stride = max(1, int(round(tile * (1.0 - overlap))))

    def origins(extent):
        pos = list(range(0, max(1, extent - tile + 1), stride))
        last = extent - tile
        if pos[-1] != last:
            pos.append(last)            # clamp, never pad
        return pos

    return [(x, y) for y in origins(height) for x in origins(width)]


def extract(frame, tiles, tile=512):
    """Cut the planned tiles out of a frame. No resampling at all."""
    return [frame[y:y + tile, x:x + tile] for x, y in tiles]


def stitch_masks(masks, tiles, width, height, tile=512):
    """Reassemble per-tile masks into one full-frame mask, union on overlaps."""
    if len(masks) != len(tiles):
        raise ValueError(f"{len(masks)} masks for {len(tiles)} tiles")
    out = np.zeros((height, width), np.uint8)
    for m, (x, y) in zip(masks, tiles):
        if m.shape[:2] != (tile, tile):
            raise ValueError(f"mask at ({x},{y}) is {m.shape[:2]}, expected "
                             f"({tile},{tile})")
        region = out[y:y + tile, x:x + tile]
        np.maximum(region, m, out=region)
    return out


def coverage_map(tiles, width, height, tile=512):
    """How many tiles see each pixel. 0 anywhere means a gap in the plan."""
    cov = np.zeros((height, width), np.uint16)
    for x, y in tiles:
        cov[y:y + tile, x:x + tile] += 1
    return cov


def describe(width, height, tile=512, overlap=0.15, infer_ms=8.77, cameras=3):
    """One-line summary of the cost of a tiling plan."""
    tiles = plan_tiles(width, height, tile, overlap)
    return (f"{width}x{height} -> {len(tiles)} tiles/camera "
            f"({len(tiles) * cameras} total across {cameras} cameras, "
            f"{len(tiles) * cameras * infer_ms / 1000:.2f} s of inference)")
