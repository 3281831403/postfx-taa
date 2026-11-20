"""Temporal anti-aliasing (TAA) resolve helpers for an offline post-FX tool.

Data conventions
----------------
* Images are ``list[list[float]]`` with ``img[y][x]``, row-major: the
  number of rows is the image height and ``len(row)`` is the width.  Pixel
  values are normalized to ``0.0 .. 1.0`` unless a docstring says otherwise.
  Absolute comparisons use a tolerance of ``1e-6``.
* Motion vectors are an image-shaped ``list[list[tuple[float, float]]]``:
  ``motion[y][x] = (dx, dy)`` in units of pixels, describing where the
  pixel content of the previous frame sits relative to the current frame.
* All weights (``alpha``) and reported metrics (ghosting severity) live in
  the closed interval ``[0.0, 1.0]``.
* Functions are pure: inputs are never mutated and results depend only on
  the arguments.  There is no randomness and no I/O.
"""

from __future__ import annotations

import math

__all__ = [
    "TaaError",
    "clamp_history",
    "detect_ghosting",
    "jitter_offset",
    "neighborhood_stats",
    "reproject",
    "resolve",
    "resolve_batch",
    "validate_state",
]

TOL = 1e-6
_HALTON_PERIOD = 8
_GRID_PERIOD = 16
_REQUIRED_CONFIG_KEYS = ("alpha", "pattern", "box", "sigma")


class TaaError(ValueError):
    """Raised for malformed frames, motion fields, or resolve parameters."""


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _safe_shape(img):
    """Return ``(height, width)`` for a rectangular image, else ``None``.

    Never raises: used by the reporting-style :func:`validate_state` and by
    the upfront batch validation so that one malformed frame is reported
    instead of crashing the whole rollback check.
    """
    if not isinstance(img, list) or len(img) == 0:
        return None
    width = len(img[0])
    if not isinstance(img[0], list) or width == 0:
        return None
    for row in img:
        if not isinstance(row, list) or len(row) != width:
            return None
    return len(img), width


def _shape(img, what: str) -> tuple[int, int]:
    """Strict variant of :func:`_safe_shape` that raises :class:`TaaError`."""
    shape = _safe_shape(img)
    if shape is None:
        raise TaaError(f"{what}: empty or ragged image")
    return shape


def _require_same_shape(shape_a, shape_b, what_a: str, what_b: str) -> None:
    if shape_a != shape_b:
        raise TaaError(
            f"size mismatch: {what_a} {shape_a[1]}x{shape_a[0]} vs "
            f"{what_b} {shape_b[1]}x{shape_b[0]}"
        )


def _pixels_finite(img) -> bool:
    return all(
        isinstance(v, (int, float)) and math.isfinite(v)
        for row in img
        for v in row
    )


def _motion_finite(motion) -> bool:
    for row in motion:
        for vector in row:
            if (
                not isinstance(vector, (tuple, list))
                or len(vector) != 2
                or not math.isfinite(vector[0])
                or not math.isfinite(vector[1])
            ):
                return False
    return True


def _halton(index: int, base: int) -> float:
    """Return the (1-based) Halton sequence value of ``index`` in ``base``."""
    result = 0.0
    fraction = 1.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


# ---------------------------------------------------------------------------
# ① / ② Sub-pixel jitter sequence
# ---------------------------------------------------------------------------


def jitter_offset(
    frame_index: int,
    pattern: str,
    size: tuple[int, int],
) -> tuple[float, float]:
    """Return the sub-pixel camera jitter ``(dx, dy)`` for a frame.

    The jitter perturbs the projection by a fraction of a pixel so a static
    edge lands on different sample positions each frame and TAA can
    accumulate detail instead of repeating one sample.

    ``size`` is ``(width, height)`` in pixels.  ``pattern`` selects the
    sub-pixel sequence: ``"halton"`` uses the Halton sequence in bases 2
    and 3, ``"grid"`` uses a fixed 4x4 sub-pixel grid cycled by
    ``frame_index``.  Every offset differs from frame to frame (the
    sequence must not collapse to a constant ``(0.0, 0.0)``), and both
    components satisfy ``-1.0 < d < 1.0`` in pixel units: an amplitude
    of one pixel or more visibly shifts the picture instead of adding a
    sample.

    The returned tuple is a pair of floats; offsets for out-of-range
    ``frame_index`` values wrap around the sequence.  Comparisons use a
    tolerance of ``1e-6``.  Raises :class:`TaaError` for unknown patterns,
    non-positive sizes, or a negative frame index.
    """
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise TaaError(f"frame index must be a non-negative int, got {frame_index!r}")
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise TaaError(f"size must be (width, height), got {size!r}")
    width, height = size
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or isinstance(width, bool)
        or isinstance(height, bool)
        or width <= 0
        or height <= 0
    ):
        raise TaaError(f"size must be positive, got {size!r}")

    if pattern == "halton":
        # Cycle 8 Halton (base 2, base 3) samples and center them on 0.
        # Raw Halton values lie in (0, 1), so the centered offset lies in
        # (-0.5, 0.5): always strictly sub-pixel, at any resolution.
        index = (frame_index % _HALTON_PERIOD) + 1
        dx = _halton(index, 2) - 0.5
        dy = _halton(index, 3) - 0.5
    elif pattern == "grid":
        # 4x4 sub-pixel grid: cell centers at +/- 1/8, +/- 3/8 of a pixel.
        step = frame_index % _GRID_PERIOD
        gx = (step % 4)
        gy = (step // 4)
        dx = (gx + 0.5) / 4.0 - 0.5
        dy = (gy + 0.5) / 4.0 - 0.5
    else:
        raise TaaError(f"unknown jitter pattern: {pattern!r}")

    return (dx, dy)


# ---------------------------------------------------------------------------
# ③ / ④ Motion-vector reprojection with border clamping
# ---------------------------------------------------------------------------


def reproject(
    history: list[list[float]],
    motion: list[list[tuple[float, float]]],
    frame_index: int,
) -> list[list[float]]:
    """Resample the history frame onto the current frame's pixel grid.

    ``history`` is the previously resolved image and ``motion`` is an
    equally sized field of ``(dx, dy)`` pixel vectors.  The convention is
    reverse sampling: ``out[y][x]`` takes ``history[y - dy][x - dx]``,
    because ``(dx, dy)`` is where the current pixel's content was in the
    previous frame.  Sample coordinates are rounded to the nearest integer
    pixel.

    When a sample falls outside the image, clamp it to the nearest valid
    border pixel; never fill with ``0.0`` (black padding shows up as dark
    borders around moving edges).  The returned image has the same shape
    as ``history`` and does not alias its rows.  ``frame_index`` is
    accepted for logging/telemetry and does not change the rule.  Absolute
    comparisons use a tolerance of ``1e-6``.

    Raises :class:`TaaError` if the images are empty, their shapes differ,
    the motion field contains NaN/Inf, or the pixel values are non-finite.
    """
    height, width = _shape(history, "history")
    motion_shape = _shape(motion, "motion")
    _require_same_shape(motion_shape, (height, width), "motion", "history")
    if not _pixels_finite(history):
        raise TaaError("history contains NaN/Inf pixel values")
    if not _motion_finite(motion):
        raise TaaError("motion contains NaN/Inf vectors")

    out: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        for x in range(width):
            dx, dy = motion[y][x]
            # Reverse sampling: fetch where this pixel came from.
            sample_x = int(math.floor(x - dx + 0.5))
            sample_y = int(math.floor(y - dy + 0.5))
            # Clamp to the nearest legal pixel instead of zero-filling.
            sample_x = min(max(sample_x, 0), width - 1)
            sample_y = min(max(sample_y, 0), height - 1)
            row.append(history[sample_y][sample_x])
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# ⑤ / ⑥ Per-pixel local variance clamping
# ---------------------------------------------------------------------------


def clamp_history(
    current: list[list[float]],
    history: list[list[float]],
    box: int,
    sigma: float,
) -> list[list[float]]:
    """Variance-clamp the history to the local neighborhood of ``current``.

    This is the anti-ghosting step: history that no longer matches the
    current neighborhood (for example, the stale background left behind a
    moving object) is pulled back into the plausible range.

    For every pixel, the window is ``(2*box + 1) x (2*box + 1)`` around
    that pixel (clamped at the image border).  Compute the window's mean
    ``mean`` and population standard deviation ``local_sigma`` from
    ``current``, then::

        lo = mean - sigma * local_sigma
        hi = mean + sigma * local_sigma
        out[y][x] = min(max(history[y][x], lo), hi)

    The statistics are **local, per pixel**: a whole-frame sigma is
    inflated by a large moving object and then either fails to remove the
    ghost or clips flat static regions.  If ``local_sigma`` is ``0`` the
    accepted range degenerates to ``[mean, mean]``.  The output has the
    same shape as ``history`` and pixel values stay in ``0.0 .. 1.0``;
    tolerance is ``1e-6``.

    Raises :class:`TaaError` if either image is empty, the shapes differ,
    ``box`` is negative, or ``sigma`` is negative/non-finite.
    """
    height, width = _shape(current, "current")
    history_shape = _shape(history, "history")
    _require_same_shape(history_shape, (height, width), "history", "current")
    if not isinstance(box, int) or isinstance(box, bool) or box < 0:
        raise TaaError(f"box must be a non-negative int, got {box!r}")
    if not isinstance(sigma, (int, float)) or isinstance(sigma, bool) or not math.isfinite(sigma):
        raise TaaError(f"sigma must be finite, got {sigma!r}")
    if sigma < 0.0:
        raise TaaError(f"sigma must be non-negative, got {sigma!r}")
    if not _pixels_finite(current):
        raise TaaError("current contains NaN/Inf pixel values")
    if not _pixels_finite(history):
        raise TaaError("history contains NaN/Inf pixel values")

    out: list[list[float]] = []
    for y in range(height):
        y0 = max(0, y - box)
        y1 = min(height, y + box + 1)
        row: list[float] = []
        for x in range(width):
            x0 = max(0, x - box)
            x1 = min(width, x + box + 1)

            total = 0.0
            count = 0
            for ny in range(y0, y1):
                current_row = current[ny]
                for nx in range(x0, x1):
                    total += current_row[nx]
                    count += 1
            mean = total / count
            variance = 0.0
            for ny in range(y0, y1):
                current_row = current[ny]
                for nx in range(x0, x1):
                    delta = current_row[nx] - mean
                    variance += delta * delta
            local_sigma = math.sqrt(variance / count)

            lo = mean - sigma * local_sigma
            hi = mean + sigma * local_sigma
            value = history[y][x]
            row.append(min(max(value, lo), hi))
        out.append(row)
    return out


def neighborhood_stats(img: list[list[float]], radius: int) -> dict:
    """Return statistics of one square neighborhood of ``img``.

    The window is ``(2*radius + 1) x (2*radius + 1)`` centered on the
    image's center pixel ``(height // 2, width // 2)``.  Pixels outside the
    image are clipped, not zero-padded, so border windows average only the
    samples that exist.  This is the building block for the per-pixel local
    windows used by variance clamping.

    Returns a dict with exactly these keys:

    * ``"min"`` -- minimum pixel value in the window.
    * ``"max"`` -- maximum pixel value in the window.
    * ``"mean"`` -- arithmetic mean of the window pixels.
    * ``"sigma"`` -- population standard deviation of the window (divide
      by ``n``, not ``n - 1``), in the same unit as the pixel values.

    Pixel values are ``0.0 .. 1.0`` floats and comparisons use a tolerance
    of ``1e-6``.  ``radius`` is a non-negative integer.  Raises
    :class:`TaaError` for an empty or ragged image, or a negative radius.
    """
    height, width = _shape(img, "img")
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise TaaError(f"radius must be a non-negative int, got {radius!r}")
    if not _pixels_finite(img):
        raise TaaError("img contains NaN/Inf pixel values")

    center_y = height // 2
    center_x = width // 2
    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)

    values = [img[y][x] for y in range(y0, y1) for x in range(x0, x1)]
    count = len(values)
    mean = sum(values) / count
    variance = sum((v - mean) * (v - mean) for v in values) / count
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "sigma": math.sqrt(variance),
    }


# ---------------------------------------------------------------------------
# ⑦ Motion-adaptive history blend
# ---------------------------------------------------------------------------


def resolve(
    current: list[list[float]],
    history: list[list[float]],
    motion: list[list[tuple[float, float]]],
    alpha: float,
) -> list[list[float]]:
    """Blend the current frame with the history frame into a resolved frame.

    ``alpha`` is the maximum history weight and must lie in ``[0.0, 1.0]``
    (out-of-range values raise :class:`TaaError`).  For each pixel the
    history weight adapts to the motion magnitude ``m = hypot(dx, dy)``::

        w = clamp(alpha / (1.0 + m), 0.0, 1.0)
        out[y][x] = (1.0 - w) * current[y][x] + w * history[y][x]

    A static pixel (``m == 0``) keeps the full weight ``w = alpha`` so the
    still image stops shimmering; a fast pixel gets ``w -> 0`` so the
    current frame wins and no smear/trail is left behind.  Consequences
    that callers rely on: ``alpha == 0`` returns ``current`` exactly, and
    with zero motion ``alpha == 1`` returns ``history`` exactly.  Results
    stay in ``0.0 .. 1.0`` within a ``1e-6`` tolerance.

    All three fields must agree in shape.  Raises :class:`TaaError` for
    empty images, mismatched shapes, NaN/Inf values, or an alpha outside
    ``[0.0, 1.0]``; no partial output is returned.
    """
    height, width = _shape(current, "current")
    history_shape = _shape(history, "history")
    _require_same_shape(history_shape, (height, width), "history", "current")
    motion_shape = _shape(motion, "motion")
    _require_same_shape(motion_shape, (height, width), "motion", "current")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha):
        raise TaaError(f"alpha must be finite, got {alpha!r}")
    if alpha < 0.0 or alpha > 1.0:
        raise TaaError(f"alpha out of range: {alpha!r}")
    if not _pixels_finite(current):
        raise TaaError("current contains NaN/Inf pixel values")
    if not _pixels_finite(history):
        raise TaaError("history contains NaN/Inf pixel values")
    if not _motion_finite(motion):
        raise TaaError("motion contains NaN/Inf vectors")

    out: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        for x in range(width):
            dx, dy = motion[y][x]
            motion_magnitude = math.hypot(dx, dy)
            weight = alpha / (1.0 + motion_magnitude)
            weight = min(max(weight, 0.0), 1.0)
            row.append(
                (1.0 - weight) * current[y][x] + weight * history[y][x]
            )
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# ⑧ Local ghosting metric
# ---------------------------------------------------------------------------


def detect_ghosting(
    current: list[list[float]],
    resolved: list[list[float]],
    threshold: float,
) -> float:
    """Measure visible ghosting between ``current`` and ``resolved``.

    Ghosting is the local leftover of the history: a moving object leaves
    a patch whose values no longer match the current frame.  A whole-frame
    average dilutes a small patch to nearly zero, so the metric is local:

    1. Per pixel, the effective residual is
       ``d[y][x] = max(0.0, abs(current[y][x] - resolved[y][x]) - threshold)``
       where ``threshold`` (``>= 0``) is the noise floor.
    2. For each pixel take the mean of ``d`` over its clipped 3x3
       neighborhood.
    3. ``severity = min(1.0, max(neighborhood_mean))`` -- a float in
       ``[0.0, 1.0]``.

    A small but strong local ghost therefore reports close to ``1.0``
    instead of being averaged away.  The inputs are pixel images
    (``0.0 .. 1.0``, tolerance ``1e-6``) and the result is a plain float.
    Raises :class:`TaaError` for empty/ragged/mismatched images, NaN/Inf
    values, or a negative threshold.
    """
    height, width = _shape(current, "current")
    resolved_shape = _shape(resolved, "resolved")
    _require_same_shape(resolved_shape, (height, width), "resolved", "current")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold):
        raise TaaError(f"threshold must be finite, got {threshold!r}")
    if threshold < 0.0:
        raise TaaError(f"threshold must be non-negative, got {threshold!r}")
    if not _pixels_finite(current):
        raise TaaError("current contains NaN/Inf pixel values")
    if not _pixels_finite(resolved):
        raise TaaError("resolved contains NaN/Inf pixel values")

    residual = [
        [max(0.0, abs(current[y][x] - resolved[y][x]) - threshold) for x in range(width)]
        for y in range(height)
    ]

    worst = 0.0
    for y in range(height):
        y0 = max(0, y - 1)
        y1 = min(height, y + 2)
        for x in range(width):
            x0 = max(0, x - 1)
            x1 = min(width, x + 2)
            total = 0.0
            count = 0
            for ny in range(y0, y1):
                for nx in range(x0, x1):
                    total += residual[ny][nx]
                    count += 1
            local_mean = total / count
            if local_mean > worst:
                worst = local_mean

    return min(1.0, worst)


# ---------------------------------------------------------------------------
# ⑩ Reporting-style state validation
# ---------------------------------------------------------------------------


def validate_state(
    history: list[list[float]],
    current: list[list[float]],
    alpha: float | None = None,
) -> list[str]:
    """Return a list of human-readable problems, empty when the state is OK.

    This function reports instead of raising, so a caller can decide
    whether to abort or roll back a batch.  It checks:

    * shape -- either image is empty/ragged, or ``history`` and ``current``
      have different dimensions (width/height);
    * values -- any pixel is NaN or infinite;
    * if ``alpha`` is given (optional) -- it is outside ``[0.0, 1.0]``.

    Messages are short strings (for example ``"size mismatch: ..."`` or
    ``"alpha out of range: ..."``), one entry per distinct problem, and
    images with valid shapes are not indexed before the size check passes.
    Returns ``[]`` when no problem is found.  Inputs are treated as
    ``list[list[float]]``, ``img[y][x]``.
    """
    issues: list[str] = []
    history_shape = _safe_shape(history)
    current_shape = _safe_shape(current)

    if history_shape is None or current_shape is None or history_shape != current_shape:
        issues.append(
            f"size mismatch: history {history_shape} vs current {current_shape}"
        )
    elif not _pixels_finite(history) or not _pixels_finite(current):
        # Only index pixels once the shape check has passed.
        issues.append("nan or non-finite pixel value in history/current")

    if alpha is not None:
        if (
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or not math.isfinite(alpha)
            or alpha < 0.0
            or alpha > 1.0
        ):
            issues.append(f"alpha out of range: {alpha}")

    return issues


# ---------------------------------------------------------------------------
# ⑨ Serial batch resolve with per-frame history and atomic rollback
# ---------------------------------------------------------------------------


def resolve_batch(
    frames: list[list[list[float]]],
    motions: list[list[list[tuple[float, float]]]],
    config: dict,
) -> dict:
    """Resolve a whole frame sequence, one frame at a time.

    ``frames`` is a list of pixel images and ``motions`` is an equally long
    list of motion fields (one per frame, each matching its frame's shape).
    ``config`` must contain at least ``"alpha"``, ``"pattern"``, ``"box"``,
    and ``"sigma"``; the jitter for frame ``i`` comes from
    ``jitter_offset(i, config["pattern"], (width, height))``.

    Frames are resolved in order.  Frame 0 has no history, so its history
    is the frame itself.  Afterwards each frame's only history is the
    previously resolved frame -- frame buffers are not shared, reused
    across frames, or written back into the input list.

    On success the return value is::

        {"frames": [...], "ghosting": [...], "issues": [], "rolled_back": False}

    ``frames`` keeps the input order and has one resolved image per input
    frame; ``ghosting`` holds one ``detect_ghosting`` severity per frame.

    If any frame is invalid (empty, size mismatch, NaN/Inf, alpha out of
    range, ...) the whole batch rolls back atomically: nothing is
    partially committed, and the call returns
    ``{"frames": [], "ghosting": [], "issues": [...], "rolled_back": True}``
    with the collected problem messages.  Images and weights are normalized
    as in the single-frame helpers and comparisons use a ``1e-6``
    tolerance.  Raises :class:`TaaError` if ``frames`` and ``motions`` have
    different lengths or ``config`` is missing a required key.
    """
    if not isinstance(config, dict):
        raise TaaError("config must be a dict")
    missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise TaaError(f"config missing required keys: {missing}")
    if not isinstance(frames, list) or not isinstance(motions, list):
        raise TaaError("frames and motions must be lists")
    if len(frames) != len(motions):
        raise TaaError(
            f"frames/motions length mismatch: {len(frames)} vs {len(motions)}"
        )

    alpha = config["alpha"]
    pattern = config["pattern"]
    box = config["box"]
    sigma = config["sigma"]
    issues: list[str] = []

    if (
        not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or not math.isfinite(alpha)
        or alpha < 0.0
        or alpha > 1.0
    ):
        issues.append(f"alpha out of range: {alpha}")
    if not isinstance(box, int) or isinstance(box, bool) or box < 0:
        issues.append(f"box out of range: {box}")
    if (
        not isinstance(sigma, (int, float))
        or isinstance(sigma, bool)
        or not math.isfinite(sigma)
        or sigma < 0.0
    ):
        issues.append(f"sigma out of range: {sigma}")

    # Upfront validation of every frame: nothing is committed until the
    # whole sequence is known to be legal.
    base_shape: tuple[int, int] | None = None
    for index, frame in enumerate(frames):
        frame_shape = _safe_shape(frame)
        if frame_shape is None:
            issues.append(f"frame {index}: empty or ragged image")
            continue
        if base_shape is None:
            base_shape = frame_shape
        if frame_shape != base_shape:
            issues.append(
                f"frame {index}: size mismatch {frame_shape[1]}x{frame_shape[0]} "
                f"vs base {base_shape[1]}x{base_shape[0]}"
            )
        elif not _pixels_finite(frame):
            issues.append(f"frame {index}: nan or non-finite pixel value")

        motion = motions[index]
        motion_shape = _safe_shape(motion)
        if motion_shape != frame_shape:
            issues.append(f"frame {index}: motion size mismatch")
        elif not _motion_finite(motion):
            issues.append(f"frame {index}: motion contains nan/inf vectors")

    if base_shape is not None:
        width, height = base_shape[1], base_shape[0]
        for index in range(len(frames)):
            try:
                jitter_offset(index, pattern, (width, height))
            except TaaError as exc:
                issues.append(f"frame {index}: {exc}")
                break

    if issues:
        return {"frames": [], "ghosting": [], "issues": issues, "rolled_back": True}

    if not frames:
        return {"frames": [], "ghosting": [], "issues": [], "rolled_back": False}

    width, height = base_shape[1], base_shape[0]
    resolved_frames: list[list[list[float]]] = []
    ghosting: list[float] = []

    # Frame 0 has no history: the frame itself is its own history.  Deep
    # copy so the output never aliases or writes back into the input list.
    history = [row[:] for row in frames[0]]
    first = [row[:] for row in history]
    resolved_frames.append(first)
    ghosting.append(detect_ghosting(frames[0], first, 0.0))

    for index in range(1, len(frames)):
        # The jitter drives the camera sample position; asking for it here
        # also guarantees the configured pattern is valid for every frame.
        jitter_offset(index, pattern, (width, height))
        reprojected = reproject(history, motions[index], index)
        resolved = resolve(frames[index], reprojected, motions[index], alpha)
        resolved_frames.append(resolved)
        ghosting.append(detect_ghosting(frames[index], resolved, 0.0))
        # Independent history buffer: only the previous resolved frame.
        history = resolved

    return {
        "frames": resolved_frames,
        "ghosting": ghosting,
        "issues": [],
        "rolled_back": False,
    }
