"""Red tests for the TAA resolve exercise.

Every test calls at least one public function from ``taa.resolve``.  The
stubs raise ``NotImplementedError`` until the module is implemented, so all
15 tests fail at runtime without collection, import, or syntax errors.

Images are 2-D ``list[list[float]]`` with ``img[y][x]`` and pixel values in
``0.0 .. 1.0``; motion fields store ``(dx, dy)`` in pixels.  Absolute
comparisons use a tolerance of ``1e-6``.
"""

import math

import pytest

from taa.resolve import (
    TaaError,
    clamp_history,
    detect_ghosting,
    jitter_offset,
    neighborhood_stats,
    reproject,
    resolve as resolve_frame,
    resolve_batch,
    validate_state,
)

TOL = 1e-6


def make_image(width, height, value=0.0):
    """Return a ``height`` x ``width`` image filled with ``value``."""
    return [[float(value) for _ in range(width)] for _ in range(height)]


def copy_image(img):
    """Return a shallow copy of the rows so tests can mutate a clone."""
    return [row[:] for row in img]


def gradient_image(width, height):
    """Return an image with unique, normalized values to detect shifts."""
    total = float(width * height)
    return [[(y * width + x) / total for x in range(width)] for y in range(height)]


def zero_motion(width, height):
    """Return a zero motion-vector field matching a ``width`` x ``height`` image."""
    return [[(0.0, 0.0) for _ in range(width)] for _ in range(height)]


def test_defect_01_jitter_sequence_is_not_constant():
    """Every frame needs a different sub-pixel offset; (0, 0) disables TAA."""
    size = (64, 48)
    offsets = [jitter_offset(frame, "halton", size) for frame in range(4)]

    for dx, dy in offsets:
        assert -1.0 < dx < 1.0
        assert -1.0 < dy < 1.0
    assert len(set(offsets)) > 1, "相邻帧抖动偏移相同，TAA 无法累积新采样"
    assert any(offset != (0.0, 0.0) for offset in offsets), "抖动序列恒为 (0, 0)"


def test_defect_02_jitter_amplitude_stays_subpixel():
    """Offsets must be normalized to fractions of a pixel at any resolution."""
    for size in ((2, 2), (7, 5), (16, 16), (1920, 1080), (4096, 2160)):
        for pattern in ("halton", "grid"):
            for frame in range(8):
                dx, dy = jitter_offset(frame, pattern, size)
                assert -1.0 < dx < 1.0, f"{pattern} {size} frame={frame}: dx={dx}"
                assert -1.0 < dy < 1.0, f"{pattern} {size} frame={frame}: dy={dy}"


def test_defect_03_reproject_follows_motion_vectors():
    """A moving pixel must be fetched from its previous position."""
    history = gradient_image(8, 6)
    motion = [[(1.0, 0.0) for _ in range(8)] for _ in range(6)]
    shifted = reproject(history, motion, 0)

    assert len(shifted) == 6
    assert all(len(row) == 8 for row in shifted)
    # 运动 (1, 0) 表示内容向右移动了一个像素，因此反向采样历史左侧邻居。
    assert shifted[3][5] == pytest.approx(history[3][4], abs=TOL)
    assert abs(shifted[3][5] - history[3][5]) > TOL
    # 左边界 clamp 到第 0 列。
    assert shifted[3][0] == pytest.approx(history[3][0], abs=TOL)


def test_defect_04_reproject_clamps_at_borders():
    """Sampling outside the history must clamp, never fill with black."""
    history = make_image(6, 6, 4.0)
    motion = [[(-3.0, 2.0) for _ in range(6)] for _ in range(6)]
    reprojected = reproject(history, motion, 0)

    assert len(reprojected) == 6
    for row in reprojected:
        assert len(row) == 6
    for y in range(6):
        for x in range(6):
            assert reprojected[y][x] == pytest.approx(4.0, abs=TOL), (
                f"({x}, {y}) 出现黑边，边界未 clamp"
            )


def test_defect_05_history_clamp_rejects_outliers():
    """Stale history far outside the local range must be pulled back in."""
    current = make_image(5, 5, 1.0)
    history = make_image(5, 5, 1.0)
    history[2][2] = 0.0
    clamped = clamp_history(current, history, box=2, sigma=2.5)

    assert clamped[2][2] == pytest.approx(1.0, abs=TOL)
    assert clamped[2][2] >= 1.0 - TOL
    assert clamped[0][0] == pytest.approx(1.0, abs=TOL)


def test_defect_06_history_clamp_uses_local_sigma():
    """A globally inflated sigma must not corrupt an otherwise static corner."""
    current = make_image(8, 8, 0.0)
    for y in range(6, 8):
        for x in range(6, 8):
            current[y][x] = 0.5
    history = copy_image(current)

    window = make_image(3, 3, 0.0)
    window[1][1] = 1.0
    stats = neighborhood_stats(window, radius=1)
    assert set(stats) == {"min", "max", "mean", "sigma"}
    assert stats["min"] == pytest.approx(0.0, abs=TOL)
    assert stats["max"] == pytest.approx(1.0, abs=TOL)
    assert stats["mean"] == pytest.approx(1.0 / 9.0, abs=TOL)
    assert stats["sigma"] == pytest.approx(math.sqrt(8.0 / 81.0), abs=TOL)

    clamped = clamp_history(current, history, box=1, sigma=1.5)

    for y in range(6, 8):
        for x in range(6, 8):
            assert clamped[y][x] == pytest.approx(0.5, abs=TOL), (
                f"({x}, {y}) 静止区被全局 sigma 误裁"
            )
    assert clamped[0][0] == pytest.approx(0.0, abs=TOL)


def test_defect_07_resolve_weights_adapt_to_motion():
    """Static pixels keep more history than fast-moving pixels."""
    current = make_image(6, 6, 0.0)
    history = make_image(6, 6, 1.0)
    still = resolve_frame(current, history, zero_motion(6, 6), alpha=0.5)
    moving_motion = [[(4.0, 0.0) for _ in range(6)] for _ in range(6)]
    moving = resolve_frame(current, history, moving_motion, alpha=0.5)

    assert still[3][3] == pytest.approx(0.5, abs=TOL)
    assert moving[3][3] == pytest.approx(0.1, abs=TOL)
    assert moving[3][3] < still[3][3] - 0.05
    assert 0.0 <= moving[3][3] <= 1.0


def test_defect_08_ghosting_metric_detects_local_ghosts():
    """A small local ghost must not be diluted by the whole-frame average."""
    current = make_image(16, 16, 0.0)
    resolved = make_image(16, 16, 0.0)
    for y in range(6, 10):
        for x in range(6, 10):
            resolved[y][x] = 1.0

    severity = detect_ghosting(current, resolved, threshold=0.1)

    assert 0.0 <= severity <= 1.0
    assert severity >= 0.5, "局部残影被全帧平均淹没"


def test_defect_09_batch_isolates_history_between_frames():
    """Each frame may only see the previous resolved frame as history."""
    frames = [make_image(4, 4, float(index) / 4.0) for index in range(3)]
    motions = [zero_motion(4, 4) for _ in range(3)]
    config = {"alpha": 0.5, "pattern": "halton", "box": 1, "sigma": 1.5}

    result = resolve_batch(frames, motions, config)

    assert result["rolled_back"] is False
    assert result["issues"] == []
    assert len(result["frames"]) == 3
    assert len(result["ghosting"]) == 3
    # 第 0 帧没有历史：结果就是它自己。
    assert result["frames"][0][2][2] == pytest.approx(0.0, abs=TOL)
    # 之后每帧只与上一帧的解析结果混合；共用缓冲会把这里串成别的值。
    assert result["frames"][1][2][2] == pytest.approx(0.125, abs=TOL)
    assert result["frames"][2][2][2] == pytest.approx(0.3125, abs=TOL)


def test_defect_10_size_mismatch_is_reported():
    """Mismatched frames must be reported, not indexed out of bounds."""
    history = make_image(4, 4, 0.0)
    current = make_image(6, 5, 1.0)

    issues = validate_state(history, current)
    assert issues, "尺寸不一致却返回空问题列表"
    assert any("尺寸" in issue or "size" in issue.lower() for issue in issues)

    bad = copy_image(current)
    bad[0][0] = float("nan")
    nan_issues = validate_state(current, bad)
    assert any("nan" in issue.lower() for issue in nan_issues), "NaN 未被报告"

    alpha_issues = validate_state(current, current, alpha=1.5)
    assert any("alpha" in issue.lower() for issue in alpha_issues), "alpha 越界未被报告"

    with pytest.raises(TaaError):
        resolve_frame(current, history, zero_motion(6, 5), alpha=0.5)


def test_alpha_zero_returns_current_frame():
    """alpha = 0 means the history has no weight at all."""
    current = gradient_image(5, 4)
    history = make_image(5, 4, 1.0)

    resolved = resolve_frame(current, history, zero_motion(5, 4), alpha=0.0)

    for y in range(4):
        for x in range(5):
            assert resolved[y][x] == pytest.approx(current[y][x], abs=TOL)


def test_alpha_one_returns_history_in_static_regions():
    """alpha = 1 with zero motion keeps the history exactly."""
    current = make_image(5, 4, 0.25)
    history = gradient_image(5, 4)

    resolved = resolve_frame(current, history, zero_motion(5, 4), alpha=1.0)

    for y in range(4):
        for x in range(5):
            assert resolved[y][x] == pytest.approx(history[y][x], abs=TOL)


def test_empty_frame_raises_taa_error():
    """An empty frame is not a valid image and must raise TaaError."""
    with pytest.raises(TaaError):
        resolve_frame([], [], [], alpha=0.5)


def test_jitter_offsets_are_subpixel():
    """The jitter stays inside one pixel for every supported pattern."""
    for pattern in ("halton", "grid"):
        for frame in range(6):
            offset = jitter_offset(frame, pattern, (32, 32))
            assert isinstance(offset, tuple)
            assert len(offset) == 2
            dx, dy = offset
            assert -1.0 < dx < 1.0
            assert -1.0 < dy < 1.0


def test_batch_success_preserves_frame_order():
    """Successful batches keep the input order and return per-frame metrics."""
    frames = [make_image(4, 4, float(index) / 4.0) for index in range(3)]
    motions = [zero_motion(4, 4) for _ in range(3)]
    config = {"alpha": 0.0, "pattern": "halton", "box": 1, "sigma": 1.5}

    result = resolve_batch(frames, motions, config)

    assert result["rolled_back"] is False
    assert result["issues"] == []
    assert set(result) >= {"frames", "ghosting", "issues", "rolled_back"}
    assert len(result["frames"]) == 3
    assert len(result["ghosting"]) == 3
    for index in range(3):
        for y in range(4):
            for x in range(4):
                assert result["frames"][index][y][x] == pytest.approx(
                    frames[index][y][x], abs=TOL
                )
    assert result["frames"][0][0][0] == pytest.approx(0.0, abs=TOL)
    assert result["frames"][1][0][0] == pytest.approx(0.25, abs=TOL)
    assert result["frames"][2][0][0] == pytest.approx(0.5, abs=TOL)
