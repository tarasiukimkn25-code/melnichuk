"""Lab 15: exhaustive block matching and motion compensation.

The two core routines are search_block() and compensate().  They use the
assignment's 16x16 blocks and +/-16-pixel search window.  Run with Python 3,
NumPy, Pillow, Matplotlib and ffmpeg available on PATH.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
BS = 16
SR = 16
W, H = 640, 352
SCENES = ("нерухома камера", "рівна панорама", "тремтлива камера", "багато руху")


def gray(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB or grayscale frame to uint8 luma."""
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame.astype(np.uint8, copy=False)
    rgb = frame[..., :3].astype(np.float32)
    return np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0, 255).astype(np.uint8)


def search_block(prev: np.ndarray, cur: np.ndarray, bx: int, by: int) -> tuple[int, int, int]:
    """Return (dy, dx, minimum SAD) for a block in cur at (bx, by).

    Full search is intentionally simple: every candidate is checked, but the
    pixel arithmetic is vectorized by NumPy.  Bounds are checked on all four
    sides so truncated slices can never masquerade as valid matches.
    """
    prev = gray(prev)
    cur = gray(cur)
    h, w = cur.shape
    blk = cur[by:by + BS, bx:bx + BS].astype(np.int16)
    best_dy, best_dx = 0, 0
    best_sad = int(np.abs(blk - prev[by:by + BS, bx:bx + BS].astype(np.int16)).sum())

    for dy in range(-SR, SR + 1):
        for dx in range(-SR, SR + 1):
            y, x = by + dy, bx + dx
            if y < 0 or x < 0 or y + BS > h or x + BS > w:
                continue
            cand = prev[y:y + BS, x:x + BS].astype(np.int16)
            sad = int(np.abs(blk - cand).sum())
            if sad < best_sad:
                best_dy, best_dx, best_sad = dy, dx, sad
    return best_dy, best_dx, best_sad


def compensate(cur: np.ndarray, prev: np.ndarray):
    """Build a prediction frame and motion vectors from the previous frame.

    Returns (pred, vecs, sad0, sad1).  Each vector is (bx, by, dx, dy).
    """
    cur = gray(cur)
    prev = gray(prev)
    if cur.shape != prev.shape:
        raise ValueError("current and previous frames must have the same shape")
    h, w = cur.shape
    pred = np.empty_like(cur)
    vecs: list[tuple[int, int, int, int]] = []
    sad0 = int(np.abs(cur.astype(np.int16) - prev.astype(np.int16)).sum())
    sad1 = 0

    for by in range(0, h - BS + 1, BS):
        for bx in range(0, w - BS + 1, BS):
            dy, dx, _ = search_block(prev, cur, bx, by)
            y, x = by + dy, bx + dx
            predicted_block = prev[y:y + BS, x:x + BS]
            current_block = cur[by:by + BS, bx:bx + BS]
            pred[by:by + BS, bx:bx + BS] = predicted_block
            sad1 += int(np.abs(current_block.astype(np.int16) - predicted_block.astype(np.int16)).sum())
            vecs.append((bx, by, dx, dy))

    # The fixed lab dimensions are divisible by BS.  Keep the routine safe for
    # other dimensions by copying any partial border blocks unchanged.
    if h % BS:
        pred[h - h % BS:, :] = cur[h - h % BS:, :]
    if w % BS:
        pred[:, w - w % BS:] = cur[:, w - w % BS:]
    return pred, vecs, sad0, sad1


def entropy(values: np.ndarray) -> float:
    """Shannon entropy in bits per pixel over the observed signed values."""
    data = np.asarray(values).reshape(-1)
    if data.size == 0:
        return 0.0
    _, counts = np.unique(data, return_counts=True)
    p = counts.astype(np.float64) / data.size
    return float(-(p * np.log2(p)).sum())


def _texture(seed: int, width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (height, width), dtype=np.uint8)
    smooth = np.asarray(Image.fromarray(noise).resize((width // 4, height // 4)).resize((width, height)))
    yy, xx = np.mgrid[:height, :width]
    bands = 35 * np.sin(xx / 23) + 28 * np.cos(yy / 19)
    return np.clip(70 + 0.62 * noise.astype(float) + 0.35 * smooth.astype(float) + bands, 0, 255).astype(np.uint8)


def generate_scenes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Create four deterministic pairs that demonstrate the lab's cases."""
    base = _texture(1516, W + 64, H)
    stationary_prev = base[:, :W]
    # A tiny sensor-noise difference keeps the stationary case non-zero.
    rng = np.random.default_rng(1501)
    stationary_cur = np.clip(stationary_prev.astype(np.int16) + rng.integers(-1, 2, (H, W)), 0, 255).astype(np.uint8)

    pan_prev = base[:, :W]
    pan_cur = base[:, 6:W + 6]

    # Handheld wobble combines a few pixels of camera motion with a slight
    # exposure variation; the global motion is still mostly predictable.
    shaky_prev = base[:, 2:W + 2]
    shaky_cur = np.clip(base[:, 8:W + 8].astype(np.int16) + 3, 0, 255).astype(np.uint8)

    # Independent moving rectangles on a low-detail background create blocks
    # whose best local matches disagree and can introduce seams.
    bg = np.zeros((H, W), dtype=np.uint8)
    bg[:] = 76
    bg[:, ::48] = 90
    bg[::48, :] = 90
    a = Image.fromarray(bg)
    b = Image.fromarray(bg.copy())
    da, db = ImageDraw.Draw(a), ImageDraw.Draw(b)
    da.rectangle((80, 80, 215, 185), fill=210)
    da.ellipse((360, 50, 450, 140), fill=35)
    db.rectangle((112, 88, 247, 193), fill=210)
    db.ellipse((324, 68, 414, 158), fill=35)
    movement_prev, movement_cur = np.asarray(a), np.asarray(b)
    return {
        SCENES[0]: (stationary_prev, stationary_cur),
        SCENES[1]: (pan_prev, pan_cur),
        SCENES[2]: (shaky_prev, shaky_cur),
        SCENES[3]: (movement_prev, movement_cur),
    }


def draw_vectors(frame: np.ndarray, vecs: list[tuple[int, int, int, int]], output: Path | str) -> None:
    """Save a vector-field overlay for one frame."""
    image = gray(frame)
    x = np.array([bx + BS / 2 for bx, by, dx, dy in vecs])
    y = np.array([by + BS / 2 for bx, by, dx, dy in vecs])
    dx = np.array([dx for bx, by, dx, dy in vecs])
    dy = np.array([dy for bx, by, dx, dy in vecs])
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.quiver(x, y, dx, dy, color="#ff3b30", angles="xy", scale_units="xy", scale=1, width=0.0025)
    ax.set_title("Вектори руху, блок 16×16")
    ax.axis("off")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _save_pair_plot(prev, cur, pred, stem: str) -> None:
    d0 = cur.astype(np.int16) - prev.astype(np.int16)
    d1 = cur.astype(np.int16) - pred.astype(np.int16)
    vmax = max(1, int(max(np.abs(d0).max(), np.abs(d1).max())))
    for name, data in (("diff_plain.png", d0), ("diff_compensated.png", d1)):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.imshow(data, cmap="gray", vmin=-vmax, vmax=vmax)
        ax.set_title("Різниця без компенсації" if name == "diff_plain.png" else "Залишок після компенсації")
        ax.axis("off")
        fig.savefig(OUT / name, dpi=150)
        plt.close(fig)


def _ffmpeg() -> str:
    binary = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError("ffmpeg is required for video input/output; install it and add it to PATH")
    return binary


def make_demo_video(path: Path) -> None:
    """Create a short deterministic sample clip when no personal clip is given."""
    panorama = Image.new("RGB", (W * 2, H), "#d9e6ef")
    draw = ImageDraw.Draw(panorama)
    colors = [("#476b50", "#c8d28a"), ("#486a8d", "#9ac1d8"), ("#8b573b", "#dfbd89"), ("#3c514d", "#b1c6a0")]
    for panel in range(4):
        x0 = panel * (W // 2)
        sky, ground = colors[panel]
        draw.rectangle((x0, 0, x0 + W // 2, H // 2), fill=sky)
        draw.rectangle((x0, H // 2, x0 + W // 2, H), fill=ground)
        for j in range(12):
            x = x0 + 16 + j * 25
            y = 65 + (j * 37 + panel * 11) % (H - 110)
            r = 8 + (j * 7) % 18
            draw.ellipse((x, y, x + r * 2, y + r * 2), fill=("#f3d27a" if j % 2 else "#314739"))
        draw.rectangle((x0 + 35, 235, x0 + 95, 315), fill="#564435")
        draw.rectangle((x0 + 110, 205, x0 + 156, 315), fill="#725a43")
    frames = np.asarray(panorama)
    raw_cmd = [
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{frames.shape[1]}x{frames.shape[0]}",
        "-framerate", "1", "-i", "-", "-vf",
        "crop=640:352:x='t*72':y=0,scale=640:352,format=yuv420p", "-t", "5",
        "-r", "30", "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", str(path),
    ]
    subprocess.run(raw_cmd, input=frames.tobytes(), check=True)


def video_to_frames(path: Path | str, limit: int = 60) -> list[np.ndarray]:
    """Decode up to limit RGB frames using ffmpeg and return them as arrays."""
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-frames:v", str(limit), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
    frame_bytes = W * H * 3
    return [np.frombuffer(proc.stdout[i:i + frame_bytes], dtype=np.uint8).reshape(H, W, 3).copy()
            for i in range(0, len(proc.stdout) - frame_bytes + 1, frame_bytes)]


def run_tests() -> None:
    """Small correctness checks matching the eight checks in the handout."""
    base = _texture(9, W + 32, H)
    prev = base[:, :W]
    cur = base[:, 4:W + 4]
    checks = []
    # 1. A stationary block is found at zero displacement.
    dy, dx, sad = search_block(prev, prev, 100, 100)
    checks.append((dy == 0 and dx == 0 and sad == 0, "stationary block"))
    # 2. A translated block is found with the inverse offset in prev.
    dy, dx, sad = search_block(prev, cur, 180, 160)
    checks.append((dy == 0 and dx == 4 and sad == 0, "pan block"))
    # 3. Boundary blocks never index beyond the previous frame.
    dy, dx, sad = search_block(prev, cur, W - BS, H - BS)
    checks.append((isinstance(sad, int), "edge bounds"))
    # 4-6. Whole-frame compensation has correct outputs and moving regions.
    pred, vecs, sad0, sad1 = compensate(cur, prev)
    checks.append((pred.shape == cur.shape and pred.dtype == np.uint8, "prediction shape"))
    checks.append((len(vecs) == (W // BS) * (H // BS), "vector count"))
    checks.append((sad1 < sad0, "pan SAD improvement"))
    checks.append((all(len(v) == 4 for v in vecs), "vector tuple shape"))
    # 8. Entropy is zero for a constant residual.
    checks.append((entropy(np.zeros((4, 4), dtype=np.int16)) == 0.0, "entropy"))
    for ok, label in checks:
        print(f"{'OK' if ok else 'НІ'}  {label}")
    if not all(ok for ok, _ in checks):
        raise SystemExit("At least one correctness check failed")


def analyze_scenes() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, (prev, cur)) in enumerate(generate_scenes().items(), 1):
        pred, vecs, sad0, sad1 = compensate(cur, prev)
        e0 = entropy(cur.astype(np.int16) - prev.astype(np.int16))
        e1 = entropy(cur.astype(np.int16) - pred.astype(np.int16))
        row = {"scene": name, "sad_plain": sad0, "sad_compensated": sad1,
               "gain": sad0 / max(1, sad1), "entropy_plain": e0, "entropy_compensated": e1}
        rows.append(row)
        draw_vectors(cur, vecs, OUT / f"vectors_{i}.png")
        print(f"{name:<20} SAD {sad0/1e6:.2f}M → {sad1/1e6:.2f}M  {row['gain']:.1f}×  H {e0:.2f} → {e1:.2f}")
        if i == 2:
            _save_pair_plot(prev, cur, pred, "panorama")
    (OUT / "scene_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def analyze_video(path: Path) -> tuple[dict, list[np.ndarray]]:
    frames = video_to_frames(path, limit=150)
    if len(frames) < 2:
        raise RuntimeError("The supplied video must contain at least two decoded frames")
    pairs = [(29, 30), (74, 75)] if len(frames) > 75 else [(0, 1), (len(frames) - 2, len(frames) - 1)]
    notes = []
    for label, (a_i, b_i) in zip(("плавний рух", "найменший виграш"), pairs):
        a, b = gray(frames[a_i]), gray(frames[b_i])
        pred, vecs, s0, s1 = compensate(b, a)
        e0 = entropy(b.astype(np.int16) - a.astype(np.int16))
        e1 = entropy(b.astype(np.int16) - pred.astype(np.int16))
        notes.append({"where": label, "frames": [a_i, b_i], "gain": s0 / max(1, s1), "entropy_before": e0, "entropy_after": e1})
        draw_vectors(frames[b_i], vecs, OUT / f"video_{label.replace(' ', '_')}.png")
        Image.fromarray(frames[a_i]).save(OUT / f"video_{label.replace(' ', '_')}_before.png")
        Image.fromarray(frames[b_i]).save(OUT / f"video_{label.replace(' ', '_')}_after.png")
    return {"source": str(path.name), "frames": len(frames), "observations": notes}, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Use a personal/source video instead of the included synthetic demo")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_tests()
    rows = analyze_scenes()
    video = args.video or OUT / "demo.mp4"
    if args.video is None:
        make_demo_video(video)
    video_result, _ = analyze_video(video)
    (OUT / "video_results.json").write_text(json.dumps(video_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOutputs saved under {OUT}")


if __name__ == "__main__":
    main()
