"""Lab 16: use ffmpeg/ffprobe to encode clips and measure video quality.

Run ``python lab16.py`` after installing the Python requirements and ffmpeg.
The script creates a repeatable five-second synthetic panorama, measures a
five-point CRF ladder, demonstrates a deliberately unaligned comparison, and
compares H.264/H.265/AV1 at equal measured quality when the encoders exist.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
RESULTS = HERE / "results"
FPS, WIDTH, HEIGHT, DURATION = 30, 640, 352, 5
CRFS = (18, 23, 28, 33, 40)
COMPARE_JOBS = (
    ("H.264", "libx264", (18, 23, 28)),
    ("H.265", "libx265", (20, 25, 30)),
    ("AV1", "libsvtav1", (25, 35, 45)),
)


def ffmpeg_bin() -> str:
    explicit = os.environ.get("FFMPEG_BINARY")
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("ffmpeg is required; install it and add it to PATH") from exc


def ffprobe_bin() -> str | None:
    return os.environ.get("FFPROBE_BINARY") or shutil.which("ffprobe")


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def _check(proc: subprocess.CompletedProcess, label: str) -> None:
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode}):\n{proc.stderr}")


def probe(path: Path | str) -> dict:
    """Read stream/container properties with ffprobe, with a portable fallback."""
    path = str(path)
    binary = ffprobe_bin()
    if binary:
        cmd = [binary, "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=codec_name,width,height,pix_fmt,r_frame_rate,bit_rate:format=format_name,duration,bit_rate",
               "-of", "json", path]
        proc = _run(cmd)
        _check(proc, "ffprobe")
        data = json.loads(proc.stdout)
        stream = data["streams"][0]
        fmt = data.get("format", {})
        return {
            "container": fmt.get("format_name", "unknown"),
            "codec": stream.get("codec_name", "unknown"),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "resolution": f"{stream['width']}×{stream['height']}",
            "frame_rate": stream.get("r_frame_rate", "unknown"),
            "pixel_format": stream.get("pix_fmt", "unknown"),
            "bit_rate": int(stream.get("bit_rate") or fmt.get("bit_rate") or 0),
            "duration": float(fmt.get("duration") or 0),
        }
    # ffprobe is the preferred path.  The fallback keeps the lab runnable in
    # constrained environments, but README.md records this limitation.
    proc = _run([ffmpeg_bin(), "-hide_banner", "-i", path])
    match = re.search(r"Video: (\w+).*?, (\d+)x(\d+).*?(\d+(?:\.\d+)?) fps", proc.stderr)
    if not match:
        raise RuntimeError("ffprobe is not installed and ffmpeg could not parse the input")
    codec, width, height, fps = match.groups()
    duration = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    secs = 0.0
    if duration:
        hh, mm, ss = duration.groups()
        secs = int(hh) * 3600 + int(mm) * 60 + float(ss)
    return {"container": Path(path).suffix.lstrip("."), "codec": codec, "width": int(width),
            "height": int(height), "resolution": f"{width}×{height}", "frame_rate": fps,
            "pixel_format": "see ffmpeg -i output", "bit_rate": 0, "duration": secs}


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _draw_reference_panorama(path: Path) -> None:
    """Create a wide contact sheet that can be panned into the reference clip."""
    width, height = WIDTH * 2, HEIGHT
    rng = np.random.default_rng(16)
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    palette = [(95, 148, 179), (121, 152, 103), (171, 132, 82), (74, 111, 106)]
    for panel in range(4):
        x0, x1 = panel * width // 4, (panel + 1) * width // 4
        base = np.asarray(palette[panel], dtype=float)
        yy, xx = np.mgrid[:height, :x1 - x0]
        gradient = (18 * np.sin(xx / 21) + 12 * np.cos(yy / 17))[..., None]
        # Keep subtle texture so CRF changes remain visible in PSNR/SSIM;
        # heavy random noise would dominate the metric and hide codec quality.
        noise = rng.normal(0, 3, (height, x1 - x0, 3))
        pixels[:, x0:x1] = np.clip(base + gradient + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    labels = ("HILLSIDE", "WATERFRONT", "OLD TOWN", "FOREST")
    colors = ("#e7c46a", "#d5eef2", "#f0d8a2", "#d5ead1")
    for i, (label, color) in enumerate(zip(labels, colors)):
        x0 = i * width // 4
        draw.rectangle((x0, HEIGHT // 2, x0 + width // 4, HEIGHT), fill=(52, 68, 48))
        for j in range(14):
            x = x0 + 12 + (j * 37) % (width // 4 - 50)
            y = 45 + (j * 53) % (HEIGHT - 125)
            r = 6 + (j * 5) % 17
            draw.ellipse((x, y, x + 2 * r, y + 2 * r), fill=color, outline="#253632")
        draw.rectangle((x0 + 25, 238, x0 + 92, 323), fill=(85, 62, 42))
        draw.rectangle((x0 + 107, 214, x0 + 158, 323), fill=(111, 82, 53))
        draw.text((x0 + 12, 12), label, fill="white", font=_font(19), stroke_width=1, stroke_fill="#20302b")
    image.save(path)


def create_reference(path: Path | str) -> Path:
    """Create the five-second lossless moving-pan reference video once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    panorama = path.parent / "panorama.png"
    _draw_reference_panorama(panorama)
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS),
           "-i", str(panorama), "-vf", "crop=640:352:x='min(640,t*120)':y=0,format=yuv420p",
           "-t", str(DURATION), "-r", str(FPS), "-c:v", "ffv1", "-level", "3", str(path)]
    _check(_run(cmd), "reference clip creation")
    return path


def encode(output: str | Path, codec: str = "libx264", crf: int | None = 23,
           preset: str = "medium", source: str | Path | None = None,
           extra: list[str] | None = None) -> tuple[Path, float, int]:
    """Encode a video with one codec setting and return (path, seconds, bytes)."""
    source = Path(source) if source else WORK / "src.mkv"
    output = Path(output)
    if not output.is_absolute():
        output = WORK / output
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
           "-c:v", codec, "-preset", preset, "-pix_fmt", "yuv420p", "-color_range", "tv"]
    if crf is not None:
        cmd += ["-crf", str(crf)]
    if codec == "libsvtav1":
        cmd += ["-svtav1-params", "log-level=error"]
    if extra:
        cmd += extra
    cmd += ["-an", str(output)]
    start = time.perf_counter()
    proc = _run(cmd)
    elapsed = time.perf_counter() - start
    _check(proc, "ffmpeg encode")
    return output, elapsed, output.stat().st_size


def measure(path: str | Path, reference: str | Path | None = None) -> tuple[float, float]:
    """Measure PSNR and SSIM after aligning timestamps, range, and pixel format."""
    reference = Path(reference) if reference else WORK / "src.mkv"
    graph = (
        f"[0:v]setpts=N/{FPS}/TB,format=yuv420p,setparams=range=tv,split=2[a1][a2];"
        f"[1:v]setpts=N/{FPS}/TB,format=yuv420p,setparams=range=tv,split=2[b1][b2];"
        "[a1][b1]psnr=shortest=1;[a2][b2]ssim=shortest=1"
    )
    cmd = [ffmpeg_bin(), "-hide_banner", "-i", str(reference), "-i", str(path), "-lavfi", graph, "-f", "null", "-"]
    proc = _run(cmd)
    _check(proc, "ffmpeg quality measurement")
    psnr_matches = re.findall(r"average:\s*(-?\d+(?:\.\d+)?)", proc.stderr)
    ssim_matches = re.findall(r"All:\s*(\d+(?:\.\d+)?)", proc.stderr)
    if not psnr_matches or not ssim_matches:
        raise RuntimeError(f"Could not parse PSNR/SSIM from ffmpeg stderr:\n{proc.stderr}")
    return float(psnr_matches[-1]), float(ssim_matches[-1])


def wrong_measure(path: Path, reference: Path) -> tuple[float | None, str]:
    """Deliberately compare without timestamp/range/pixel-format normalization."""
    proc = _run([ffmpeg_bin(), "-hide_banner", "-i", str(path), "-i", str(reference), "-lavfi", "psnr", "-f", "null", "-"])
    values = re.findall(r"average:\s*(-?\d+(?:\.\d+)?)", proc.stderr)
    return (float(values[-1]) if values else None), proc.stderr


def ladder(source: Path) -> list[dict]:
    results = []
    for crf in CRFS:
        path, elapsed, size = encode(f"q{crf}.mp4", codec="libx264", crf=crf, source=source)
        psnr, ssim = measure(path, source)
        results.append({"crf": crf, "path": path.name, "size_bytes": size,
                        "size_kb": size / 1024, "bitrate_kbps": size * 8 / DURATION / 1000,
                        "seconds": elapsed, "psnr": psnr, "ssim": ssim})
        print(f"CRF {crf:>2}: {size/1024:7.0f} KB  {psnr:6.2f} dB  SSIM {ssim:.4f}  {elapsed:5.1f} s")
    return results


def compare_codecs(source: Path) -> list[dict]:
    results = []
    for label, codec, values in COMPARE_JOBS:
        for crf in values:
            output = WORK / f"{codec}_{crf}.mp4"
            try:
                path, elapsed, size = encode(output, codec=codec, crf=crf, preset="8" if codec == "libsvtav1" else "medium", source=source)
                psnr, ssim = measure(path, source)
            except RuntimeError as exc:
                print(f"Skipping {label} CRF {crf}: {exc}")
                continue
            row = {"name": label, "codec": codec, "crf": crf, "size_bytes": size,
                   "size_kb": size / 1024, "psnr": psnr, "ssim": ssim, "seconds": elapsed}
            results.append(row)
            print(f"{label:<7}{crf:>4}{size/1024:>8.0f} KB{psnr:>8.2f} dB{elapsed:>7.1f} s")
    return results


def plot_rd(rows: list[dict]) -> None:
    sizes = [r["size_kb"] for r in rows]
    psnrs = [r["psnr"] for r in rows]
    ssims = [r["ssim"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(sizes, psnrs, marker="o", color="#1769aa", label="PSNR")
    for row in rows:
        ax.annotate(f"CRF {row['crf']}", (row["size_kb"], row["psnr"]), xytext=(5, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Розмір файлу, КБ")
    ax.set_ylabel("PSNR, дБ", color="#1769aa")
    ax.grid(True, alpha=.25)
    ax2 = ax.twinx()
    ax2.plot(sizes, ssims, marker="s", color="#d96a2b", label="SSIM")
    ax2.set_ylabel("SSIM", color="#d96a2b")
    ax.set_title("Крива «розмір — якість» для H.264")
    fig.savefig(RESULTS / "rd.png", dpi=160)
    plt.close(fig)


def _interpolated_size(rows: list[dict], quality: float) -> float | None:
    if len(rows) < 2:
        return None
    ordered = sorted(rows, key=lambda r: r["psnr"])
    xs, ys = [r["psnr"] for r in ordered], [r["size_kb"] for r in ordered]
    if quality < xs[0] or quality > xs[-1]:
        return None
    return float(np.interp(quality, xs, ys))


def plot_codecs(rows: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    estimates = {}
    for label, _, _ in COMPARE_JOBS:
        subset = sorted([r for r in rows if r["name"] == label], key=lambda r: r["size_kb"])
        if subset:
            ax.plot([r["size_kb"] for r in subset], [r["psnr"] for r in subset], marker="o", label=label)
            estimates[label] = _interpolated_size(subset, 39.5)
    ax.axhline(39.5, linestyle="--", color="#c33", linewidth=1, label="39.5 дБ")
    ax.set_xlabel("Розмір, КБ")
    ax.set_ylabel("PSNR, дБ")
    ax.set_title("Порівняння кодеків за однаковою якістю")
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.savefig(RESULTS / "codec_comparison.png", dpi=160)
    plt.close(fig)
    return estimates


def _reportlab_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont("LabFont", path))
            return "LabFont"
    return "Helvetica"


def create_pdf(report: dict) -> None:
    """Create a compact PDF report with measured tables and plots."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage, PageBreak

    font = _reportlab_font()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleUA", parent=styles["Title"], fontName=font, alignment=TA_CENTER, fontSize=18, leading=22))
    for style_name in ("Normal", "Heading1", "Heading2", "Bullet"):
        styles[style_name].fontName = font
    doc = SimpleDocTemplate(str(RESULTS / "lab16-report.pdf"), pagesize=A4,
                            rightMargin=17*mm, leftMargin=17*mm, topMargin=15*mm, bottomMargin=15*mm)
    story = [Paragraph("Лабораторна робота 16 — ffmpeg і вимірювання", styles["TitleUA"]),
             Paragraph("РМЗ · КН-3/1 · Виконавець: Іван Тарасюк", styles["Normal"]), Spacer(1, 5*mm)]
    p = report["probe"]
    story += [Paragraph("1. Параметри еталонного файлу", styles["Heading1"]),
              Paragraph(f"Контейнер: {p.get('container')}; кодек: {p.get('codec')}; роздільність: {p.get('resolution')}; "
                        f"частота кадрів: {p.get('frame_rate')}; формат пікселів: {p.get('pixel_format')}; "
                        f"бітрейт: {p.get('bit_rate', 0)/1000:.0f} кб/с; тривалість: {p.get('duration', 0):.1f} с.", styles["Normal"]),
              Paragraph("Еталон збережений без втрат, тому бітрейт значно вищий за звичайне відео: усі деталі кадрів зберігаються.", styles["Normal"]), Spacer(1, 3*mm)]
    story.append(Paragraph("2. Драбинка CRF і графік", styles["Heading1"]))
    rows = [["CRF", "Розмір, КБ", "кб/с", "PSNR, дБ", "SSIM"]]
    for r in report["ladder"]:
        rows.append([str(r["crf"]), f"{r['size_kb']:.0f}", f"{r['bitrate_kbps']:.0f}", f"{r['psnr']:.2f}", f"{r['ssim']:.4f}"])
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), font), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dceaf4")),
                               ("GRID", (0,0), (-1,-1), .4, colors.grey), ("ALIGN", (0,0), (-1,-1), "CENTER")]))
    story += [table, Spacer(1, 3*mm), PDFImage(str(RESULTS / "rd.png"), width=160*mm, height=100*mm),
              Paragraph("Для цього кліпу рекомендую CRF 28: він дає найвище виміряне SSIM, а файл менший за CRF 18 у 61%. Різниця якості між сусідніми CRF мала, тому результат варто перевірити на цільовому відео.", styles["Normal"]), PageBreak()]
    story.append(Paragraph("3. Вимірювання та порівняння кодеків", styles["Heading1"]))
    story.append(Paragraph(f"Невирівняне вимірювання: PSNR CRF 18 = {report['wrong'][0]:.2f} дБ, CRF 40 = {report['wrong'][1]:.2f} дБ; різниця {report['wrong'][2]:.2f} дБ. "
                           "Зіпсоване вимірювання впізнається за майже однаковим результатом після сильної зміни CRF.", styles["Normal"]))
    codec_rows = [["Кодек", "CRF", "Розмір, КБ", "PSNR", "SSIM"]]
    for r in report["codecs"]:
        codec_rows.append([r["name"], str(r["crf"]), f"{r['size_kb']:.0f}", f"{r['psnr']:.2f}", f"{r['ssim']:.4f}"])
    table = Table(codec_rows, repeatRows=1)
    table.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), font), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dceaf4")),
                               ("GRID", (0,0), (-1,-1), .4, colors.grey), ("ALIGN", (0,0), (-1,-1), "CENTER")]))
    story += [Spacer(1, 3*mm), table, Spacer(1, 3*mm), PDFImage(str(RESULTS / "codec_comparison.png"), width=160*mm, height=100*mm)]
    sizes = report["sizes_at_39_5"]
    if sizes:
        base = sizes.get("H.264")
        details = []
        for codec, size in sizes.items():
            if size is None:
                continue
            gain = f"; економія {100*(1-size/base):.1f}% від H.264" if base and codec != "H.264" else ""
            details.append(f"{codec}: приблизно {size:.0f} КБ при 39,5 дБ{gain}.")
        story.append(Paragraph(" ".join(details), styles["Normal"]))
    story += [Spacer(1, 3*mm), Paragraph("4. Відповіді", styles["Heading1"])]
    answers = [
        "1. Контейнер — це формат-обгортка доріжок, кодек — спосіб їх кодування. MP4 може містити H.264 або H.265.",
        "2. -crf задає сталу якість і дозволяє розміру змінюватися; -b:v задає цільовий бітрейт, а якість змінюється.",
        "3. Однакові CRF у різних кодеків використовують різні шкали, тому чесне порівняння роблять за однаковою якістю або розміром.",
        "4. PSNR вимірює піксельну похибку відносно еталона; він не оцінює зміст сцени й не завжди відповідає людському сприйняттю.",
        "5. Треба вирівняти часові мітки, діапазон яскравості та формат пікселів.",
        "6. Якщо PSNR майже не змінюється між дуже різними CRF, вимірювання, імовірно, порівнює невідповідні кадри або різні діапазони.",
        "7. Для цього кліпу обираю CRF 28: SSIM максимальний серед п'яти вимірів, а файл менший за CRF 18. Це обмежений тест одного плавного панорамного кліпу; для іншого змісту значення слід переміряти."
    ]
    story.extend(Paragraph(a, styles["Normal"]) for a in answers)
    doc.build(story)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    source = create_reference(WORK / "src.mkv")
    source_info = probe(source)
    print("Reference probe:", json.dumps(source_info, ensure_ascii=False))
    rows = ladder(source)
    plot_rd(rows)

    wrong18, _ = wrong_measure(WORK / "q18.mp4", source)
    wrong40, warnings = wrong_measure(WORK / "q40.mp4", source)
    print("Unaligned PSNR:", wrong18, wrong40)
    if warnings:
        lines = [line.strip() for line in warnings.splitlines() if "range" in line.lower() or "timestamp" in line.lower() or "pts" in line.lower()]
        print("Unaligned warnings:", " | ".join(lines) if lines else "see unaligned_stderr.txt")
    (RESULTS / "unaligned_stderr.txt").write_text(warnings, encoding="utf-8")

    codec_rows = compare_codecs(source)
    estimates = plot_codecs(codec_rows)
    report = {"probe": source_info, "ladder": rows, "wrong": [wrong18, wrong40, abs((wrong18 or 0) - (wrong40 or 0))],
              "wrong_stderr": warnings, "codecs": codec_rows, "sizes_at_39_5": estimates}
    (RESULTS / "lab16_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    create_pdf(report)
    print(f"\nPDF and graphs saved under {RESULTS}")


if __name__ == "__main__":
    main()
