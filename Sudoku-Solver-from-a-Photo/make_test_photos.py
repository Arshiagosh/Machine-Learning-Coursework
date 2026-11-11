"""Make synthetic "photos" of printed Sudoku puzzles to test the pipeline.

The original test photos came with the course and are not included here, so this
script renders a puzzle on a sheet of paper, puts it on a background, and applies a
random perspective, rotation, shading, blur and noise, so that it looks roughly like
a phone photo. Only NumPy and Pillow are needed.

    python make_test_photos.py            # writes Test/synthetic_1.jpg ... synthetic_3.jpg
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = "Test"


def make_solution(rng):
    """A random valid 9x9 grid: base pattern, then shuffle bands, rows, columns, digits."""
    base = lambda r, c: (3 * (r % 3) + r // 3 + c) % 9
    rows = [g * 3 + r for g in rng.permutation(3) for r in rng.permutation(3)]
    cols = [g * 3 + c for g in rng.permutation(3) for c in rng.permutation(3)]
    digits = rng.permutation(9) + 1
    return np.array([[digits[base(r, c)] for c in cols] for r in rows])


def make_puzzle(rng, n_clues=30):
    sol = make_solution(rng)
    mask = np.zeros(81, bool)
    mask[rng.choice(81, n_clues, replace=False)] = True
    return np.where(mask.reshape(9, 9), sol, 0), sol


def find_font(size):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_sheet(puzzle, cell=70, margin=110):
    """White sheet with the printed grid: thick lines every 3 cells."""
    size = 9 * cell
    w, h = size + 2 * margin, size + 2 * margin + 160
    sheet = Image.new("L", (w, h), 225)
    d = ImageDraw.Draw(sheet)
    font = find_font(int(cell * 0.62))
    title = find_font(40)
    d.text((margin, 40), "SUDOKU", fill=30, font=title)
    x0, y0 = margin, margin + 60
    for i in range(10):
        lw = 6 if i % 3 == 0 else 2
        d.line([(x0 + i * cell, y0), (x0 + i * cell, y0 + size)], fill=20, width=lw)
        d.line([(x0, y0 + i * cell), (x0 + size, y0 + i * cell)], fill=20, width=lw)
    for r in range(9):
        for c in range(9):
            if puzzle[r, c]:
                cx, cy = x0 + c * cell + cell / 2, y0 + r * cell + cell / 2
                d.text((cx, cy), str(puzzle[r, c]), fill=15, font=font, anchor="mm")
    return sheet


def perspective_coeffs(src, dst):
    """Coefficients for PIL's PERSPECTIVE transform mapping dst points back to src."""
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return np.linalg.solve(np.array(A, float), np.array(b, float))


def photograph(sheet, rng, out=(960, 1280)):
    W, H = out
    # background: smooth wood-ish gradient plus noise
    yy, xx = np.mgrid[0:H, 0:W]
    bg = 90 + 40 * np.sin(xx / 57.0 + rng.uniform(0, 6)) * np.cos(yy / 230.0) + rng.normal(0, 6, (H, W))
    photo = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    # where the sheet corners land: a random convex quadrilateral
    sw, sh = sheet.size
    cx, cy = W / 2 + rng.uniform(-40, 40), H / 2 + rng.uniform(-40, 40)
    scale = rng.uniform(0.70, 0.82) * min(W / sw, H / sh)
    ang = np.deg2rad(rng.uniform(-12, 12))
    corners = np.array([[-sw / 2, -sh / 2], [sw / 2, -sh / 2], [sw / 2, sh / 2], [-sw / 2, sh / 2]]) * scale
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = corners @ R.T + [cx, cy] + rng.normal(0, 18, (4, 2))
    src = [(0, 0), (sw, 0), (sw, sh), (0, sh)]
    coeffs = perspective_coeffs(src, dst.tolist())
    warped = sheet.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=0)
    mask = Image.new("L", sheet.size, 255).transform((W, H), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=0)
    photo.paste(warped, (0, 0), mask)
    # uneven lighting, blur and sensor noise
    a = np.asarray(photo, float)
    light = 0.80 + 0.25 * (xx / W) * rng.uniform(0.5, 1) + 0.10 * (yy / H)
    a = a * light + rng.normal(0, 4, a.shape)
    photo = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.2)))
    return photo.convert("RGB")


def main(n=3, seed=7):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    for k in range(1, n + 1):
        puzzle, sol = make_puzzle(rng, n_clues=int(rng.integers(28, 36)))
        photo = photograph(render_sheet(puzzle), rng)
        path = os.path.join(OUT_DIR, f"synthetic_{k}.jpg")
        photo.save(path, quality=90)
        np.savetxt(os.path.join(OUT_DIR, f"synthetic_{k}_puzzle.txt"), puzzle, fmt="%d")
        np.savetxt(os.path.join(OUT_DIR, f"synthetic_{k}_solution.txt"), sol, fmt="%d")
        print("wrote", path)


if __name__ == "__main__":
    main()
