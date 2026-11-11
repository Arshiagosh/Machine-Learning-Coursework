"""Run the four-phase pipeline from sudoku.py on the photos in Test/ and save figures.

The pipeline functions (rectify, grid detection, cell segmentation, digit extraction,
backtracking solver, drawing) are taken unchanged from sudoku.py. Only the model loader
is swapped for the NumPy reader in keras_numpy.py, so TensorFlow is not needed.

    python make_test_photos.py     # synthetic photos (once)
    python run_demo.py             # writes figures/ and results.json
"""
import ast
import glob
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import keras_numpy

PIPELINE = ["preprocess_image", "find_paper_contour", "order_points", "find_corners",
            "perspective_transform", "phase1_rectify", "find_sudoku_grid", "phase2_draw_grid",
            "detect_cell_coords", "extract_digit", "segment_and_recognize", "solve_sudoku",
            "draw_solution"]


def load_pipeline(path="sudoku.py"):
    """Exec only the pipeline function definitions from sudoku.py."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in PIPELINE]
    missing = set(PIPELINE) - {d.name for d in defs}
    assert not missing, missing
    ns = {"cv2": cv2, "np": np, "plt": plt, "display_image": lambda *a, **k: None}
    exec(compile(ast.Module(body=defs, type_ignores=[]), path, "exec"), ns)
    return ns


def valid_solution(g):
    g = np.asarray(g)
    ok = lambda v: sorted(v.ravel().tolist()) == list(range(1, 10))
    return all(ok(g[i]) for i in range(9)) and all(ok(g[:, j]) for j in range(9)) and \
        all(ok(g[3 * a:3 * a + 3, 3 * b:3 * b + 3]) for a in range(3) for b in range(3))


def main(model_path="models/improved_digit_classifier.h5"):
    P = load_pipeline()
    model = keras_numpy.load_model(model_path)
    os.makedirs("figures", exist_ok=True)
    results = []
    for path in sorted(glob.glob(os.path.join("Test", "*.jpg"))):
        name = os.path.splitext(os.path.basename(path))[0]
        photo = cv2.imread(path)
        rectified = P["phase1_rectify"](path, visualize=False)
        grid_overlay = P["phase2_draw_grid"](rectified, visualize=False)
        cells = P["detect_cell_coords"](rectified)
        recognised = P["segment_and_recognize"](rectified, model, cells, visualize=False)
        solved = P["solve_sudoku"](recognised)
        answer = P["draw_solution"](rectified, cells, recognised, solved) if solved else rectified

        truth_file = os.path.join("Test", f"{name}_puzzle.txt")
        row = {"image": os.path.basename(path), "clues_recognised": int(np.count_nonzero(recognised)),
               "solved": solved is not None, "solution_valid": bool(solved) and valid_solution(solved)}
        if os.path.exists(truth_file):
            truth = np.loadtxt(truth_file, dtype=int)
            rec = np.asarray(recognised)
            row["clues_true"] = int(np.count_nonzero(truth))
            row["cells_correct"] = int((rec == truth).sum())
            row["clues_missed"] = int(((truth > 0) & (rec == 0)).sum())
            row["clues_wrong"] = int(((truth > 0) & (rec > 0) & (rec != truth)).sum())
            row["false_clues"] = int(((truth == 0) & (rec > 0)).sum())
            if solved:
                # with missed clues the solver may pick a different (but valid) completion
                row["solution_agrees_with_all_true_clues"] = bool(((truth == 0) | (np.asarray(solved) == truth)).all())
        results.append(row)
        print(row)

        fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
        for a, img, title in zip(ax, [photo, rectified, grid_overlay, answer],
                                 ["1. Photo", "2. Rectified paper", "3. Grid found", "4. Solved (red)"]):
            a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); a.set_title(title); a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join("figures", f"{name}_pipeline.png"), dpi=110)
        plt.close(fig)

    with open("results.json", "w") as f:
        json.dump({"model": os.path.basename(model_path), "photos": results}, f, indent=2)


if __name__ == "__main__":
    main()
