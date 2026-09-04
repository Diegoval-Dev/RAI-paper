"""Run SHAP explanations on ConvNeXt-Tiny (facebook/convnext-tiny-224)
image classification over test images.

Two modes are supported (see --mode):

- "full" (default): the original baseline pass over all 22 test images at
  max_evals=100. Outputs SHAP figures to paper/figures/shap_<name>.png and a
  results summary to logs/shap_results.json for insertion into
  paper/sections/results.tex.
- "highres-errors": isolates only the images the model misclassified
  (identified from the "full" pass's logs/shap_results.json) and re-explains
  them with a much finer max_evals budget (HIGH_RES_MAX_EVALS), to test
  whether the coarse superpixel grid of the baseline pass -- rather than the
  model itself -- was responsible for attribution not isolating
  fine-grained discriminative features (e.g. kit fox ears, cat facial
  pattern). Outputs to paper/figures/shap_highres_<name>.png; does not
  touch the baseline figures or logs/shap_results.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import shap
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from fetch_images import CLASS_PAIRS

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "data" / "images"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
RESULTS_PATH = REPO_ROOT / "logs" / "shap_results.json"

MODEL_NAME = "facebook/convnext-tiny-224"
TOP_K_LABELS_TO_EXPLAIN = 1

# max_evals used for the original, coarse baseline pass over all 22 images.
BASELINE_MAX_EVALS = 100
# max_evals used for the fine-grained refinement pass, restricted to the
# images the model got wrong at BASELINE_MAX_EVALS. Kept within [300, 500]
# per the finer-resolution requirement while remaining tractable on CPU.
HIGH_RES_MAX_EVALS = 400
HIGH_RES_FIGURE_PREFIX = "shap_highres_"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_group_lookup() -> dict[str, str]:
    return {name: group for group, items in CLASS_PAIRS.items() for name, _ in items}


def load_test_images(names: list[str] | None = None) -> tuple[list[str], list[Image.Image]]:
    """Load test images from IMAGES_DIR. If `names` is given, restrict to
    those image stems (in that order) instead of loading every image."""
    if names is None:
        paths = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No test images found in {IMAGES_DIR}")
        loaded_names = [p.stem for p in paths]
        images = [Image.open(p).convert("RGB") for p in paths]
        return loaded_names, images

    images = []
    for name in names:
        candidates = list(IMAGES_DIR.glob(f"{name}.*"))
        if not candidates:
            raise FileNotFoundError(f"No image file found for '{name}' in {IMAGES_DIR}")
        images.append(Image.open(candidates[0]).convert("RGB"))
    return list(names), images


def is_misclassified(name: str, predicted_label: str) -> bool:
    """Heuristic used to flag an error case: the image's own name (its true
    class, e.g. "kit_fox") does not appear as a substring of the model's
    predicted label (e.g. "grey fox, gray fox, Urocyon cinereoargenteus").
    This matches every error already called out by hand in
    paper/sections/results.tex (husky, tabby_cat, tiger_cat, jaguar,
    kit_fox) and every correct prediction in logs/shap_results.json."""
    true_class = name.replace("_", " ").lower()
    return true_class not in predicted_label.lower()


def find_error_image_names() -> list[str]:
    """Read the baseline logs/shap_results.json (produced by `--mode full`)
    and return the image names the model misclassified."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run `python code/run_shap.py --mode full` "
            "(or `make shap`) first to produce the baseline results that "
            "`--mode highres-errors` isolates errors from."
        )
    results = json.loads(RESULTS_PATH.read_text())
    error_names = [
        r["image"] for r in results if is_misclassified(r["image"], r["predicted_label"])
    ]
    return error_names


def build_predict_fn(model, processor):
    def predict(images: np.ndarray) -> np.ndarray:
        pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
        inputs = processor(images=pil_images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits.softmax(dim=-1).cpu().numpy()

    return predict


def run_explainer(names, images, max_evals: int, predict_fn, id2label):
    """Build a fresh Image-masker partition explainer and run it over
    `images` at the given `max_evals` budget. Returns (shap_values, probs)."""
    image_arrays = np.stack([np.array(img.resize((224, 224))) for img in images])
    masker = shap.maskers.Image("blur(32,32)", image_arrays[0].shape)
    explainer = shap.Explainer(
        predict_fn, masker, output_names=[id2label[i] for i in range(len(id2label))]
    )
    shap_values = explainer(
        image_arrays,
        max_evals=max_evals,
        batch_size=50,
        outputs=shap.Explanation.argsort.flip[:TOP_K_LABELS_TO_EXPLAIN],
    )
    probs = predict_fn(image_arrays)
    return shap_values, probs


def run_full_pass(model, processor, id2label) -> None:
    """Baseline pass: all 22 test images at BASELINE_MAX_EVALS, writing
    paper/figures/shap_<name>.png and logs/shap_results.json."""
    import matplotlib.pyplot as plt

    group_lookup = build_group_lookup()
    names, images = load_test_images()
    predict_fn = build_predict_fn(model, processor)

    shap_values, probs = run_explainer(names, images, BASELINE_MAX_EVALS, predict_fn, id2label)

    results = []
    for i, name in enumerate(names):
        top_idx = int(np.argmax(probs[i]))
        results.append(
            {
                "image": name,
                "group": group_lookup.get(name, "unknown"),
                "predicted_label": id2label[top_idx],
                "confidence": float(probs[i][top_idx]),
            }
        )
        shap.image_plot(shap_values[i : i + 1], show=False)
        plt.savefig(FIGURES_DIR / f"shap_{name}.png", bbox_inches="tight")
        plt.close()

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to {RESULTS_PATH}")


def run_highres_errors_pass(model, processor, id2label) -> None:
    """Refinement pass: only the images misclassified in the baseline run
    (per logs/shap_results.json), re-explained at HIGH_RES_MAX_EVALS.
    Writes paper/figures/shap_highres_<name>.png and does not touch the
    baseline figures or logs/shap_results.json."""
    import matplotlib.pyplot as plt

    error_names = find_error_image_names()
    if not error_names:
        print("No misclassified images found in logs/shap_results.json; nothing to refine.")
        return
    print(f"Refining {len(error_names)} misclassified image(s) at max_evals={HIGH_RES_MAX_EVALS}: "
          f"{', '.join(error_names)}")

    names, images = load_test_images(error_names)
    predict_fn = build_predict_fn(model, processor)

    shap_values, probs = run_explainer(names, images, HIGH_RES_MAX_EVALS, predict_fn, id2label)

    for i, name in enumerate(names):
        top_idx = int(np.argmax(probs[i]))
        print(
            f"  {name}: predicted '{id2label[top_idx]}' "
            f"(conf. {float(probs[i][top_idx]):.3f}) at max_evals={HIGH_RES_MAX_EVALS}"
        )
        shap.image_plot(shap_values[i : i + 1], show=False)
        out_path = FIGURES_DIR / f"{HIGH_RES_FIGURE_PREFIX}{name}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["full", "highres-errors"],
        default="full",
        help=(
            "'full': baseline SHAP pass over all 22 images at "
            f"max_evals={BASELINE_MAX_EVALS} (default). "
            "'highres-errors': re-run SHAP only on the images misclassified "
            f"in the baseline pass, at max_evals={HIGH_RES_MAX_EVALS}."
        ),
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    id2label = model.config.id2label

    if args.mode == "full":
        run_full_pass(model, processor, id2label)
    else:
        run_highres_errors_pass(model, processor, id2label)


if __name__ == "__main__":
    main()
