"""Run SHAP explanations on ConvNeXt-Tiny (facebook/convnext-tiny-224)
image classification over test images.

Outputs SHAP figures to paper/figures/ and a results summary to
logs/shap_results.json for insertion into paper/sections/results.tex.
"""

import json
from pathlib import Path

import numpy as np
import shap
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "data" / "images"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
RESULTS_PATH = REPO_ROOT / "logs" / "shap_results.json"

MODEL_NAME = "facebook/convnext-tiny-224"
TOP_K_LABELS_TO_EXPLAIN = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_test_images() -> tuple[list[str], list[Image.Image]]:
    paths = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No test images found in {IMAGES_DIR}")
    names = [p.stem for p in paths]
    images = [Image.open(p).convert("RGB") for p in paths]
    return names, images


def build_predict_fn(model, processor):
    def predict(images: np.ndarray) -> np.ndarray:
        pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
        inputs = processor(images=pil_images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits.softmax(dim=-1).cpu().numpy()

    return predict


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    id2label = model.config.id2label

    names, images = load_test_images()
    image_arrays = np.stack([np.array(img.resize((224, 224))) for img in images])

    predict_fn = build_predict_fn(model, processor)
    masker = shap.maskers.Image("blur(32,32)", image_arrays[0].shape)
    explainer = shap.Explainer(
        predict_fn, masker, output_names=[id2label[i] for i in range(len(id2label))]
    )

    shap_values = explainer(
        image_arrays,
        max_evals=100,
        batch_size=50,
        outputs=shap.Explanation.argsort.flip[:TOP_K_LABELS_TO_EXPLAIN],
    )

    probs = predict_fn(image_arrays)

    results = []
    import matplotlib.pyplot as plt

    for i, name in enumerate(names):
        top_idx = int(np.argmax(probs[i]))
        results.append(
            {
                "image": name,
                "predicted_label": id2label[top_idx],
                "confidence": float(probs[i][top_idx]),
            }
        )
        shap.image_plot(shap_values[i : i + 1], show=False)
        plt.savefig(FIGURES_DIR / f"shap_{name}.png", bbox_inches="tight")
        plt.close()

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
