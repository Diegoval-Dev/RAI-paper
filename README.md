# SHAP Confusable-Class Paper

A short paper using SHAP to explain a ConvNeXt-Tiny image classifier's
predictions on 10 groups of visually similar ImageNet classes (dog breeds,
cat variants, birds, sharks, big cats, foxes, elephants, bears, terriers,
retrievers). Question: when the model confuses two similar classes, does it
attend to the feature a human would use to tell them apart, or does it
attend elsewhere?

## Structure

```
paper/
  main.tex             entry point, includes every section
  sections/            abstract, introduction, model_case, method,
                        results, discussion, conclusion
  figures/             SHAP heatmaps (baseline shap_<name>.png and
                        high-res refinement shap_highres_<name>.png)
  refs/references.bib
code/
  fetch_images.py      downloads one sample image per class for every
                        confusable pair/trio (see CLASS_PAIRS)
  run_shap.py          runs the classifier + SHAP, writes figures and
                        logs/shap_results.json (see modes below)
data/images/           downloaded test images (gitignored)
logs/
  shap_results.json    predicted label, confidence, and group per image
  prompt_log.md        iteration-by-iteration record: what changed, what
                        claims were checked against the data, why
```

## Model and method

- Model: `facebook/convnext-tiny-224` (ConvNeXt-Tiny, ImageNet-1k),
  via Hugging Face `transformers`.
- Explainer: `shap.Explainer` with a model-agnostic partition explainer
  over an `Image` masker (`blur(32,32)`), since ConvNeXt is not a
  gradient-explainer-friendly network in SHAP's sense.
- Two resolutions: a baseline pass at `max_evals=100` over all 22 images,
  and a high-resolution refinement pass at `max_evals=400` restricted to
  just the misclassified images, used to check whether the baseline's
  coarse superpixel grid was hiding fine-grained attention (e.g. ear
  shape) rather than the model genuinely ignoring it. See
  `paper/sections/discussion.tex` (`Closing the model-vs-resolution
  question`) for the per-case outcome.

## Running the pipeline

```
make fetch          # download the 22 test images (skips existing files)
make shap            # baseline SHAP pass, all images, max_evals=100
make shap-highres     # refinement pass, misclassified images only, max_evals=400
make compile          # build paper/main.pdf
make loop             # fetch + shap + shap-highres + compile
```

`fetch_images.py` and `run_shap.py` are both idempotent: re-running skips
already-downloaded images and always writes fresh figures/results from
whatever is in `data/images/`.

## Workflow for extending the paper

1. Add a new confusable-class group to `CLASS_PAIRS` in
   `code/fetch_images.py` (or add images/rewrite a section directly).
2. `make loop` to regenerate figures, results, and the compiled PDF.
3. Update the relevant `paper/sections/*.tex` — ground every quantitative
   claim in `logs/shap_results.json` and every visual claim in the actual
   figure before writing it.
4. Record the iteration in `logs/prompt_log.md`: what changed, and which
   claims were checked against which figure/number.

This is the loop this repo was built around: define structure -> generate
.tex -> compile -> run SHAP -> insert results -> review claims -> repeat.

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# latexmk + a TeX distribution (e.g. MacTeX/TinyTeX) required for compile
```
