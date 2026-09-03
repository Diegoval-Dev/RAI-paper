# SHAP Image Classifier Paper

## Structure

- `paper/` — LaTeX source (`main.tex`, `sections/`, `figures/`, `refs/`)
- `code/run_shap.py` — loads model, runs SHAP on test images, writes figures + `logs/shap_results.json`
- `data/images/` — 1-3 test images (not tracked in git)
- `logs/` — SHAP results, prompt log

## Loop

1. Define structure — edit section stubs in `paper/sections/`
2. Generate .tex — write content
3. Compile — `make compile`
4. Run SHAP — `make shap` (needs images in `data/images/`)
5. Insert results — pull `logs/shap_results.json` + `paper/figures/*.png` into `paper/sections/results.tex`
6. Review claims — check every stated claim against SHAP output, log in `logs/prompt_log.md`

Full loop: `make loop`

## Setup

```
pip install -r requirements.txt
# latexmk + a TeX distribution (e.g. MacTeX/TinyTeX) required for compile
```
