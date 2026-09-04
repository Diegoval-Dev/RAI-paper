# Prompt Log

Record each ChatGPT/loop iteration here: prompt used, what changed, what got reviewed.

## Iteration 1 -- 2026-09-03

- **Step:** structure / .tex gen / results insert / claim review (all sections)
- **What changed:** Wrote abstract, introduction, method, discussion, and
  conclusion (previously stub `% TODO` files). Expanded results.tex to
  include the misclassified-example figures (husky, tabby_cat, jaguar,
  kit_fox) and correctly-classified contrast figures (leopard, malamute)
  with per-image descriptions grounded in the actual SHAP heatmaps.
- **Claims reviewed:** Read every SHAP figure referenced in results.tex
  (husky, tabby_cat, jaguar, kit_fox, leopard, malamute, tiger_cat) directly
  before writing the corresponding prose, to avoid describing attribution
  patterns that aren't actually in the image. Numbers in results.tex (23%
  error rate, 5/22, per-image confidences) cross-checked against
  logs/shap_results.json.
- **Correction made during review:** the kit_fox claim was softened from an
  assertion about the model ignoring ear size to an explicit resolution
  limitation (coarse SHAP superpixel grid can't isolate ears from the rest
  of the head at max_evals=100) -- the figure doesn't support a stronger
  claim.
- **Outcome:** `make compile` succeeds, 9 pages, no undefined references, no
  remaining `% TODO` in paper/sections/. Added a paragraph for the 5th
  error (tiger_cat) after noticing results.tex only covered 4/5 errors in
  detail -- all 5 misclassifications now have a figure + description.

**Stopping criteria check (all met):**
- PDF compiles clean, no undefined refs
- No `% TODO` left in paper/sections/
- All quantitative claims verified against logs/shap_results.json
- 5/5 errors + 2 correct examples described with their SHAP figure
- discussion.tex has an explicit Limitations subsection

## Iteration 2 -- 2026-09-03

- **Step:** SHAP run (highres refinement) / .tex gen / results insert / claim
  review (discussion, method, conclusion)
- **Prompt (summarized):** Refine SHAP explainability by increasing
  computational resolution only for the 5 failure cases identified in
  `logs/shap_results.json` (kit fox, cat, etc.), save new heatmaps to
  `paper/figures/`, and use the visual evidence to close the open question
  in the paper's discussion about whether the model ignores distinctive
  features (kit fox ears, cat facial pattern) or SHAP simply can't resolve
  them at `max_evals=100`.
- **What changed:**
  - `code/run_shap.py`: added `--mode {full,highres-errors}`. `full`
    reproduces the original baseline behavior unchanged (all 22 images,
    `max_evals=100`, writes `shap_<name>.png` +
    `logs/shap_results.json`). `highres-errors` reads the existing
    `logs/shap_results.json`, programmatically identifies misclassified
    images via `is_misclassified()` (true class name not a substring of
    the predicted label -- verified this reproduces exactly the 5 errors
    already called out by hand in `results.tex`: husky, tabby_cat,
    tiger_cat, jaguar, kit_fox), loads only those 5 source images, and
    re-runs the same partition explainer/blur masker at
    `HIGH_RES_MAX_EVALS = 400` (within the requested 300-500 range),
    writing `paper/figures/shap_highres_<name>.png` without touching the
    baseline outputs.
  - `Makefile`: added `shap-highres` target (`python code/run_shap.py
    --mode highres-errors`); `make loop` now runs `fetch shap shap-highres
    compile`.
  - Ran `python code/fetch_images.py` (fresh checkout, `data/images/` is
    gitignored) then `python code/run_shap.py --mode highres-errors`
    (~4 minutes on CPU for the 5 images at max_evals=400). Verified the
    refined predictions/confidences exactly match the baseline
    `logs/shap_results.json` (max_evals only changes the SHAP
    approximation, not the classifier), so the two resolutions' heatmaps
    are directly comparable for the same prediction.
  - `paper/sections/results.tex`: added `\subsection{High-resolution
    refinement...}` with side-by-side baseline/refined figures for all 5
    error cases and per-image analysis.
  - `paper/sections/discussion.tex`: added `\subsection{Closing the
    model-vs-resolution question}` and updated the "Explainer resolution"
    limitation paragraph to report the concrete outcome instead of posing
    it as an open hypothetical.
  - `paper/sections/method.tex`, `paper/sections/conclusion.tex`: added a
    paragraph each documenting the refinement methodology and updating the
    closing claim.
- **Claims reviewed:** Visually inspected all 5 baseline/high-res image
  pairs side by side before writing any claim about them (kit_fox,
  tabby_cat, tiger_cat, jaguar, husky). Only the kit fox pair showed a
  qualitatively different attribution pattern (sharper concentration on
  the ear/head cluster) between the two resolutions; the other four
  reproduced the same pattern at finer granularity. Wrote per-image claims
  to match exactly what was visible, and avoided claiming the refinement
  "proves" ear-attention beyond what a qualitative heatmap comparison
  supports (framed as "elevated importance", not as a quantitative
  feature-importance ranking). Confirmed predicted labels/confidences
  printed by `--mode highres-errors` match `logs/shap_results.json`
  exactly for all 5 images.
- **Outcome:** `make compile` succeeds (see below), no undefined
  references, all 5 new figures render. The open question about kit fox
  ears (and by extension the model-vs-resolution question generally) is
  now answered explicitly: resolution artifact for kit fox (1/5), not for
  the other four errors.

## Format

- **Date:**
- **Step:** (structure / .tex gen / compile fix / SHAP run / results insert / claim review)
- **Prompt:**
- **Outcome:**
- **Claims reviewed:** (which statements in the paper were checked against SHAP output)
