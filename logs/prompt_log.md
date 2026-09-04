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

## Format

- **Date:**
- **Step:** (structure / .tex gen / compile fix / SHAP run / results insert / claim review)
- **Prompt:**
- **Outcome:**
- **Claims reviewed:** (which statements in the paper were checked against SHAP output)
