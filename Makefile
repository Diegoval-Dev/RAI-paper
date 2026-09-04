.PHONY: fetch shap shap-highres compile loop clean

fetch:
	python code/fetch_images.py

shap:
	python code/run_shap.py --mode full

shap-highres:
	python code/run_shap.py --mode highres-errors

compile:
	latexmk -pdf -cd paper/main.tex

loop: fetch shap shap-highres compile
	@echo "SHAP results in logs/shap_results.json, figures in paper/figures/"
	@echo "Next: insert results into paper/sections/results.tex, then review claims."

clean:
	latexmk -C -cd paper/main.tex
