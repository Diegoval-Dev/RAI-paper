.PHONY: fetch shap compile loop clean

fetch:
	python code/fetch_images.py

shap:
	python code/run_shap.py

compile:
	latexmk -pdf -cd paper/main.tex

loop: fetch shap compile
	@echo "SHAP results in logs/shap_results.json, figures in paper/figures/"
	@echo "Next: insert results into paper/sections/results.tex, then review claims."

clean:
	latexmk -C -cd paper/main.tex
