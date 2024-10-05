all:
	pdflatex -halt-on-error seismic_model.tex
	bibtex seismic_model
	pdflatex -halt-on-error seismic_model.tex
	pdflatex -halt-on-error seismic_model.tex
