# Target PDF file
TARGET = main.pdf

# Source .tex file
SOURCE = main.tex

# Rule to compile the .tex file to PDF
$(TARGET): $(SOURCE)
	pdflatex $(SOURCE)
	bibtex $(basename $(SOURCE))
	pdflatex $(SOURCE)
	pdflatex $(SOURCE)

# Clean rule to remove auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.bcf *.blg *.run.xml $(TARGET)