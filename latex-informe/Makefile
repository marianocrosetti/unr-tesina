# Makefile for LaTeX document compilation
# Usage: make compile, make clean, make view, etc.

# Configuration
LATEX = pdflatex
LATEXMK = latexmk
MAIN = informe
TEXFILE = $(MAIN).tex
OUTDIR = out
PDFFILE = $(OUTDIR)/$(MAIN).pdf

# Partial compilation sources
CHAPTERS = introduccion marco_teorico formulacion_del_problema \
           trabajos_relacionados_y_estado_del_arte experimentos \
           construccion_del_conjunto_de_datos trabajo_futuro

# Set PATH to include TeX Live binaries
export PATH := /usr/local/texlive/2025/bin/universal-darwin:$(PATH)

# Default target
.PHONY: all
all: compile

# Compile the document (run pdflatex twice for cross-references)
.PHONY: compile
compile:
	@mkdir -p $(OUTDIR)
	@echo "Compiling $(TEXFILE)..."
	$(LATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(TEXFILE)
	$(LATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(TEXFILE)
	@echo "✓ Compilation complete: $(PDFFILE)"

# Compile using latexmk (automatically handles multiple passes)
.PHONY: latexmk
latexmk:
	@mkdir -p $(OUTDIR)
	@echo "Compiling with latexmk..."
	$(LATEXMK) -pdf -interaction=nonstopmode -output-directory=$(OUTDIR) $(TEXFILE)
	@echo "✓ Compilation complete: $(PDFFILE)"

# Partial compilation - compile only specific chapters
# Usage: make partial CHAPTERS="introduccion marco_teorico"
.PHONY: partial
partial:
	@echo "Partial compilation mode..."
	@if [ -z "$(ONLY)" ]; then \
		echo "✗ Error: Please specify chapters with ONLY=chapter1,chapter2"; \
		echo "  Available chapters: $(CHAPTERS)"; \
		exit 1; \
	fi
	@echo "Creating temporary main file with only: $(ONLY)"
	@sed 's/\\include{\(.*\)}/\\includeonly{$(subst $(space),$(comma),$(ONLY))}\n\\include{\1}/' $(TEXFILE) > $(MAIN)_partial.tex || true
	@echo "\\includeonly{$(subst $(,),$(comma),$(subst $(space),$(comma),$(ONLY)))}" > $(MAIN)_partial.tex
	@cat $(TEXFILE) >> $(MAIN)_partial.tex
	$(LATEX) -interaction=nonstopmode -jobname=$(MAIN) $(MAIN)_partial.tex
	$(LATEX) -interaction=nonstopmode -jobname=$(MAIN) $(MAIN)_partial.tex
	@rm -f $(MAIN)_partial.tex
	@echo "✓ Partial compilation complete: $(PDFFILE)"

# Fast partial compilation for specific chapters
.PHONY: fast
fast:
	@if [ -z "$(ONLY)" ]; then \
		echo "✗ Error: Specify chapters: make fast ONLY='introduccion marco_teorico'"; \
		echo "  Available: $(CHAPTERS)"; \
		exit 1; \
	fi
	@mkdir -p $(OUTDIR)
	@echo "Fast compiling: $(ONLY)"
	@echo "\\includeonly{$(subst $(space),$(comma),$(ONLY))}" | cat - $(TEXFILE) > $(MAIN)_tmp.tex
	$(LATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) -jobname=$(MAIN) $(MAIN)_tmp.tex
	@rm -f $(MAIN)_tmp.tex
	@echo "✓ Fast compilation done (run twice if references broken)"

# Compile bibliography standalone
.PHONY: bib
bib:
	@mkdir -p $(OUTDIR)
	@echo "Compiling bibliography standalone..."
	$(LATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) bibliografia.tex
	@echo "✓ Bibliography compilation complete: $(OUTDIR)/bibliografia.pdf"

# Compile and view bibliography
.PHONY: showbib
showbib: bib
	@if [ -f $(OUTDIR)/bibliografia.pdf ]; then \
		open $(OUTDIR)/bibliografia.pdf; \
		echo "✓ Opening $(OUTDIR)/bibliografia.pdf"; \
	else \
		echo "✗ $(OUTDIR)/bibliografia.pdf not found."; \
	fi

# Individual chapter targets for convenience
.PHONY: intro
intro:
	@$(MAKE) fast ONLY=introduccion

.PHONY: teoria
teoria:
	@$(MAKE) fast ONLY=marco_teorico

.PHONY: formulacion
formulacion:
	@$(MAKE) fast ONLY=formulacion_del_problema

.PHONY: trabajos
trabajos:
	@$(MAKE) fast ONLY=trabajos_relacionados_y_estado_del_arte

.PHONY: exp
exp:
	@$(MAKE) fast ONLY=experimentos

.PHONY: construccion
construccion:
	@$(MAKE) fast ONLY=construccion_del_conjunto_de_datos

.PHONY: futuro
futuro:
	@$(MAKE) fast ONLY=trabajo_futuro

# Clean auxiliary files
.PHONY: clean
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(OUTDIR)/$(MAIN).aux $(OUTDIR)/$(MAIN).log $(OUTDIR)/$(MAIN).out $(OUTDIR)/$(MAIN).toc $(OUTDIR)/$(MAIN).lof $(OUTDIR)/$(MAIN).lot
	rm -f $(OUTDIR)/$(MAIN).bbl $(OUTDIR)/$(MAIN).blg $(OUTDIR)/$(MAIN).synctex.gz $(OUTDIR)/$(MAIN).fdb_latexmk $(OUTDIR)/$(MAIN).fls
	rm -f $(OUTDIR)/$(MAIN).nav $(OUTDIR)/$(MAIN).snm $(OUTDIR)/$(MAIN).vrb
	rm -f *_partial.tex *_tmp.tex
	rm -f $(OUTDIR)/bibliografia.aux $(OUTDIR)/bibliografia.log $(OUTDIR)/bibliografia.out $(OUTDIR)/bibliografia.synctex.gz
	rm -f $(OUTDIR)/*.aux
	@echo "✓ Clean complete"

# Deep clean (including PDF)
.PHONY: cleanall
cleanall: clean
	@echo "Removing PDF..."
	rm -f $(PDFFILE) $(OUTDIR)/bibliografia.pdf
	@echo "✓ All files removed"

# Open the PDF
.PHONY: view
view:
	@if [ -f $(PDFFILE) ]; then \
		open $(PDFFILE); \
		echo "✓ Opening $(PDFFILE)"; \
	else \
		echo "✗ $(PDFFILE) not found. Run 'make compile' first."; \
	fi

# Compile and view
.PHONY: show
show: compile view

# Help message
.PHONY: help
help:
	@echo "LaTeX Makefile Commands:"
	@echo ""
	@echo "Full compilation:"
	@echo "  make compile   - Compile the full document (runs pdflatex twice)"
	@echo "  make latexmk   - Compile using latexmk (automatic passes)"
	@echo ""
	@echo "Partial/Fast compilation:"
	@echo "  make fast ONLY='chapter1 chapter2' - Fast compile specific chapters"
	@echo "  make intro     - Compile only introduccion"
	@echo "  make teoria    - Compile only marco_teorico"
	@echo "  make formulacion - Compile only formulacion_del_problema"
	@echo "  make trabajos  - Compile only trabajos_relacionados"
	@echo "  make exp       - Compile only experimentos"
	@echo "  make construccion - Compile only construccion_del_conjunto_de_datos"
	@echo "  make futuro    - Compile only trabajo_futuro"
	@echo "  make bib       - Compile only bibliography standalone"
	@echo "  make showbib   - Compile and open bibliography"
	@echo ""
	@echo "Available chapters: $(CHAPTERS)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean     - Remove auxiliary files"
	@echo "  make cleanall  - Remove all generated files including PDF"
	@echo "  make view      - Open the PDF"
	@echo "  make show      - Compile and open the PDF"
	@echo "  make help      - Show this help message"

# Define comma and space for substitution
comma := ,
space := $(empty) $(empty)

