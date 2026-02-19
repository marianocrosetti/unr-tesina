LATEX = pdflatex
MAIN = presentacion
OUTDIR = out

# Set PATH to include TeX Live binaries
export PATH := /usr/local/texlive/2025/bin/universal-darwin:$(PATH)

.PHONY: compile
compile:
	@mkdir -p $(OUTDIR)
	@echo "Compiling $(MAIN).tex..."
	$(LATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(MAIN).tex
	$(LATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(MAIN).tex
	@echo "✓ Compilation complete: $(OUTDIR)/$(MAIN).pdf"

.PHONY: clean
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(OUTDIR)/$(MAIN).aux $(OUTDIR)/$(MAIN).log $(OUTDIR)/$(MAIN).out $(OUTDIR)/$(MAIN).toc
	rm -f $(OUTDIR)/$(MAIN).nav $(OUTDIR)/$(MAIN).snm $(OUTDIR)/$(MAIN).vrb $(OUTDIR)/$(MAIN).synctex.gz
	@echo "✓ Clean complete"

.PHONY: view
view:
	@if [ -f $(OUTDIR)/$(MAIN).pdf ]; then \
		open $(OUTDIR)/$(MAIN).pdf; \
		echo "✓ Opening $(OUTDIR)/$(MAIN).pdf"; \
	else \
		echo "✗ $(OUTDIR)/$(MAIN).pdf not found. Run 'make compile' first."; \
	fi

.PHONY: show
show: compile view


