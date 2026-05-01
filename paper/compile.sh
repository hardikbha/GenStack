#!/bin/bash
# Compile IJCB 2026 paper
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo ""
echo "=== Compilation complete ==="
echo "Output: $(ls -lh main.pdf | awk '{print $5, $NF}')"
echo "Pages: $(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}')"
grep -c "Overfull" main.log && echo " overfull hbox warnings" || echo "No overfull warnings"
grep "undefined" main.log | grep -v "Font shape" | head -5 || echo "No undefined references"
