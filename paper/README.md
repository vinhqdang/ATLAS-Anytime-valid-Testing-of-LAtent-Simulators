# ATLAS — manuscript

Review-ready draft of the ATLAS paper.

- `main.tex` — the manuscript (article class, 10 pages).
- `references.bib` — verified bibliography (mirror of `docs/references.bib`).
- `figures/` — figures used by the paper (copies of `results/` and `figures/`).
- `main.pdf` — compiled PDF (committed for convenient review).

## Build

```bash
latexmk -pdf main.tex        # or: pdflatex; bibtex main; pdflatex; pdflatex
```

Requires a TeX distribution with `amsmath`, `graphicx`, `booktabs`, `natbib`,
`hyperref`, `subcaption` (e.g. TeX Live, or upload to Overleaf).

## Structure

Introduction · Related work · Problem setup · Method (e-process construction +
frontier) · Theory (three theorems) · Experiments (ground-truth validation,
real-world data, learned neural world models on GPU) · Discussion & limitations ·
Conclusion · Reproduction appendix.
