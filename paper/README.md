# ATLAS — manuscript (Springer Nature, Autonomous Robots)

Prepared with the official Springer Nature LaTeX class (`sn-jnl.cls`, v3.1 Dec 2024),
author-year "Basic" reference style (`sn-basic.bst`) — the style used by
*Autonomous Robots*.

- `main.tex`   — manuscript source (single-column `sn-basic`; switch to `iicol` for
  the two-column production layout via the document-class option).
- `references.bib` — bibliography.
- `sn-jnl.cls`, `sn-basic.bst` — Springer Nature class and bibliography style.
- `figures/`   — figures.
- `main.pdf`   — compiled PDF.

## Build
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
or `latexmk -pdf main.tex`.
