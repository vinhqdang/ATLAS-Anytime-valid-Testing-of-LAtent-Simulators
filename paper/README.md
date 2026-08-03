# ATLAS — manuscript (Elsevier, Pattern Recognition)

Prepared with the official Elsevier `elsarticle` class, author-year (Harvard)
reference style (`elsarticle-harv.bst`) — for submission to *Pattern Recognition*.

- `main.tex`   — manuscript source.
- `references.bib` — bibliography.
- `sn-jnl.cls`, `sn-basic.bst` — legacy Springer Nature class/style, retained for
  reference from an earlier submission target; not used by `main.tex`.
- `figures/`   — figures.
- `main.pdf`   — compiled PDF.
- `highlights.pdf` — the separate Elsevier "Highlights" submission file (3-5 bullet
  points, ≤85 characters each, no author names); regenerate with
  `python3 make_highlights.py` after editing `make_highlights.py`.

## Build
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
or `latexmk -pdf main.tex`.
