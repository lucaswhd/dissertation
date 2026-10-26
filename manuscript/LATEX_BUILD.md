# LaTeX build requirements

The controlled final source package uses the verified native v6 layout: US Letter, the original fonts and margins, the established front-matter and chapter pagination, and the original equation/reference style. The build entry point is `main.tex`. A TeX Live installation with `latexmk`, `pdflatex`, `newtxtext`, `newtxmath`, `microtype`, `cleveref`, `tocloft`, `titlesec`, `caption`, and `hyperref` is required.

On an authorised machine with the source archive extracted, run from its project root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The command was executed on the research server with its existing TinyTeX toolchain; the corrected source produced a 96-page PDF before the final repository reference was added. The final release receipt records the subsequent build hash and page checks. External BEHAVIOR/OmniGibson images in the source package require separate redistribution review; this repository intentionally does not include the source package.

Check the build log for undefined citations/references, overfull boxes, missing glyphs, and font substitution. Inspect the actual rendered Figure 3.5 page: its cached map and task-area panel should both be upright, with horizontal labels, world axes, and the caption below. Check pagination and the List of Figures/Tables after any edit. Compilation alone is not a submission approval.
