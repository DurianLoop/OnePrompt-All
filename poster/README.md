# OnePrompt-All A4 Poster for Overleaf

This package is an editable LaTeX recreation of the OnePrompt-All A4 portrait poster.

## Compile in Overleaf

1. Create a new blank Overleaf project.
2. Upload the contents of this ZIP, keeping the `figures/` folder.
3. Set the compiler to **pdfLaTeX**.
4. Compile `main.tex`.

## Main edit points

- Title, subtitle, author, and institution: near the beginning of `main.tex`.
- Section text: the `Problem`, `Method`, `Results`, and `Conclusion & Discussion` boxes.
- Benchmark table: search for `Table 1`.
- Figures: replace the files in `figures/` while keeping the same filenames.
- Layout: edit the fixed heights in `postersection` calls and the minipage width ratios.

## Included figures

- `figures/fig1_architecture.jpg`
- `figures/fig2_quadrant.png`
- `figures/fig3_workbench.jpg`

The poster is fixed to A4 portrait and designed to compile as a single page.
