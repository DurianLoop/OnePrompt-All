# AdaPromptSet latexmk Clean-Build Audit

Status: `FAIL`

Command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

## Checks

- latexmk_exit_zero: `FAIL`
- clean_copy_pdf_exists: `PASS`
- clean_copy_log_exists: `PASS`
- log_problem_scan_clean: `PASS`

## PDF

- exists: `True`
- bytes: `361714`
- pages: `8`

## Violations

- `latexmk_exit_zero`: latexmk_clean_build_failed
