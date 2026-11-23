## What changed

## Why
<!-- Link the issue. PRs into dev do not close issues automatically: close it by hand after the merge. -->

## How it was tested
<!-- Paste the real output: pytest (passed, skipped, coverage), flake8, pylint, radon, isort. For a stage result, the totals it produced. -->

## Skip ledger
<!-- Every skipped test and the issue it waits for, or "none". Acceptance tests are written before the code and skipped until their issue is done. -->

## Review
<!-- Filled after reading the diff as a reviewer, before merging. Say what was found and what was changed. -->
- Complexity: could anything be shorter, flatter, or use fewer parameters, layers or options? Nothing built for a stage that has not started.
- Bugs: edge cases, leakage of future or test data, wrong joins or units (bp versus percent).
- Readability: names say what things are, one job per function, docstrings say why.
- Tests: they test behaviour against the spec, not the implementation.

## Checklist
- [ ] Base branch is `dev` (or `main` for a release)
- [ ] Branch updated from `dev` before merging
- [ ] CI green: flake8, isort, radon (nothing above complexity 10), pylint 10.00, pytest with 90% coverage
- [ ] No data rows, secrets, models or logs committed; outputs are totals only
- [ ] No hard-coded paths or settings: paths in `.env`, settings in `config.yaml`
- [ ] Milestone set and issue linked
