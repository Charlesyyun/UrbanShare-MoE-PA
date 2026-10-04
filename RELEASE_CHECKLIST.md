# Code-only release checklist

Before adding local inputs or using the code for a paper reproduction:

- verify that all behavior data, epidemic curves, calendars, parameters, checkpoints, and generated outputs stay outside Git;
- provide the final paper citation and contact information if this repository is cited;
- install Python 3.10 or newer dependencies from `requirements.txt`;
- run `python -m compileall -q .` and `python tools/validate_release.py`;
- inspect `git status` before every commit;
- enable GitHub secret scanning and do not attach restricted data to public issues.

The repository intentionally publishes source code only. Numerical reproduction requires authorized local inputs described in `DATA.md`.

