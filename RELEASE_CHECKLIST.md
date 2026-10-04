# Public-release checklist

The code and compact experiment artifacts have been separated from the full
working directory. Before publishing, the repository owner should still:

- add the final paper title, authors, citation/BibTeX, and contact information;
- choose and add a software license, and separately confirm whether the derived
  trajectory files may be redistributed under that license;
- run `python tools/validate_release.py` after any last edit;
- inspect `git status` and the first commit before pushing;
- enable GitHub secret scanning and avoid committing any subsequently generated
  raw data, checkpoints, logs, or full scenario-run directories.

Suggested local initialization (run only after the points above are resolved):

```bash
cd Github_code
git init
git add .
git status
git commit -m "Initial reproducibility release"
```

No remote repository is created and nothing is pushed by the preparation
process.
