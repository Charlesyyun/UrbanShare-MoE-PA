# Contributing

1. Create a focused branch; keep raw data, checkpoints, and generated runs out of Git.
2. Use Python 3.10 or newer and install `requirements.txt`.
3. Run `python -m compileall -q .` and `python tools/validate_release.py`.
4. Update the workflow or architecture docs when public interfaces change.
5. Never commit personal paths, credentials, exact locations, POI identifiers,
   or non-redistributable source data.

Bug reports should include the command, Python/platform versions, and the
shortest reproducible error. Do not attach restricted data to public issues.
