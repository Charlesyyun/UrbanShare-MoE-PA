# Calibration input

The automatic calibration script requires an archived official epidemic curve
that can be parsed by `pandas.read_html`. Place it at `official_curve.html` or
pass another file with `--official_curve_html`.

This third-party source is not redistributed in the repository. Its absence does
not affect the packaged counterfactual workflow, which uses the released
calibrated parameters in `../results/`.
