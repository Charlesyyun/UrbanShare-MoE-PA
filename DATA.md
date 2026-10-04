# Data and input requirements

This is a code-only repository. No behavior data, policy calendars, SEIR result tables, figures, checkpoints, or third-party epidemic curves are included.

The behavioral pipeline expects authorized agent-day activity inputs matching the schema consumed by `preference_dataset.py`. The SEIR engine expects daily activity shares, feature metadata, hazard parameters, policy-calendar phase labels, and a seed list. These files must be provided through local command-line arguments or a local manifest.

The calibration utility in `SEIR/calibration/auto_calibrate_seir.py` additionally expects an archived official epidemic curve that can be parsed by `pandas.read_html`. Place the file at `official_curve.html` in the working input directory, or pass another local file with `--official_curve_html`. The source is intentionally not redistributed here because it is a third-party/reference artifact.

Because calibrated parameters and paper outputs have also been removed, the absence of that HTML file affects calibration only; it does not prevent importing or inspecting the SEIR code. A numerical reproduction requires all authorized inputs and parameters, not this repository alone.

Before using any input, confirm that collection, processing, and sharing of derived outputs are allowed by the applicable data agreement. Do not commit local inputs, private paths, identifiers, checkpoints, or generated results to this repository.

