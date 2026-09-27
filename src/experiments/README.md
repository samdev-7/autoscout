# experiments/

One-off analysis and research scripts from building the pipeline: dataset audits,
distortion probes, calibration attempts that were superseded, tracker variants
(`link.py`, `repair.py`, `pipeline.py`, `run_tracks.py`, `run_assign.py`), sweeps and
plots. They are kept for reference and are **not** part of the pipeline described in
`INSTRUCTIONS.md`; many read artefacts that no longer exist or use an older layout.

They import the pipeline modules by directory, so run them with the source dir on the
path, from the repository root:

```bash
PYTHONPATH=src .venv/bin/python src/experiments/<script>.py
```
