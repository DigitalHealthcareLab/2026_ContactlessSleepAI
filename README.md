# 2026_ContactlessSleepAI

## How to run

```bash
export PSG_DATA=/path/to/data      # inputs (see psgscreen/paths.py)
export PSG_WORK=/path/to/work      # everything the code writes
PY=… PYB=… PYP=… PYF=… PYFC=… PYV=… bash scripts/run_all.sh
```

`scripts/run_all.sh` lists every step in execution order with the Python interpreter it runs under (`PY`, `PYB`,
`PYP`, `PYF`, `PYFC`, `PYV`); the environments are in `environment/ENVIRONMENTS.md`. A single step can be run on its
own with its command from that script, e.g. `python -m psgscreen.model.disorder`.
