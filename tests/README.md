# Test suites

These tests protect the core software's algorithms, public API, optional R
adapters, plotting and reports. They do not require a supplementary checkout.

- `unit/`: isolated Python behavior and component contracts.
- `integration/`: bundled resources and pipeline orchestration; complete
  pipeline tests are marked `slow`.
- `quality/`: source documentation and structural-comment conventions.
- `reference/`: optional production R-adapter fidelity and robust LOESS checks.

## Native development checks

From the repository root:

```bash
pip install -e ".[test]"
python -m pytest -m "not slow"
```

Default discovery includes only unit, integration and quality suites. It does
not initialize R/rpy2. Omit `-m "not slow"` to include the full pipeline tests,
or select individual files when changing a specific component.

`unit/test_tutorial_api.py` checks the public notebook's syntax, API calls and
configuration boundary without executing its scientific workflow.
`unit/test_compact_plot_exports.py` checks the packaged compact plotting APIs
without private manuscript notebooks or figures.

## Optional R-adapter checks

Install R and the relevant original packages as described in the
[installation guide](../docs/r_backend.md), then:

```bash
pip install -e ".[test,r]"
python -m pytest tests/reference
```

These checks exercise installed original implementations, input/output scales,
missing-value handling, held-out QC support, RNG restoration, provenance and
serialization. Robust QC-RLSC tests cover current Python/R fitting contracts
and sparse-QC safety. For SERRF, set `PIMQC_SERRF_R_SOURCE` to the independently
obtained, verified source; tests never download it. Missing optional dependencies
or source files produce skips, not successful validation of those methods.

Independent Python-port comparison benchmarks live in the supplementary
repository, not this suite. Adapter fidelity checks do not establish universal
Python–R equivalence or biological efficacy.

To select all core suites explicitly:

```bash
python -m pytest tests/unit tests/integration tests/quality tests/reference
```

Marker filtering occurs after collection: `-m "not reference"` does not prevent
R imports if `tests/reference` is explicitly collected. Conversely,
`-m reference` alone does not discover the optional directory.

If the default temporary directory is inaccessible, pass `--basetemp` with a
dedicated scratch directory. Pytest may clear that directory's contents; never
use an input-data or results directory.
