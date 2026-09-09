# Test Suites

- `unit/`: isolated Python behavior and lightweight component contracts.
- `integration/`: bundled-resource checks plus complete pipeline orchestration; the full pipeline module is marked `slow`.
- `quality/`: source documentation and structural-comment conventions.
- `reference/`: optional comparisons with system-R implementations. Reference code that is not distributed as a regular R package is embedded in its method-specific bridge test with upstream attribution and license terms.

The default `pytest` command runs unit, integration, and quality checks without initializing R or rpy2. Use `pytest -m "not slow"` for a quick development pass, or select a layer directly with `pytest tests/unit`, `pytest tests/integration`, or `pytest tests/quality`.

Run the optional cross-language suite explicitly with `pytest tests/reference` in an environment with R, rpy2, and the required R packages installed. To execute every Python and R check in one command, use `pytest tests/unit tests/integration tests/quality tests/reference`.

`pytest.ini` intentionally limits default discovery to the three native suites. Marker selection happens after collection, so `-m "not reference"` does not prevent importing `reference/conftest.py` and is not an isolation mechanism. Likewise, `pytest -m reference` alone does not discover the optional directory; use `pytest tests/reference -m reference` instead. No global marker exclusion is configured, so explicitly requested reference tests are not silently deselected.
