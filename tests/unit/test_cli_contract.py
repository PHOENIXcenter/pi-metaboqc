"""Exercise CLI exit codes and input handling with the current pipeline API."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.mark.parametrize("report_generated, exit_code", [(True, 0), (False, 2)])
def test_cli_distinguishes_processing_from_report_success(
    tmp_path, monkeypatch, report_generated, exit_code
):
    """Distinguish report failure from successful scientific processing."""
    script = Path(__file__).parents[2] / "examples" / "run_pimqc.py"
    main = runpy.run_path(str(script))["main"]
    namespace = main.__globals__
    args = SimpleNamespace(
        quiet=True,
        meta="meta.csv",
        intensity="intensity.csv",
        config="parameters.toml",
        outdir=str(tmp_path / "outputs"),
    )
    monkeypatch.setitem(namespace, "parse_arguments", lambda: args)
    monkeypatch.setattr(namespace["pimqc"], "init", lambda **kwargs: None)
    monkeypatch.setitem(namespace, "load_pipeline_config", lambda **kwargs: {})
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pd.DataFrame())
    result = SimpleNamespace(
        report_generated=report_generated,
        data=SimpleNamespace(intensity=pd.DataFrame([[1.0]])),
    )
    monkeypatch.setitem(namespace, "run_pipeline", lambda **kwargs: result)
    if exit_code:
        with pytest.raises(SystemExit) as raised:
            main()
        assert raised.value.code == exit_code
    else:
        main()
    assert not Path(args.outdir).exists()


def test_cli_invalid_input_does_not_create_output_directory(
    tmp_path, monkeypatch
):
    """Leave no empty workspace behind when configuration loading fails."""
    script = Path(__file__).parents[2] / "examples" / "run_pimqc.py"
    main = runpy.run_path(str(script))["main"]
    namespace = main.__globals__
    target = tmp_path / "outputs"
    args = SimpleNamespace(
        quiet=True,
        meta="missing.csv",
        intensity="missing.csv",
        config="missing.toml",
        outdir=str(target),
    )
    monkeypatch.setitem(namespace, "parse_arguments", lambda: args)
    monkeypatch.setattr(namespace["pimqc"], "init", lambda **kwargs: None)

    def invalid(**kwargs):
        raise ValueError("Invalid input")

    monkeypatch.setitem(namespace, "load_pipeline_config", invalid)
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1
    assert not target.exists()
