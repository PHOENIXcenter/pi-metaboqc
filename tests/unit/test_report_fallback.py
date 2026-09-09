"""Verify report fallback ordering without invoking external installations."""

from pathlib import Path

import pytest

from pimqc.reporting import NarrativeStatsReporter


@pytest.mark.parametrize(
    "weasy_available, latex_available, successful, expected",
    [
        (True, True, "weasyprint", ["weasyprint"]),
        (False, True, "xelatex", ["xelatex"]),
        (True, True, "xelatex", ["weasyprint", "xelatex"]),
        (False, False, "html", ["html"]),
        (True, True, "html", ["weasyprint", "xelatex", "html"]),
        (False, False, None, ["html"]),
    ],
)
def test_weasyprint_preference_preserves_fallback_order(
    tmp_path,
    monkeypatch,
    weasy_available,
    latex_available,
    successful,
    expected,
):
    """A preferred engine is not an exclusive-engine requirement."""
    import pypandoc

    source = tmp_path / "report.md"
    source.write_text("# Report\n", encoding="utf-8")
    reporter = NarrativeStatsReporter(tmp_path)
    reporter._generated_md_paths = [source]
    monkeypatch.setattr(pypandoc, "get_pandoc_version", lambda: "3.8")
    monkeypatch.setattr(
        reporter, "_is_weasyprint_operational", lambda: weasy_available
    )
    monkeypatch.setattr(
        reporter, "_is_xelatex_available", lambda: latex_available
    )
    monkeypatch.setattr(reporter, "_is_rsvg_operational", lambda: True)
    calls = []

    def convert_file(*, outputfile, extra_args, **kwargs):
        engine = next(
            (
                arg.split("=", 1)[1]
                for arg in extra_args
                if arg.startswith("--pdf-engine=")
            ),
            "html",
        )
        calls.append(engine)
        if engine != successful:
            raise RuntimeError("Converter unavailable")
        Path(outputfile).write_text("Rendered test output", encoding="utf-8")

    monkeypatch.setattr(pypandoc, "convert_file", convert_file)
    assert reporter.export_report(pdf_engine="weasyprint") is bool(successful)
    assert calls == expected


def test_ci_validation_cannot_publish_on_push_or_manual_dispatch():
    """Keep validation triggers separate from the release-only publish job."""
    root = Path(__file__).parents[2]
    workflow = (root / ".github/workflows/publish.yml").read_text(
        encoding="utf-8"
    )
    for trigger in ("  push:", "  pull_request:", "  workflow_dispatch:"):
        assert trigger in workflow
    publish_job = workflow.split("  build-and-publish:", 1)[1]
    assert (
        "if: github.event_name == 'release' "
        "&& github.event.action == 'published'"
    ) in publish_job
    assert "needs: verify" in publish_job
