"""Keep the native reference aligned with executable public contracts."""

import ast
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import re

import numpy as np
import pandas as pd

from pimqc import MetaboDataset
from pimqc.config import PipelineConfig


ROOT = Path(__file__).parents[2]
DOCUMENT = ROOT / "docs" / "native_api.md"


def _examples():
    return re.findall(
        r"^```python\n(.*?)^```",
        DOCUMENT.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )


@dataclass
class _Instance:
    """Resolve instance method signatures without constructing processors."""

    cls: type


def _resolve(node, namespace):
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.Call):
        target = _resolve(node.func, namespace)
        if inspect.isclass(target):
            return _Instance(target)
    if isinstance(node, ast.Attribute):
        parent = _resolve(node.value, namespace)
        if isinstance(parent, _Instance):
            method = getattr(parent.cls, node.attr)
            return method.__get__(object(), parent.cls)
        if parent is not None:
            return getattr(parent, node.attr)
    return None


def test_reference_examples_import_and_bind_public_calls():
    """Check real imports, constructors, and methods without costly runs."""
    namespace = {}
    checked = 0
    for index, source in enumerate(_examples()):
        tree = ast.parse(source, filename=f"native API example {index}")
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                if isinstance(statement, ast.Import):
                    for alias in statement.names:
                        namespace[alias.asname or alias.name] = (
                            importlib.import_module(alias.name)
                        )
                else:
                    module = importlib.import_module(statement.module)
                    for alias in statement.names:
                        namespace[alias.asname or alias.name] = getattr(
                            module, alias.name
                        )
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                target = _resolve(node.func, namespace)
                if target is None:
                    continue
                assert not any(isinstance(a, ast.Starred) for a in node.args)
                assert all(kw.arg is not None for kw in node.keywords)
                inspect.signature(target).bind(
                    *[object() for _ in node.args],
                    **{kw.arg: object() for kw in node.keywords},
                )
                checked += 1
            if isinstance(statement, ast.Assign):
                value = _resolve(statement.value, namespace)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        namespace[target.id] = value
    assert checked >= 20


def test_reference_config_inventory_matches_schema():
    """Reject undocumented or deleted configuration fields in the table."""
    source = DOCUMENT.read_text(encoding="utf-8")
    for name, field in PipelineConfig.model_fields.items():
        row = re.search(rf"^\| `{name}` \| (.+) \|$", source, re.MULTILINE)
        assert row is not None, name
        documented = set(re.findall(r"`([^`]+)`", row.group(1)))
        assert documented == set(field.annotation.model_fields), name


def test_reference_headings_and_links():
    """Validate hierarchical numbering and local/README navigation."""
    source = DOCUMENT.read_text(encoding="utf-8")
    prose = re.sub(
        r"^```.*?^```",
        "",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    headings = re.findall(r"^(#{1,6}) (.+)$", prose, re.MULTILINE)
    previous = []
    anchors = set()
    for marks, title in headings:
        match = re.match(r"(\d+(?:\.\d+)*)\s+", title)
        assert match is not None, title
        number = [int(part) for part in match.group(1).split(".")]
        assert len(number) == len(marks)
        if len(number) > len(previous):
            assert len(number) == len(previous) + 1
            assert number[:-1] == previous and number[-1] == 1
        else:
            assert number[:-1] == previous[: len(number) - 1]
            assert number[-1] == previous[len(number) - 1] + 1
        previous = number
        anchor = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        anchors.add(anchor)
    for target in re.findall(r"\]\(([^)]+)\)", source):
        if target.startswith("#"):
            assert target[1:] in anchors
        elif not target.startswith("https://"):
            assert (DOCUMENT.parent / target).is_file()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    prefix = "https://github.com/PHOENIXcenter/pi-metaboqc/blob/main/"
    for path in (
        "examples/interactive_tutorial.ipynb",
        "examples/run_pimqc.py",
        "docs/native_api.md",
    ):
        assert f"]({prefix}{path}) |" in readme
        assert (ROOT / path).is_file()


def test_reference_filter_and_normalization_examples(tmp_path, monkeypatch):
    """Run selected documentation examples on a small real dataset."""
    rng = np.random.default_rng(42)
    ids = [f"S{i}" for i in range(12)]
    samples = pd.DataFrame(
        {
            "Sample Type": ["QC"] * 6 + ["Sample"] * 6,
            "Batch": ["B1"] * 12,
            "Inject Order": list(range(1, 13)),
        },
        index=ids,
    )
    intensity = pd.DataFrame(
        rng.lognormal(mean=5, sigma=0.1, size=(40, 12)),
        index=[f"F{i}" for i in range(40)],
        columns=ids,
    )
    dataset = MetaboDataset.from_tables(intensity, samples)
    namespace = {"raw_dataset": dataset, "params": {}}
    monkeypatch.chdir(tmp_path)
    markers = (
        "from pimqc import FeatureMissingValueFilter,",
        "from pimqc import FeatureQualityFilter",
        "from pimqc import MissingValueImputer",
        "from pimqc import DataNormalizer",
        "from pimqc.serialization import",
    )
    for marker in markers:
        source = next(s for s in _examples() if marker in s)
        tree = ast.parse(source)
        # Keep the published argument shape; disable rendering in this smoke.
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "output_dir":
                node.value = ast.Constant(value=None)
        ast.fix_missing_locations(tree)
        namespace["corrected_dataset"] = namespace.get(
            "mv_filtered_dataset", dataset
        )
        exec(compile(tree, str(DOCUMENT), "exec"), namespace)
    assert namespace["imputation_result"].audit.skipped
    restored = namespace["restored_dataset"]
    pd.testing.assert_frame_equal(
        restored.intensity, namespace["normalized_dataset"].intensity
    )
    assert namespace["restored_audit"].selection["is_auto"] is False
