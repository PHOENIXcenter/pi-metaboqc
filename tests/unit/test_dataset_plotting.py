"""Regression tests for dataset-construction plotting metadata."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pimqc.core import (
    DatasetSchema,
    MetaboDataset,
    ProcessingContext,
    SampleRoleLabels,
)
from pimqc.dataset import build_dataset
from pimqc.plotting.dataset import DatasetPlotter
from pimqc.plotting.payloads import DatasetPlotPayload, snapshot_dataset


def test_dataset_plotter_uses_resolved_dataset_schema() -> None:
    """Use labels resolved on the dataset, without reparsing configuration."""
    dataset = MetaboDataset(
        intensity=pd.DataFrame(
            [[1.0]],
            index=pd.Index(["F1"], name="Feature"),
            columns=pd.Index(["S1"], name="Specimen"),
        ),
        sample_metadata=pd.DataFrame(
            {"Class": ["Pool"], "Run": ["Run-A"], "Sequence": [1]},
            index=pd.Index(["S1"], name="Specimen"),
        ),
        feature_metadata=pd.DataFrame(index=pd.Index(["F1"], name="Feature")),
        schema=DatasetSchema(
            feature_id="Feature",
            sample_id="Specimen",
            sample_type="Class",
            batch="Run",
            injection_order="Sequence",
            biological_group=None,
            roles=SampleRoleLabels(actual="Study", blank="Solvent", qc="Pool"),
            sample_metadata_order=("Specimen", "Class", "Run", "Sequence"),
        ),
        context=ProcessingContext(),
    )

    payload = DatasetPlotPayload(data=snapshot_dataset(dataset))
    metadata = DatasetPlotter(payload)._get_plot_metadata()

    assert metadata == {
        "batch_column": "Run",
        "sample_type_column": "Class",
        "inject_order_column": "Sequence",
        "qc_label": "Pool",
        "actual_label": "Study",
        "blank_label": "Solvent",
    }


def test_build_dataset_renders_dashboard_with_output_dir(
    tmp_path: Path,
) -> None:
    """Keep dashboard export inside the dataset execution entry point."""
    metadata = pd.DataFrame(
        {
            "Sample Name": ["S1"],
            "Sample Type": ["QC"],
            "Batch": ["Batch-1"],
            "Inject Order": [1],
        }
    )
    intensity = pd.DataFrame(
        {"S1": [1.0]},
        index=pd.Index(["Feature-1"], name="Metabolite"),
    )

    with (
        patch.object(
            DatasetPlotter,
            "plot_dataset_dashboard",
            return_value=object(),
        ) as plot_dashboard,
        patch.object(
            DatasetPlotter,
            "save_and_show_pw",
        ) as save_dashboard,
    ):
        build_dataset(
            meta_info=metadata,
            int_df=intensity,
            output_dir=tmp_path,
        )

    plot_dashboard.assert_called_once_with()
    save_dashboard.assert_called_once()
    assert (
        Path(save_dashboard.call_args.kwargs["file_path"]).name
        == "Global_Acquisition_Overview.svg"
    )
