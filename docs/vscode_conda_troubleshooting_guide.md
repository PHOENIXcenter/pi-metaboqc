# VS Code and Conda troubleshooting — pi-metaboqc 1.5.0

This guide covers the public [interactive tutorial](../examples/interactive_tutorial.ipynb) and [CLI example](../examples/run_pimqc.py). Follow the [installation instructions](../README.md#-installation) first. The optional R backend has its own [setup guide](r_backend.md).

## Select and verify the environment

Select the `metaboqc` interpreter with **Python: Select Interpreter**. For notebooks, also select the matching kernel: a terminal's active environment does not determine the notebook kernel. Open a fresh terminal, activate the environment, and check the actual executable and installed package:

```bash
conda activate metaboqc
python -c "import sys, pimqc; print(sys.executable); print(pimqc.__version__)"
```

If environment activation is missing, review VS Code's `python.terminal.activateEnvironment` and `terminal.integrated.inheritEnv` settings, then close the old terminal and open a new one. Activation behavior can also depend on the installed Python environment extensions. Selecting an interpreter alone does not prove that every external executable or shared library is available.

The notebook uses the packaged configuration without additional worker caps. Adjust `n_jobs` in your configuration for your machine; `-1` requests the available CPU workers. Saved notebook displays are historical results, not evidence that your local environment has completed the workflow.

## Diagnose report conversion separately from processing

Generating Markdown reports does not require a PDF converter. Exporting HTML or PDF uses Pandoc; PDF export additionally needs WeasyPrint or XeLaTeX. The recommended Conda setup is:

```bash
conda install -c conda-forge pandoc weasyprint tinycss2 librsvg -y
pandoc --version
weasyprint --version
```

On Windows, `where.exe pandoc` and `where.exe weasyprint` help check which executables the terminal resolves. Shared-library errors such as a missing GObject or Pango library can indicate incomplete activation, a missing dependency, or incompatible library versions; they are not automatically a VS Code defect.

Compare these checks in a newly activated Conda terminal and the VS Code terminal. Correct the selected environment or its dependencies before changing system paths. The report layer checks the active environment's `Library/bin` on Windows, but this cannot repair missing or incompatible libraries. Avoid committing machine-specific paths to the repository.

With `pdf_engine="weasyprint"`, export tries WeasyPrint, then XeLaTeX, then HTML. XeLaTeX is optional and needs `rsvg-convert` for SVG assets. Tools are not downloaded automatically. Successful HTML fallback returns `True`, so inspect the generated file and log to determine the actual format. CLI exit code `2` means scientific processing completed but report export failed; completed stage outputs remain available.

The tutorial defaults to Markdown generation with `EXPORT_PDF = False`. Enable that switch only when you want final report conversion. It does not disable scientific computation or the notebook's diagnostic displays.

## PowerShell activation policy

If PowerShell reports that script execution is disabled, inspect the effective policies:

```powershell
Get-ExecutionPolicy -List
```

You can use Anaconda Prompt or another approved shell without changing the PowerShell policy. If local policy permits and you choose to allow local activation scripts, the following setting affects only your user account and normally does not require an administrator console:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Organizational Group Policy can override this setting. Do not bypass an enforced policy; ask your administrator for an approved activation route. Reopen the terminal after changing activation settings.

## Before running

- Confirm that the terminal and notebook kernel use the intended Python environment.
- Verify the package version and load a valid TOML or JSON configuration.
- Check report converters only if report conversion is required.
- For `implementation="r"`, check R, the needed packages and any external SERRF source using the [R backend guide](r_backend.md).
- Save the executed notebook to retain visible tables, plots, warnings and logs in VS Code.
