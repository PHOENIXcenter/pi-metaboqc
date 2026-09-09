"""Grid-column selection and the disabled legacy SVG composition implementation.

SVG stitching is retained below as comments for reference. Active QA reports
are drawn from audits by the assessment patchworklib comparison renderer.
"""

import math
# from pathlib import Path
# from typing import Optional, Union

# from loguru import logger

# from .plot_utils import dashboard_display_width

# CairoSVG defaults to a 96 dpi raster surface.  Report previews are displayed
# at a reduced CSS width, so render the stitched SVG at a larger pixel surface
# before handing it to the notebook image widget.
STITCHED_PNG_SCALE = 3.0


# =============================================================================
# Atomic Utility Functions
# =============================================================================
def _get_optimal_cols(n_docs: int, max_cols: int = 4) -> int:
    """Calculates optimal grid columns for subplot layout.

    Args:
        n_docs (int): Total number of documents to stitch.
        max_cols (int): Maximum allowed columns. Defaults to 4.

    Returns:
        int: Optimal number of columns bounded by max_cols.
    """
    if n_docs <= 0:
        return 1

    # Preset aesthetic mappings for typical plot counts to avoid
    # disproportionate grid aspect ratios (e.g., forcing 2x2 for 4 plots)
    layout_map = {
        1: 1,
        2: 2,
        3: 3,
        4: 2,
        5: 3,
        6: 3,
        7: 4,
        8: 4,
        9: 3,
        10: 4,
        11: 4,
        12: 4,
    }

    if n_docs in layout_map:
        cols = layout_map[n_docs]
    else:
        # Fallback for dynamic calculation on arbitrary large numbers
        cols = math.ceil(math.sqrt(n_docs))

    return min(max_cols, cols)


# Legacy SVG stitching retained for reference; report QA now uses patchworklib.
# def stitch_svg_grids(
#     svg_paths: list,
#     file_path: str,
#     cols: Union[int, str] = "auto",
#     max_cols: int = 4,
#     show_plot: bool = True,
#     save_format: Optional[Union[str, list, tuple]] = ("svg", "pdf"),
#     display_format: str = "png",
#     width: Optional[Union[int, str]] = "auto",
#     legend_paths: Optional[list] = None,
#     suptitle: Optional[str] = None,
#     target_width_cm: Optional[float] = 17.8,
# ) -> bool:
#     """
#     Stitches multiple SVGs into a grid, saves to disk, and displays inline.
#
#     Assembles individual SVG plots into a unified master SVG grid. Provides
#     dynamic file conversion to PDF/PNG via CairoSVG and aligns with the
#     project's standard Jupyter rendering logic (supporting native VS Code
#     image toolbars and responsive layouts).
#
#     Args:
#     svg_paths (list): List of paths to source SVG subplots.
#     file_path (str): Destination base path for the stitched file(s).
#     cols (Union[int, str]): Number of columns, or 'auto'.
#     max_cols (int): Maximum allowed columns when using 'auto'.
#     show_plot (bool): Whether to render the result in Jupyter/VS Code.
#     save_format (Optional[Union[str, list, tuple]]): Disk export formats.
#     display_format (str): Inline format for notebook preview ('svg' or 'png').
#     width (Optional[Union[int, str]]): CSS width for Jupyter display.
# ``"auto"``
#     selects the same 60/40/20% policy used by stage dashboards.
#     legend_paths (Optional[list]): Candidate shared-legend SVGs. At most
#     one valid legend is placed beside the main grid.
#     suptitle (Optional[str]): Shared title placed above the stitched grid.
#     target_width_cm (Optional[float]): Physical target width of the assembled
#     figure in centimetres. Source panels and sidecar legends are scaled
#     together so their typography remains consistent. ``None`` preserves
#     the native source width.
#
#     Returns:
#     bool: True if the stitching and saving were successful, False otherwise.
#
#     """
#     try:
#         import svgutils.transform as sg
#     except ImportError:
#         logger.error("Please install 'svgutils' via 'pip install svgutils'.")
#         return False
#
#     try:
#         # Validate and filter input paths
#         valid_paths = [
#             p
#             for p in svg_paths
#             if Path(p).exists() and Path(p).stat().st_size > 0
#         ]
#         if not valid_paths:
#             return False
#
#         # Calculate optimal grid dimensions
#         n_docs = len(valid_paths)
#         active_cols = (
#             _get_optimal_cols(n_docs, max_cols) if cols == "auto" else
# int(cols)
#         )
#         rows = (n_docs + active_cols - 1) // active_cols
#         display_width = (
#             dashboard_display_width(rows, active_cols)
#             if width is None or str(width).lower() == "auto"
#             else width
#         )
#
#         # Robust UTF-8 reading to prevent GBK codec errors on Windows
#         svg_figs = []
#         for p in valid_paths:
#             with open(p, "r", encoding="utf-8") as f:
#                 svg_figs.append(sg.fromstring(f.read()))
#
#         # Extract maximum dimensions for uniform grid alignment
#         def parse_dim(val: str) -> float:
#             import re
#
#             match = re.search(r"(\d+\.?\d*)", str(val))
#             return float(match.group(1)) if match else 0.0
#
#         def figure_dims(figure: object) -> tuple[float, float]:
#             """Read width/height, falling back to the SVG viewBox."""
#             width = parse_dim(getattr(figure, "width", None))
#             height = parse_dim(getattr(figure, "height", None))
#             if width > 0 and height > 0:
#                 return width, height
#             view_box = (
#                 str(figure.root.get("viewBox", "")).replace(",", " ").split()
#             )
#             if len(view_box) >= 4:
#                 return parse_dim(view_box[2]), parse_dim(view_box[3])
#             return width, height
#
#         dimensions = [figure_dims(f) for f in svg_figs]
#         max_w = max(width for width, _ in dimensions)
#         max_h = max(height for _, height in dimensions)
#
#         # A sidecar legend is deliberately excluded from the main grid. This
#         # preserves the stage ordering while avoiding repeated legend blocks.
#         legend_fig = None
#         if legend_paths:
#             for legend_path in legend_paths:
#                 path = Path(legend_path)
#                 if not path.exists() or path.stat().st_size == 0:
#                     continue
#                 with open(path, "r", encoding="utf-8") as f:
#                     legend_fig = sg.fromstring(f.read())
#                 break
#
#         legend_w, legend_h = (
#             figure_dims(legend_fig) if legend_fig is not None else (0.0, 0.0)
#         )
#
#         # Source QA panels are intentionally compact. Reserve explicit gutters
#         # between their SVG canvases so axis labels and tick labels are never
#         # covered by the white background of the following panel. The outer
#         # padding also protects the first-column y label and last-row x label.
#         panel_gap_x = max(4.0, max_w * 0.04) if active_cols > 1 else 0.0
#         panel_gap_y = max(6.0, max_h * 0.07) if rows > 1 else 0.0
#         outer_pad_x = max(3.0, max_w * 0.025)
#         outer_pad_y = max(3.0, max_h * 0.025)
#         native_grid_w = (
#             max_w * active_cols
#             + panel_gap_x * max(0, active_cols - 1)
#             + 2.0 * outer_pad_x
#         )
#         native_grid_h = (
#             max_h * rows + panel_gap_y * max(0, rows - 1) + 2.0 * outer_pad_y
#         )
#         native_title_h = max(18.0, max_h * 0.13) if suptitle else 0.0
#         native_total_h = native_grid_h + native_title_h
#
#         # Seven correction candidates naturally leave one slot in a 2 x 4
#         # grid. Reuse it for the legend; six candidates retain a 2 x 3 grid
#         # with an independent legend column on the right.
#         embed_legend = (
#             legend_fig is not None
#             and n_docs == 7
#             and active_cols == 4
#             and rows == 2
#         )
#         legend_col_w = 0.0
#         if legend_fig is not None and not embed_legend:
#             # Sidecars are tightly cropped at export. Keep only a small gutter
#             # here, rather than expanding the legend to a fraction of plot
#             # width.
#             legend_col_w = legend_w + max(2.0, max_w * 0.015)
#
#         native_total_w = native_grid_w + legend_col_w
#         if target_width_cm is None:
#             scale = 1.0
#             total_w = native_total_w
#         else:
#             # SVG user units emitted by Matplotlib are points, so convert the
#             # requested centimetre width to points before applying one common
#             # scale to plots, titles, and sidecar legends.
#             target_width = float(target_width_cm) * 72.0 / 2.54
#             scale = target_width / native_total_w if native_total_w > 0
# else 1.0
#             total_w = target_width
#         title_h = native_title_h * scale
#         total_h = native_total_h * scale
#
#         # Initialize the master canvas with a white background
#         fig = sg.SVGFigure(f"{total_w}", f"{total_h}")
#         bg_svg = sg.fromstring(
#             '<svg><rect width="100%" height="100%" fill="white"/></svg>'
#         ).getroot()
#
#         plots = [bg_svg]
#
#         if suptitle:
#             title = sg.TextElement(
#                 (native_grid_w * scale) / 2,
#                 title_h * 0.66,
#                 str(suptitle),
#                 size=max(10.0, 9.0 * scale),
#                 weight="bold",
#             )
#             title.root.set("text-anchor", "middle")
#             title.root.set(
#                 "style",
#                 "font-family: Helvetica, Arial, Liberation Sans, Nimbus
# Sans, "
#                 "DejaVu Sans, sans-serif; fill: #262626;",
#             )
#             plots.append(title)
#
#         # Position each subplot into the calculated grid slot
#         for i, s_fig in enumerate(svg_figs):
#             row, col = divmod(i, active_cols)
#             plot = s_fig.getroot()
#             x_pos = (outer_pad_x + col * (max_w + panel_gap_x)) * scale
#             y_pos = (
#                 title_h + (outer_pad_y + row * (max_h + panel_gap_y)) * scale
#             )
#             plot.moveto(x_pos, y_pos, scale_x=scale)
#             plots.append(plot)
#
#         if legend_fig is not None:
#             legend = legend_fig.getroot()
#             if embed_legend:
#                 row, col = divmod(n_docs, active_cols)
#                 x_pos = (
#                     outer_pad_x
#                     + col * (max_w + panel_gap_x)
#                     + max(2.0, max_w * 0.015)
#                 ) * scale
#                 y_pos = (
#                     title_h
#                     + (outer_pad_y + row * (max_h + panel_gap_y)) * scale
#                     + max(0.0, (max_h - legend_h) / 2) * scale
#                 )
#             else:
#                 x_pos = (native_grid_w + max(2.0, max_w * 0.015)) * scale
#                 y_pos = (
#                     title_h + max(0.0, (native_grid_h - legend_h) / 2) * scale
#                 )
#             legend.moveto(x_pos, y_pos, scale_x=scale)
#             plots.append(legend)
#
#         # Assemble and set viewBox
#         fig.append(plots)
#         fig.root.set("viewBox", f"0 0 {total_w} {total_h}")
#         # Keep a physical width for PDF/Illustrator consumers while retaining
#         # the viewBox coordinates used for responsive SVG placement.
#         fig.root.set("width", f"{total_w}pt")
#         fig.root.set("height", f"{total_h}pt")
#
#         # Extract raw byte string for memory-based format conversion
#         merged_svg_bytes = fig.to_str()
#         merged_svg_str = merged_svg_bytes.decode("utf-8")
#
#         # Format Normalization & Physical Storage Logic
#         filepath_str = str(file_path)
#         base_path = (
#             filepath_str.rsplit(".", 1)[0]
#             if "." in Path(filepath_str).name
#             else filepath_str
#         )
#
#         if save_format:
#             format_list = (
#                 [save_format]
#                 if isinstance(save_format, str)
#                 else list(save_format)
#             )
#
#             for fmt in format_list:
#                 clean_fmt = fmt.lower().strip(".")
#                 out_path = f"{base_path}.{clean_fmt}"
#
#                 if clean_fmt == "svg":
#                     with open(out_path, "wb") as f:
#                         f.write(merged_svg_bytes)
#                 else:
#                     # Leverage CairoSVG for dynamic vector/raster conversion
#                     try:
#                         import cairosvg
#
#                         if clean_fmt == "pdf":
#                             cairosvg.svg2pdf(
#                                 bytestring=merged_svg_bytes, write_to=out_path
#                             )
#                         elif clean_fmt == "png":
#                             cairosvg.svg2png(
#                                 bytestring=merged_svg_bytes,
#                                 write_to=out_path,
#                                 scale=STITCHED_PNG_SCALE,
#                             )
#                     except ImportError:
#                         logger.error(
#                             f"Cannot save {clean_fmt.upper()}: 'cairosvg' is "
#                             "not installed. Run `pip install cairosvg`."
#                         )
#
#         # Environment-safe Jupyter rendering logic
#         if show_plot:
#             try:
#                 from ..runtime import is_jupyter
#
#                 if is_jupyter():
#                     from IPython.display import HTML, Image, display
#                     import re
#
#                     display_fmt = display_format.lower()
#                     if display_fmt not in ["svg", "png"]:
#                         display_fmt = "svg"
#
#                     w_css = (
#                         f"{display_width}px"
#                         if isinstance(display_width, int)
#                         else (display_width if display_width else "100%")
#                     )
#
#                     if display_fmt == "svg":
#                         # Strip absolute dimensions for responsive UI preview
#                         preview_svg = re.sub(
#                             r'(<svg[^>]*?\s)width="[^"]+"',
#                             r'\1width="100%"',
#                             merged_svg_str,
#                             count=1,
#                         )
#                         preview_svg = re.sub(
#                             r'(<svg[^>]*?\s)height="[^"]+"',
#                             r'\1height="auto"',
#                             preview_svg,
#                             count=1,
#                         )
#
#                         container_style = (
#                             f"width:{w_css}; max-width:100%; margin: 0 auto; "
#                             f"height:auto; background-color: white;"
#                         )
#                         display(
#                             HTML(
#                                 f'<div style="{container_style}">'
#                                 f"{preview_svg}</div>"
#                             )
#                         )
#
#                     elif display_fmt == "png":
#                         try:
#                             import cairosvg
#
#                             # Force solid white background for VS Code
# dark mode
#                             png_data = cairosvg.svg2png(
#                                 bytestring=merged_svg_bytes,
#                                 background_color="white",
#                                 scale=STITCHED_PNG_SCALE,
#                             )
#                             # Native Image rendering activates VS Code
# toolbars
#                             display(Image(data=png_data, width=display_width))
#                         except ImportError:
#                             logger.error(
#                                 "Cannot preview PNG: 'cairosvg' missing. "
#                                 "Falling back to SVG display."
#                             )
#                             # Safe fallback to SVG if Cairo is missing
#                             container_style = (
#                                 f"width:{w_css}; max-width:100%; "
#                                 "margin: 0 auto; "
#                                 f"height:auto; background-color: white;"
#                             )
#                             display(
#                                 HTML(
#                                     f'<div style="{container_style}">'
#                                     f"{merged_svg_str}</div>"
#                                 )
#                             )
#
#             except Exception as e:
#                 # Silently catch to keep terminal logs clean in headless mode
#                 logger.debug(f"Jupyter rendering bypassed: {e}")
#
#         return True
#
#     except Exception as e:
#         logger.error(f"Failed to stitch SVG grid: {e}")
#         return False
