from dataclasses import dataclass


@dataclass
class PlotSettings:
    """
    Visual settings for `plot_data`. Pass a customized instance to `plot_data`
    (or `StructData.plot`) via the `settings` argument to override colors,
    marker styles, and sizes without touching function call sites.
    """

    # Node markers (uniform across all nodes: white fill, black border)
    node_marker: str = "o"
    node_size: float = 7.0
    node_color: str = "#FFFFFF"
    node_outline_color: str = "#000000"
    node_outline_width: float = 0.5

    # Node labels
    node_label_color: str = "#000000"
    node_label_size: float = 10.0
    node_label_offset: float = 0.15

    # Support markers
    support_marker: str = "^"
    support_marker_size: float = 10.0
    support_marker_color: str = "#009650"
    support_marker_outline_color: str = "#000000"

    # Edge colors (tension / compression)
    edge_tension_color: str = "#E40714"
    edge_compression_color: str = "#0578BF"
    edge_alpha: float = 1.0

    # Load / support-reaction vectors
    vector_load_color: str = "#007F00"
    vector_reaction_color: str = "#FF8800"
    vector_length: float = 0.5
    vector_line_width: float = 2.5
    vector_arrow_ratio: float = 0.1
    vector_alpha: float = 0.75

    # Draw order (higher draws on top)
    zorder_node: int = 4
    zorder_edge: int = 1
    zorder_vector: int = 2
    zorder_label: int = 20
