"""Central configuration for torch_structure plotting.

Collects the visual, numeric and showal settings ``plot_3d`` / ``plot_xz``
draw with, grouped by property type — the same way a project config.py groups
settings by concern. Change a value here and it changes every subsequent call;
``plot_3d`` / ``plot_xz`` take only the ``plot_data`` dict plus ``title`` and
``legend`` — every other setting (the ``show`` flags, ``force_scale``,
``lw_scale``, ``support_marker_size``, ``support_marker_offset``, ...) is read
straight from here, not passed per call.
"""

PLOT_CONFIG = {
    'show': {
        'show_supports': False,
        'show_load': False,
        'show_residual_forces': False,
        'show_axes': False,
        'equal_axes': True,
        'lw_constant': False,
        'show_edge_indices': False,           # plot_3d only
        'show_deck': False,                   # plot_3d only
    },
    'color': {
        'edge_tension': '#E40714',                    # Edges/legend entry for force > 0
        'edge_compression': '#0578BF',                 # Edges/legend entry for force <= 0
        'edge_untensioned': 'gray',                     # Edges drawn with no force given ("Target"/"Converged Geometry")
        'vector_load': '#007F00',                       # External load arrows
        'vector_residual': 'purple',                    # Residual force arrows
        'node_support': 'black',                        # support markers, both views
        'node_highlight': '#FFEC99',                     # plot_xz "Considered in Step" markers
        'face_deck': 'grey',                             # Deck quad fill
        'face_deck_edge': 'k',                           # Deck quad outline
        'text_edge_index': 'black',                      # show_edge_indices labels
    },
    'marker': {
        'node_support': '^',
        'node_highlight': 'o',
    },
    'size': {
        'support_marker': {'3d': 6, 'xz': 3},           # support_marker_size defaults
        'node_highlight_marker': 4,
        'edge_index_fontsize': 8,
        'title_fontsize_xz': 16,
    },
    'line_width': {
        'edge_scale': 1.0,                               # lw_scale default (both views)
        'face_deck_edge': 0.5,
    },
    'line_style': {
        'edge_forced': '-',
        'edge_untensioned': ':',                          # only visible on plot_xz (plot_3d never sets a linestyle)
    },
    'offset': {
        'support_marker': {'3d': 0.02, 'xz': 0.12},      # support_marker_offset defaults
    },
    'alpha': {
        'face_deck': 0.4,
    },
    'vector': {
        'force_scale': {'3d': 1.0, 'xz': 0.6},           # force_scale defaults
        'load_arrow_length_3d': 1.5,                      # draw_arrow `length=` for external load, 3D only
        'load_arrow_head_ratio_3d': 0.5,                  # draw_arrow `arrow_length_ratio=` for external load, 3D only
        'xz_load_min_magnitude': 1e-5,                    # below this, a 2D load vector is not drawn
    },
    'legend': {
        'frameon': False,
        'ncol_xz': 3,
        'loc_xz': 'lower center',
        'bbox_to_anchor_xz': (0.5, -2.8),
    },
    'title': {
        'pad_xz': 20,
    },
    'figure': {
        'figsize': (10, 8),
    },
    'save': {
        'file_format': '.png',
        'bbox_inches': 'tight',
    },
}
