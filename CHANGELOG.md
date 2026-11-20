# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Breaking changes are marked **BREAKING**.

## [Unreleased]

### Added

### Changed

- The package name is normalized to `torch-structure` (PEP 503). Installing and importing work as before (`pip install torch-structure`, `import torch_structure`).
- Supported Python versions are now 3.9–3.12 (`requires-python = ">=3.9,<3.13"`).

### Deprecated

### Removed

### Fixed

## [0.2.0] - 2026-10-29

### Added

- **Patterns.** Model repetitive structures from a single seed: rotational, translational and mirror patterns, which can be nested (`nest_patterns`). Nodes and edges added by pattern are copied to every position (`add_nodes_by_pattern`, `add_edges_by_pattern`, `add_edges_by_pattern_level`). Attributes can be set by pattern, for a whole expansion or along chosen levels (`set_node_attr_by_pattern_level`, `set_edge_attr_by_pattern_level`). `plot(pattern=True)` draws the patterns.
- **Frame analysis with the finite element method.** `fem()` performs a linear-elastic analysis of 3D frames and trusses. Members are described by `E`, `A`, `Iy`, `Iz` and `J`, and `release` pins the ends of a member. Results are `displacement`, `force`, `reaction_force` and `reaction_moment`.
- Per-axis supports: `support_xyz` restrains translations and `support_rxyz` restrains rotations.
- **A new plotter.** `plot()` / `plot_structure` can:
  - color, size and label edges and nodes by any attribute or tensor, with colormaps, categories and booleans;
  - show 3D and plan/elevation views;
  - draw loads, reactions and arbitrary node vectors;
  - draw a ghost geometry for comparison;
  - show only part of a structure with masks;
  - combine several plots on shared axes.
- `data.to_dtype(...)` changes the precision of a structure.
- `ts.utils.scipy_jacobian` wraps a torch objective for scipy's gradient-based optimizers.
- FDM now also returns `reaction_force`.

### Changed

- **Faster batched form-finding.** FDM on a batch now builds each graph's system as its own block and solves all blocks at once.
- **Optimization in float32 and float64, on any device.** The solvers compute in the dtype of the structure, and `set_edge_attr` works inside `torch.func` and autograd.
- **BREAKING:** Node names are stored as an encoded tensor of at most 8 bytes per name. They work the same way across the package and survive saving and loading. Longer names have to be shortened.
- **BREAKING:** The `support_condition` attribute is renamed to `support_xyz`.
- **BREAKING:** `plot()` is rebuilt on `plot_structure`, with a new set of arguments. `target_edge_index`, `target_color`, `target_node_size`, `support_marker_size`, `support_marker_offset`, `edge_index`, `is_support` and `force` are no longer supported and are ignored with a `FutureWarning`.
- The documentation has been reorganized: installation, a tutorial overview (topology, patterns, plotting) and an API reference summary.

### Deprecated

- The old `plot()` arguments `show_supports`, `show_load`, `load`, `target_coords`, `show_edge_indices`, `lw_constant` and `force_scale` still work, translated to their new equivalents, but raise a `FutureWarning` and will be removed in a future version.

### Removed

- **BREAKING:** `add_edge` no longer takes a `name` argument.
- `torch_scatter` is no longer a dependency. PyTorch Geometric's own scatter is used, and the compiled extensions are still used automatically if they are installed.

### Fixed

- A crash in batched FDM for graphs with a single free node.

## [0.1.0] - 2026-09-14

Initial release.

[Unreleased]: https://github.com/lazlo-bleker/torch-structure/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/lazlo-bleker/torch-structure/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lazlo-bleker/torch-structure/releases/tag/v0.1.0
