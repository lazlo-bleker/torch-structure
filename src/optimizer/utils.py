import vtk
import networkx as nx
import numpy as np


def export_graph_to_vtp(G: nx.Graph, out_path: str, compress: bool = True):
    """
    Export a (Multi)Graph to a VTK PolyData (.vtp) for ParaView.

    Nodes:
      - must have G.nodes[n][coord_key] as (x,y) or (x,y,z)
      - all node attributes are written to PointData

    Edges:
      - all edge attributes are written to CellData
      - supports Graph/DiGraph and MultiGraph/MultiDiGraph

    Attribute types:
      - numeric scalars (int/float/bool)
      - numeric vectors (length 2–4) as separate components
      - strings (stored as vtkStringArray)
      - missing values become NaN (numeric) or "" (string)
    """
    # --- map nodes to contiguous point indices
    nodes = list(G.nodes())
    nid = {n: i for i, n in enumerate(nodes)}

    # --- build vtkPoints from node coords
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(nodes))

    def _to_xyz(v):
        if len(v) == 3:
            return (float(v[0]), float(v[1]), float(v[2]))
        else:
            return (np.nan, np.nan, np.nan)

    for n in nodes:
        coord = G.nodes[n].get("coords", None)
        x, y, z = _to_xyz(coord)
        points.SetPoint(nid[n], x, y, z)

    # --- prepare PolyData
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    # --- collect edges (support MultiGraph)
    is_multi = G.is_multigraph()
    edges_list = []
    if is_multi:
        for u, v, k, data in G.edges(keys=True, data=True):
            edges_list.append((u, v, k, data))
    else:
        for u, v, data in G.edges(data=True):
            edges_list.append((u, v, None, data))

    # --- make a vtkCellArray of lines
    lines = vtk.vtkCellArray()
    for u, v, _, _ in edges_list:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, nid[u])
        line.GetPointIds().SetId(1, nid[v])
        lines.InsertNextCell(line)
    poly.SetLines(lines)

    # --- helpers to create VTK arrays
    def _make_numeric_array(name, ncomp, nitems, dtype=float):
        if dtype in (int, np.int32, np.int64):
            arr = vtk.vtkIntArray()
        else:
            arr = vtk.vtkDoubleArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(ncomp)
        arr.SetNumberOfTuples(nitems)
        return arr

    def _make_string_array(name, nitems):
        arr = vtk.vtkStringArray()
        arr.SetName(name)
        arr.SetNumberOfValues(nitems)
        return arr

    def _write_point_attributes():
        # collect all node attribute keys
        keys = set()
        for n in nodes:
            keys.update(G.nodes[n].keys())
        keys.discard("coords")  # coords are already used
        if not keys:
            return

        pd = poly.GetPointData()

        # build arrays per key with dynamic typing
        for key in sorted(keys):
            # peek to decide type/arity
            vals = [G.nodes[n].get(key, None) for n in nodes]

            def classify(v):
                if isinstance(v, (int, np.integer, bool)):
                    return ("num", 1)
                if isinstance(v, (float, np.floating)):
                    return ("num", 1)
                if isinstance(v, (list, tuple, np.ndarray)):
                    L = len(v)
                    if 1 <= L <= 4 and all(
                        isinstance(x, (int, float, np.integer, np.floating, bool))
                        for x in v
                    ):
                        return ("numvec", L)
                    return ("str", 1)
                if v is None:
                    return ("unknown", 1)
                return ("str", 1)

            # decide final type
            types = [classify(v)[0] for v in vals if v is not None]
            arities = [
                classify(v)[1]
                for v in vals
                if v is not None and classify(v)[0] != "str"
            ]
            if any(t == "str" for t in types):
                arr = _make_string_array(key, len(nodes))
                for i, v in enumerate(vals):
                    arr.SetValue(i, "" if v is None else str(v))
                pd.AddArray(arr)
                continue

            # numeric (scalar or vector)
            ncomp = max(arities) if arities else 1
            arr = _make_numeric_array(key, ncomp, len(nodes), dtype=float)
            # fill with NaNs
            for i in range(len(nodes)):
                for c in range(ncomp):
                    arr.SetComponent(i, c, np.nan)

            for i, v in enumerate(vals):
                if v is None:
                    continue
                if isinstance(v, (int, np.integer, bool, float, np.floating)):
                    arr.SetComponent(i, 0, float(v))
                else:
                    for c, x in enumerate(v[:ncomp]):
                        arr.SetComponent(i, c, float(x))
            pd.AddArray(arr)

    def _write_cell_attributes():
        # collect all edge attribute keys
        keys = set()
        for _, _, _, data in edges_list:
            keys.update(data.keys())
        if not keys:
            return

        cd = poly.GetCellData()

        for key in sorted(keys):
            vals = [data.get(key, None) for _, _, _, data in edges_list]

            def classify(v):
                if isinstance(v, (int, np.integer, bool)):
                    return ("num", 1)
                if isinstance(v, (float, np.floating)):
                    return ("num", 1)
                if isinstance(v, (list, tuple, np.ndarray)):
                    L = len(v)
                    if 1 <= L <= 4 and all(
                        isinstance(x, (int, float, np.integer, np.floating, bool))
                        for x in v
                    ):
                        return ("numvec", L)
                    return ("str", 1)
                if v is None:
                    return ("unknown", 1)
                return ("str", 1)

            types = [classify(v)[0] for v in vals if v is not None]
            arities = [
                classify(v)[1]
                for v in vals
                if v is not None and classify(v)[0] != "str"
            ]
            if any(t == "str" for t in types):
                arr = _make_string_array(key, len(vals))
                for i, v in enumerate(vals):
                    arr.SetValue(i, "" if v is None else str(v))
                cd.AddArray(arr)
                continue

            ncomp = max(arities) if arities else 1
            arr = _make_numeric_array(key, ncomp, len(vals), dtype=float)
            for i in range(len(vals)):
                for c in range(ncomp):
                    arr.SetComponent(i, c, np.nan)

            for i, v in enumerate(vals):
                if v is None:
                    continue
                if isinstance(v, (int, np.integer, bool, float, np.floating)):
                    arr.SetComponent(i, 0, float(v))
                else:
                    for c, x in enumerate(v[:ncomp]):
                        arr.SetComponent(i, c, float(x))
            cd.AddArray(arr)

    _write_point_attributes()
    _write_cell_attributes()

    # --- write .vtp
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(poly)
    if compress:
        writer.SetCompressorTypeToZLib()
        writer.SetDataModeToBinary()
    else:
        writer.SetDataModeToAscii()
    if writer.Write() == 0:
        raise RuntimeError("VTK writer failed")
    return out_path
