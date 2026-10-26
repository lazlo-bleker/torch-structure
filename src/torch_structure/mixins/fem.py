from torch_structure.analysis import frame_analysis
from torch_structure.mixins.utils import OverrideResolveMixin

class FEMMixin(OverrideResolveMixin):
    """Adds linear-elastic frame and truss analysis to a data class."""

    def fem(
        self,
        inplace: bool=False,
        coords=None,
        load=None,
        moment=None,
        support_dofs=None,
        edge_index=None,
        E=None,
        A=None,
        Iy=None,
        Iz=None,
        J=None,
        G=None,
        roll=None,
        release=None,
        displacement_out=None,
        rotation_out=None,
        force_out=None,
        end_forces_out=None,
        reaction_out=None,
    ):
        """
        Applies a linear-elastic frame analysis to this structure and stores `displacement`,
        `rotation`, `force`, `end_forces` and `reaction`.

        Supports are read from `support_dofs`: translations from `support_xyz`, or from
        `is_support` as pins, and rotations from `support_rxyz`, free by default.
        Edge attributes are read from the canonical (`directed_mask`) edges, so `release` refers
        to the ends of each edge in its canonical direction. `Iy`, `Iz`, `J`, `G`, `roll`,
        `release` and `moment` are optional; members without bending properties act as truss
        bars. Coordinates are not updated; the deformed geometry is `coords + displacement`.

        See [`torch_structure.analysis.frame_analysis`](../analysis/frame.md).
        """
        batch = getattr(self, "batch", None)

        # Node inputs
        coords = self._resolve_override("coords", coords)
        load = self._resolve_override("load", load)
        moment = self._resolve_override("moment", moment, required=False)
        support_dofs = self._resolve_override("support_dofs", support_dofs)

        # Edge inputs
        edge_mask = self.directed_mask.view(-1)
        edge_index = self._resolve_override("directed_edge_index", edge_index)
        E = self._resolve_override("E", E, mask=edge_mask)
        A = self._resolve_override("A", A, mask=edge_mask)
        Iy = self._resolve_override("Iy", Iy, mask=edge_mask, required=False)
        Iz = self._resolve_override("Iz", Iz, mask=edge_mask, required=False)
        J = self._resolve_override("J", J, mask=edge_mask, required=False)
        G = self._resolve_override("G", G, mask=edge_mask, required=False)
        roll = self._resolve_override("roll", roll, mask=edge_mask, required=False)
        release = self._resolve_override("release", release, mask=edge_mask, required=False)

        # Output keys
        displacement_key = "displacement" if displacement_out is None else displacement_out
        rotation_key = "rotation" if rotation_out is None else rotation_out
        force_key = "force" if force_out is None else force_out
        end_forces_key = "end_forces" if end_forces_out is None else end_forces_out
        reaction_key = "reaction" if reaction_out is None else reaction_out

        # FEM computation
        result = frame_analysis(
            coords,
            edge_index,
            load,
            support_dofs,
            E,
            A,
            Iy=Iy,
            Iz=Iz,
            J=J,
            G=G,
            roll=roll,
            release=release,
            moment=moment,
            batch=batch,
        )

        # Update data object
        new_data = self if inplace else self.clone()
        new_data[displacement_key] = result.displacement
        new_data[rotation_key] = result.rotation
        new_data[force_key] = self.edge_attr_to_undirected(result.force, edge_mask)
        new_data[end_forces_key] = self.edge_attr_to_undirected(result.end_forces, edge_mask)
        new_data[reaction_key] = result.reaction

        if not inplace:
            return new_data
