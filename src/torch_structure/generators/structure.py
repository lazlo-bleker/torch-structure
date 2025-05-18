# Todo: Update outdated data structure

import numpy as np


class Structure:
    red_color = "#E40714"
    blue_color = "#0578BF"

    def __init__(self, nodes=None, edges=None):
        if nodes is None:
            nodes = {}
        if edges is None:
            edges = {}
        self.nodes = nodes
        self.edges = edges

    def _set_dict_attr(self, dict):
        for key, random_value in dict.items():
            value = self.kwargs.get(key, random_value)
            dict[key] = value
            setattr(self, key, value)

    def _set_node(
        self,
        node_idx,
        coords,
        load,
        sem_type,
        external_force=None,
        support=False,
        origin=False,
    ):
        """Save node data in global node dictionary"""
        node_dict = {
            "coordinates": coords,
            "load": load,
            "semantic_type": sem_type,
            "support": support,
            "origin": origin,
        }
        if external_force is not None:
            node_dict["external_force"] = external_force
        self.nodes[node_idx] = node_dict

    def _set_edge(self, edge, force, cem_type, sem_type):
        """Save edge data in global edge dictionary"""
        start_coords = self.nodes[edge[0]]["coordinates"]
        end_coords = self.nodes[edge[1]]["coordinates"]
        length = np.abs(np.linalg.norm(end_coords - start_coords))
        edge_dict = {
            "force": force,
            "length": length,
            "cem_type": cem_type,
            "semantic_type": sem_type,
        }
        self.edges[edge] = edge_dict

    def _are_connected(self, edge1, edge2):
        connected = edge1[0] in edge2 or edge1[1] in edge2
        return connected

    def _force_direction(self, start_node_idx, end_node_idx):
        """Compute the normed direction vector from start node to end node"""
        direction = (
            self.nodes[end_node_idx]["coordinates"]
            - self.nodes[start_node_idx]["coordinates"]
        )
        if np.linalg.norm(direction) != 0:
            direction = direction / np.linalg.norm(direction)
        return direction

    def _resultant(self, node_idx, load_factor=1.0, ignore_sem_type=None):
        """Compute the resultant of a node"""
        # Initialize zero resultant
        resultant = np.zeros(3)

        # Add load and external force
        resultant += load_factor * self.nodes[node_idx]["load"]
        if "external_force" in self.nodes[node_idx]:
            resultant += self.nodes[node_idx]["external_force"]

        # Add force vector for all connected edges
        for edge, attributes in self.edges.items():
            if node_idx in edge:
                if (
                    ignore_sem_type is not None
                    and self.edges[edge]["semantic_type"] == ignore_sem_type
                ):
                    continue
                other_node_idx = edge[1] if edge[0] == node_idx else edge[0]
                direction = self._force_direction(node_idx, other_node_idx)
                force_vector = direction * attributes["force"]
                resultant += force_vector
        return resultant

    def _verify_equilibrium(self, threshold=1e-3, verbose=False):
        """Verify global and local equilibrium"""
        equilibrium = True
        total_external_force = np.zeros(3)
        for node_idx in self.nodes:
            # Verify global equilibrium
            total_external_force += self.nodes[node_idx]["load"]
            if "external_force" in self.nodes[node_idx]:
                total_external_force += self.nodes[node_idx]["external_force"]

            # Verify local equilibrium
            resultant = self._resultant(node_idx)
            if np.linalg.norm(resultant) > threshold:
                equilibrium = False
                if verbose:
                    print(f"no node equilibrium in node {node_idx} ({resultant})")

        if np.linalg.norm(total_external_force) > threshold:
            equilibrium = False
            if verbose:
                print(f"no global equilibrium ({total_external_force}")
        return equilibrium

    def load_path(self):
        """Compute the total load path of the structure"""
        total_load_path = 0
        for edge, attributes in self.edges.items():
            load_path = attributes["length"] * np.abs(attributes["force"])
            total_load_path += load_path
            self.edges[edge]["load_path"] = load_path
        return total_load_path

    def average_member_length(self):
        """Compute the average member length of the structure"""
        total_member_length = 0
        for edge, attributes in self.edges.items():
            total_member_length += attributes["length"]
        average_member_length = total_member_length / len(self.edges)
        return average_member_length
