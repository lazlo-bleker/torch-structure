from ..data.data import StructData

class DesignVariableGroup():
    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None
        self.n_values = None
        self.x_index_start = None
        self.attr_name = None
        self.attr_mask = None


    def apply(self, x):
        raise NotImplementedError


class DesignVariableHandler():
    def __init__(self):
        self.struc_data = None
        self.dv_dict = None
        self.solver_name = None
        self.solver_kwargs = None

    def solve_graph(self, x):
        raise NotImplementedError
