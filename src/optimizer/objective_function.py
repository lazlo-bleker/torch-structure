import torch

class Function_Object():
    def __init__(self, function, kwargs={}):
        """
        Initializes a funciton with optional state variables
        """
        # NOTE: Having a function as an object allows to cache variables (a function can have a sate, store in kwargs)
        self.function = function
        self.kwargs = kwargs
        self.log = None
    
    def __call__(self, graph):
        """
        Calls a function that acts on a solved graph resulting from StrucData.cem and optional stored variables
        """
        out, log = self.function(graph, **self.kwargs)
        self.log = log
        return out

class Objective_Function_Handler():
    def __init__(self, solve_graph):
        """
        Creates an object that evaluates the objective function used in the optimization, which provides gradients from pytorch
        """
        # Link method to apply the value of the design variables and solve the system
        self.solve_graph = solve_graph
        # Init partial loss weights (compromise vector)
        self.weights = {}
        # Init collection of Function_Objects
        self.function_objects = {}
        # Init attribute to store loss values (for data logging)
        self.loss_dict = {}
    
    def add_function(self, name : str, weight : float, function, kwargs={}):
        """
        Adds a function to be integrated into the objective function
        """
        # Determines the importance of the objective function
        self.weights[name] = weight
        # Add Function object to attribute dictionary
        self.function_objects[name] = Function_Object(function, kwargs)
    
    def forward(self, x):
        """
        Evaluates the objective function
        """
        # Apply design variables & solve CEM
        graph_solved = self.solve_graph(x)
        # Compute total loss based on CEM solution
        total_loss = torch.tensor(0.0)
        for name, obj_function in self.function_objects.items():
            # Add contribution of obj function from list
            partial_loss = obj_function(graph_solved)
            self.loss_dict[name] = partial_loss.detach().item()
            total_loss += self.weights[name] * partial_loss
        # Save total loss too
        self.loss_dict["Loss/Total"] = total_loss
        return total_loss
    
    def func_grad_value_scipy(self, x_np):
        """
        Combines forward method with an automatically generated backward method
        """
        # Convert numpy to torch with gradients being required
        x = torch.tensor(x_np, dtype=torch.float64)
        # Generate grad and value modes from forward
        func_grad_value = torch.func.grad_and_value(self.forward)
        # Evaluate grad and value together
        grad, loss = func_grad_value(x)
        # Cast to numpy to use in scipy
        return loss.item(), grad.detach().numpy()
    
    def log(self):
        log = {}
        for name, obj_function in self.function_objects.items():
            log[name] = obj_function.log
        return log