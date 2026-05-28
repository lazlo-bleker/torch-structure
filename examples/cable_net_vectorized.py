"""
Example: Form-find a Cable Net with the Force Density Method (FDM)
"""

import torch_structure as ts
import matplotlib.pyplot as plt

# --------------------------------
# 1. Create inital cable net setup
# --------------------------------
input_params = {
    "n": 5,  # Number of corner supports
}

# ----------------------------------
# 2. Generate the cable net geometry
# ----------------------------------
cable_net_generator = ts.generators.CableNetGeneratorVectorized(**input_params)
data = cable_net_generator()  # Main data object of TorchStructure we'll work with

# -------------------------------
# 3. Plot the resulting structure
# -------------------------------
data.plot(title="Randomized Cable Net", legend=False)

plt.show()

# -------------------------------
# 4. Verify that the force density method found an equilibrium
# -------------------------------
print("Equilibrium: ", data.verify_equilibrium())


