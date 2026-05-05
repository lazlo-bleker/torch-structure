import torch_structure as ts
import matplotlib.pyplot as plt

generator = ts.generators.DomeGenerator(seed=0)
data = generator()
data.plot(title="Randomized  Net", legend=False)

plt.show()
