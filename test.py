import sys
import os

# Add the Model directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))

# Now you can import gjr_recursion
from gjr_recursion import gjr_recursion

# Example usage
import numpy as np

resids = np.random.randn(1000)
params = np.array([0.1, 0.85, 0.1, 0.1])
sigma = 1.0

result = gjr_recursion(resids, params, sigma)
print(result)