import numpy as np

def nonzero_values(array):
    non_zero_indices = np.nonzero(array)[0]  # Get the indices of non-zero values
    non_zero_values = array[non_zero_indices]  # Get the non-zero values
    print("Positions of non-zero values:", non_zero_indices)
    print("Non-zero values of array:", non_zero_values)