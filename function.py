from scipy import sparse
import scipy
import numpy

def mapping(sparse_array, n_spins):    
    # Example 1D sparse array
    n = n_spins  # Length of binary representation

    # Step 1: Extract non-zero positions
    non_zero_positions = sparse_array.nonzero()[0]
    print(non_zero_positions)

    # Step 2: Convert positions to binary strings of length n
    binary_positions = [f"{pos:0{n}b}" for pos in non_zero_positions]

    # Step 3: Extract positions of '1's and compute new positions
    new_binary_positions = []

    for binary in binary_positions:
        ones_positions = [i for i, bit in enumerate(binary) if bit == '1']
        new_binary = ['0'] * (2 * n)

        for pos in ones_positions:
            new_binary[2 * pos] = '1'  # Place '1' at 2 * pos
            if 2 * pos + 1 < len(new_binary):  # Ensure within bounds
                new_binary[2 * pos + 1] = '1'  # Place '1' at 2 * pos + 1

        new_binary_positions.append(''.join(new_binary))

    # Step 4: Convert new binary numbers to integers
    new_positions = [int(binary, 2) for binary in new_binary_positions]

    # Step 5: Map original data values to new positions
    data = sparse_array.data  # Extract the non-zero values from the original sparse array
    print(data)

    # Step 6: Create new sparse array
    new_sparse_array = sparse.csr_array((data, (new_positions, [0] * len(new_positions))), shape=(2**(2*n), 1))

    # Print results
    print("Original Sparse Array:")
    print(sparse_array)

    print("\nNew Sparse Array:")
    print(new_sparse_array)

    return new_sparse_array