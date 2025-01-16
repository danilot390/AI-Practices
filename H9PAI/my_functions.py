import numpy as np
def expensive_function(data):
    return sum(data)

# Function to process a chunk of data.
def process_chunk(chunk):
    # Use NumPy for fast computation
    return np.sum(np.exp(chunk) * np.log1p(chunk))