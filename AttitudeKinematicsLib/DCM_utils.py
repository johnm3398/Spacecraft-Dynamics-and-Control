import numpy as np

def validate_vec3(v):
    """
    Validates that the input vector has exactly 3 elements and that each element is numeric.

    Args:
        v (array-like): The input vector to check.

    Raises:
        ValueError: If the input vector does not have exactly 3 elements or if elements are not numeric.
        TypeError: If the input is not a list or a NumPy array.
    """
    # Check that v is a list or numpy array
    if not isinstance(v, (list, np.ndarray)):
        raise TypeError("Input vector must be a list or a NumPy array.")
    
    # Check that v has exactly 3 elements
    if len(v) != 3:
        raise ValueError("Input vector must have exactly 3 elements.")
    
    # Check that all elements in v are numbers (integers or floats)
    if not all(isinstance(element, (int, float)) for element in v):
        raise ValueError("All elements of the input vector must be numeric (int or float).")
    
def validate_vec4(v):
    """
    Validates that the input vector has exactly 4 elements and that each element is numeric.

    Args:
        v (array-like): The input vector to check.

    Raises:
        ValueError: If the input vector does not have exactly 4 elements or if elements are not numeric.
        TypeError: If the input is not a list or a NumPy array.
    """
    # Check that v is a list or numpy array
    if not isinstance(v, (list, np.ndarray)):
        raise TypeError("Input vector must be a list or a NumPy array.")
    
    # Check that v has exactly 3 elements
    if len(v) != 4:
        raise ValueError("Input vector must have exactly 3 elements.")
    
    # Check that all elements in v are numbers (integers or floats)
    if not all(isinstance(element, (int, float)) for element in v):
        raise ValueError("All elements of the input vector must be numeric (int or float).")

def skew_symmetric(v):
    """
    Returns the skew-symmetric matrix of a 3-element vector.

    Args:
        v (array-like): A 3-element array or list.

    Returns:
        np.ndarray: The 3x3 skew-symmetric matrix.
    """
    # Validate input vector
    validate_vec3(v)

    # Decompose vector into its components
    v1, v2, v3 = v

    # Construct the skew-symmetric matrix
    v_tilde_mat = np.array([[0  , -v3,  v2],
                            [v3 ,   0, -v1],
                            [-v2,  v1,   0]])

    return v_tilde_mat
