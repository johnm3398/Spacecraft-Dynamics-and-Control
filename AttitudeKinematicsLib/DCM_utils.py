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
    
    # Check that v has exactly 4 elements
    if len(v) != 4:
        raise ValueError("Input vector must have exactly 4 elements.")
    
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

def validate_DCM(dcm):
    """
    Validates if the given matrix is a proper Direction Cosine Matrix (DCM).

    A valid DCM must satisfy the following properties:
        1. It must be a 3x3 matrix.
        2. Its determinant must be +1 (proper rotation matrix).
        3. Each row and column must have a unit norm (orthonormality).
        4. The product of the DCM with its transpose must yield the identity matrix.

    Args:
        dcm (np.ndarray): A 3x3 matrix representing a DCM.

    Raises:
        ValueError: If the DCM fails any of the validation checks.
    """
    # Ensure input is a NumPy array
    dcm = np.array(dcm, dtype=float)

    # Check if the DCM is 3x3
    if dcm.shape != (3, 3):
        raise ValueError("DCM must be a 3x3 matrix.")

    # Check if determinant is close to +1
    det = np.linalg.det(dcm)
    if not np.isclose(det, 1.0, atol=1e-8):
        raise ValueError(f"Determinant of DCM must be +1. Found: {det}")

    # Check if rows and columns are unit vectors (orthonormality)
    for i in range(3):
        row_norm = np.linalg.norm(dcm[i, :])
        col_norm = np.linalg.norm(dcm[:, i])
        if not (np.isclose(row_norm, 1.0, atol=1e-8) and np.isclose(col_norm, 1.0, atol=1e-8)):
            raise ValueError(f"Row or column {i+1} of the DCM is not a unit vector.")

    # Check if the DCM multiplied by its transpose yields the identity matrix
    identity_check = np.dot(dcm, dcm.T)
    if not np.allclose(identity_check, np.eye(3), atol=1e-8):
        raise ValueError("The product of DCM and its transpose is not the identity matrix.")

    # Check if the DCM preserves handedness (right-handed coordinate system)
    handedness_check = np.dot(dcm[0], np.cross(dcm[1], dcm[2]))
    if not np.isclose(handedness_check, 1.0, atol=1e-8):
        raise ValueError("The DCM does not preserve handedness (right-handed system check failed).")

    #print("The DCM is valid.")