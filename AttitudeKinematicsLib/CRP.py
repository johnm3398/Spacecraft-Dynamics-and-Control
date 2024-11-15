import numpy as np

from .DCM_utils import *
from .EulerRodriguesParameters import DCM_to_EP

def CRP_to_DCM(q):
    """
    Converts a Classical Rodrigues Parameters (CRP) vector to a Direction Cosine Matrix (DCM).

    Args:
        q (np.array): A numpy array of size 3 representing the CRP vector (Gibbs vector).

    Returns:
        np.array: A 3x3 rotation matrix (DCM) corresponding to the rotation defined by the CRP vector.

    Notes:
        - The function assumes a passive rotation (coordinate transformation).
        - Ensure that q is a numpy array of floats for numerical precision.
    """
    # Validate input vector
    validate_vec3(q)
    
    # Ensure q is a numpy array of floats and reshape to make it a 3-element array
    q = np.array(q, dtype=np.float64).reshape(3)
    
    # Compute the skew-symmetric matrix q_cross (q^x)
    q_tilde = skew_symmetric(q)

    # Compute inner product: q^Tq (which is just squared magnitude of q)
    q_squared = np.dot(q, q)
    
    # Compute the outer product: qq^T
    q_outer = np.outer(q, q)
    
    # Identity matrix
    identity_matrix = np.eye(3)
    
    C = (1 / (1 + q_squared)) * ( ((1 - q_squared) * identity_matrix) + (2 * q_outer) - (2 * q_tilde) )
    
    return C

def DCM_to_CRP(dcm):
    """
    Converts a Direction Cosine Matrix (DCM) to the Classical Rodrigues Parameters (CRP) vector.

    Args:
        dcm (np.array): A 3x3 rotation matrix representing the DCM.

    Returns:
        np.array: A 3-element array representing the CRP vector.

    Notes:
        - The function first converts the DCM to a quaternion.
        - Then computes the CRP vector by dividing the vector part of the quaternion by its scalar part.
        - Assumes that the quaternion uses the scalar-first convention.
        - The function handles passive rotations (coordinate transformations).

    Raises:
        ZeroDivisionError: If the scalar part of the quaternion is zero (singularity at 180 degrees).

    Example:
        >>> dcm = np.eye(3)
        >>> q = DCM_to_CRP(dcm)
        >>> print(q)
        [0. 0. 0.]
    """
    # Validate Input DCM
    validate_DCM(C)

    # Convert DCM to quaternion
    b = DCM_to_EP(dcm)  # b should be a 4-element array [q0, q1, q2, q3]

    # Ensure b is a numpy array
    b = np.array(b, dtype=np.float64)

    # Extract scalar and vector parts
    q0 = b[0]      # Scalar part
    q_vec = b[1:]  # Vector part [q1, q2, q3]

    # Check for division by zero to avoid singularity at 180 degrees
    if np.isclose(q0, 0.0):
        raise ZeroDivisionError("The scalar part of the quaternion is zero; cannot compute CRP.")

    # Compute CRP vector by dividing the vector part by the scalar part
    crp = q_vec / q0

    return crp

def Bmat_CRP(q):
    """
    Computes the 3x3 B matrix that relates the body angular velocity vector (ω) to the derivative 
    of the Classical Rodrigues Parameters (CRP) vector (q̇).

    The B matrix is defined as:

        B(q) = I + [q]× + q qᵗ

    where:
        - I is the 3x3 identity matrix.
        - [q]× is the skew-symmetric matrix of the vector q.
        - q qᵗ is the outer product of q with itself.

    The relationship between the CRP rates and the body angular velocity is given by:

        q̇ = (1/2) * B(q) * ω

    Args:
        q (array-like): A 3-element Classical Rodrigues Parameters (CRP) vector.

    Returns:
        np.ndarray: A 3x3 B matrix.

    Notes:
        - The function validates that `q` is a 3-element numeric vector.
        - It uses the skew-symmetric matrix computation for cross-product representation.

    Raises:
        ValueError: If the input vector `q` is not a valid 3-element numeric vector.
    """
    # Validate the input vector
    validate_vec3(q)

    # Convert input to a NumPy array
    q = np.array(q, dtype=float)

    # Compute the skew-symmetric matrix of q
    q_tilde = skew_symmetric(q)

    # Compute the B matrix
    B = np.eye(3) + q_tilde + np.outer(q, q)

    return B

def BInvmat_CRP(q):
    """
    Computes the inverse B matrix (B_inv) that relates the derivative
    of the Classical Rodrigues Parameters (CRP) vector (q̇) to the body angular velocity vector (ω).

    The inverse B matrix is defined as:

        B_inv(q) = [I - [q]×] / (1 + qᵗ q)

    where:
        - I is the 3x3 identity matrix.
        - [q]× is the skew-symmetric matrix of the vector q.
        - qᵗ q is the dot product of q with itself (a scalar).

    The relationship between the body angular velocity and the CRP rates is given by:

        ω = 2 * B_inv(q) * q̇

    Args:
        q (array-like): A 3-element Classical Rodrigues Parameters (CRP) vector.

    Returns:
        np.ndarray: A 3x3 inverse B matrix.

    Notes:
        - The function validates that `q` is a 3-element numeric vector.
        - It uses the skew-symmetric matrix computation for cross-product representation.
        - The inverse B matrix is used to map the derivative of the CRP vector to the body angular velocity.

    Raises:
        ValueError: If the input vector `q` is not a valid 3-element numeric vector.
    """
    # Validate the input vector
    validate_vec3(q)

    # Convert input to a NumPy array
    q = np.array(q, dtype=float)

    # Compute the skew-symmetric matrix of q
    q_tilde = skew_symmetric(q)

    # Compute the denominator (1 + qᵗ q)
    denominator = 1 + np.dot(q, q)

    # Compute the inverse B matrix
    B_inv = (np.eye(3) - q_tilde) / denominator

    return B_inv