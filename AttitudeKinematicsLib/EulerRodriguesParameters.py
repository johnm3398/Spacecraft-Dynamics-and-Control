import numpy as np

from .DCM_utils import *

def EP_to_DCM(q, convention="scalar_first"):
    """
    Converts the EP/Quaternion to a direction cosine matrix (C).

     Args:
        q (np.array): A numpy array of size 4 (a row vector) representing the quaternion.
                      Depending on the convention:
                        - "scalar_first": [q0, q1, q2, q3], where q0 is the scalar part.
                        - "scalar_last": [q1, q2, q3, q0], where q0 is the scalar part.

        convention (str): Specifies the convention for quaternion representation.
                          Options ---> "scalar_first" (default) or "scalar_last"

    Returns:
        np.array: A 3x3 rotation matrix (C).
    """
    # Validate input vector
    validate_vec4(q)
    
    # Ensure q is a float array to maintain precision
    q = np.array(q, dtype=np.float64)
    
    # Check that the holonomic constraint of quaternion is satisfied, else normalize it
    q_norm = np.linalg.norm(q)
    if not np.isclose(q_norm, 1.0, atol=1e-8):
        q /= q_norm
    
    # Adjust indexing based on the specified convention
    if convention == "scalar_last":
        q1, q2, q3, q0 = q  # Swap positions to treat q0 as the last element
    elif convention == "scalar_first":
        q0, q1, q2, q3 = q  # Default behavior
    else:
        raise ValueError(f"Invalid convention '{convention}'. Choose 'scalar_first' or 'scalar_last'.")
    
    # Compute the elements of the C
    C = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1*q2 + q0*q3)          , 2 * (q1*q3 - q0*q2)          ],
        [2 * (q1*q2 - q0*q3)          , q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2*q3 + q0*q1)          ],
        [2 * (q1*q3 + q0*q2)          , 2 * (q2*q3 - q0*q1)          , q0**2 - q1**2 - q2**2 + q3**2]
    ])
    
    return C

def DCM_to_EP(C, convention="scalar_first"):
    """
    Converts a Direction Cosine Matrix (C) to a quaternion using Shepperd's method to ensure robustness against numerical issues.
    
    Args:
        C (np.array): A 3x3 rotation matrix (C).
        convention (str): Specifies the convention for quaternion representation.
                          Options ---> "scalar_first" (default) or "scalar_last"
    
    Returns:
        np.array: A quaternion represented as a numpy array of size 4.
                  Format depends on the `convention` parameter.
    """
    # Validate Input DCM
    validate_DCM(C)

    trace = np.trace(C)
    q_squared = np.zeros(4)
    q_squared[0] = (1.0 + trace) / 4.0
    q_squared[1] = (1.0 + 2 * C[0, 0] - trace) / 4.0
    q_squared[2] = (1.0 + 2 * C[1, 1] - trace) / 4.0
    q_squared[3] = (1.0 + 2 * C[2, 2] - trace) / 4.0

    q = np.zeros(4)
    max_index = np.argmax(q_squared)

    if max_index == 0:
        q[0] = np.sqrt(q_squared[0])
        q[1] = (C[1, 2] - C[2, 1]) / (4 * q[0])
        q[2] = (C[2, 0] - C[0, 2]) / (4 * q[0])
        q[3] = (C[0, 1] - C[1, 0]) / (4 * q[0])
    
    elif max_index == 1:
        q[1] = np.sqrt(q_squared[1])
        q[0] = (C[1, 2] - C[2, 1]) / (4 * q[1])
        if q[0] < 0:
            q[0] = -q[0]
            q[1] = -q[1]
        q[2] = (C[0, 1] + C[1, 0]) / (4 * q[1])
        q[3] = (C[2, 0] + C[0, 2]) / (4 * q[1])
        
    elif max_index == 2:
        q[2] = np.sqrt(q_squared[2])
        q[0] = (C[2, 0] - C[0, 2]) / (4 * q[2])
        if q[0] < 0:
            q[0] = -q[0]
            q[2] = -q[2]
        q[1] = (C[0, 1] + C[1, 0]) / (4 * q[2])
        q[3] = (C[1, 2] + C[2, 1]) / (4 * q[2])

    elif max_index == 3:
        q[3] = np.sqrt(q_squared[3])
        q[0] = (C[0, 1] - C[1, 0]) / (4 * q[3])
        if q[0] < 0:
            q[0] = -q[0]
            q[3] = -q[3]
        q[1] = (C[2, 0] + C[0, 2]) / (4 * q[3])
        q[2] = (C[1, 2] + C[2, 1]) / (4 * q[3])

    # Adjust output based on the specified convention
    if convention == "scalar_last":
        q = np.array([q[1], q[2], q[3], q[0]])
    elif convention == "scalar_first":
        q = np.array([q[0], q[1], q[2], q[3]])
    else:
        raise ValueError(f"Invalid convention '{convention}'. Choose 'scalar_first' or 'scalar_last'.")

    return q

def Bmat_EP(q, convention="scalar_first"):
    """
    Computes the 4x3 B matrix that maps body angular velocity (omega) to the derivative of the quaternion (Euler parameters) vector.

        dQ/dt = 1/2 * [B(Q)] * omega

    Args:
        q (array-like): A 4-element quaternion (Euler parameter) vector [q0, q1, q2, q3].
        convention (str): Specifies the convention for quaternion representation.
                          Options ---> "scalar_first" (default) or "scalar_last"
    
    Returns:
        np.ndarray: A 4x3 B matrix.
    
    Notes:
        - The quaternion vector q should be in the form [q0, q1, q2, q3], where q0 is the scalar component.
    """
    # Validate the input quaternion vector
    validate_vec4(q)

    # Convert input to a NumPy array if not already
    q = np.array(q, dtype=float)

    # Extract components of the quaternion
    if convention == "scalar_first":
        q0, q1, q2, q3 = q
    elif convention == "scalar_last":
        q1, q2, q3, q0 = q

    # Construct the B matrix using a structured array
    B = np.array([[-q1, -q2, -q3],
                  [ q0, -q3,  q2],
                  [ q3,  q0, -q1],
                  [-q2,  q1,  q0]])

    return B

def BInvmat_EP(q, convention="scalar_first"):
    """
    Computes the 3x4 B matrix that maps the derivative of the quaternion (Euler parameters) vector to the body angular velocity (omega).

        omega = 2 * [B(Q)]^(-1) * dQ/dt

    Args:
        q (array-like): A 4-element quaternion (Euler parameter) vector [q0, q1, q2, q3].
        convention (str): Specifies the convention for quaternion representation.
                          Options ---> "scalar_first" (default) or "scalar_last"

    Returns:
        np.ndarray: A 3x4 B matrix.
    
    Notes:
        - The quaternion vector q should be in the form [q0, q1, q2, q3], where q0 is the scalar component.
        - This matrix is used to map quaternion rates to body angular velocity.
    """
    # Validate the input quaternion vector
    validate_vec4(q)

    # Convert input to a NumPy array if not already
    q = np.array(q, dtype=float)

    # Extract components of the quaternion
    if convention == "scalar_first":
        q0, q1, q2, q3 = q
    elif convention == "scalar_last":
        q1, q2, q3, q0 = q

    # Construct the BInv matrix using a structured array
    B_inv = np.array([[-q1,  q0,  q3, -q2],
                      [-q2, -q3,  q0,  q1],
                      [-q3,  q2, -q1,  q0]])

    return B_inv

def normalize_quat(q):
    """
    Normalizes a quaternion to ensure it remains a unit quaternion.

    Args:
        q (array-like): A 4-element quaternion [q0, q1, q2, q3] or [q1, q2, q3, q0]

    Returns:
        np.ndarray: A normalized 4-element quaternion.

    Notes:
        - Ensures the quaternion maintains unit norm, which is critical for rotation representation.
        - Conventions do not matter for this function. The normalization can take place independent of conventions.
    """
    # Validate the input quaternion vector
    validate_vec4(q)

    # Convert input to a NumPy array if not already
    q = np.array(q, dtype=float)

    # Compute the norm of the quaternion
    norm_q = np.linalg.norm(q)

    # Avoid division by zero
    if norm_q == 0:
        raise ValueError("Quaternion norm is zero; cannot normalize.")

    # Normalize the quaternion
    q_normalized = q / norm_q

    return q_normalized

def quat_mult(q1, q2):
    """
    Computes the Hamilton product (quaternion multiplication) using the skew-symmetric matrix representation.

    Args:
        q1 (array-like): First quaternion [q0, q1, q2, q3].
        q2 (array-like): Second quaternion [q0, q1, q2, q3].
        convention (str): Specifies the quaternion representation.
                          Options ---> "scalar_first" (default) or "scalar_last".

    Returns:
        np.ndarray: The resulting quaternion after multiplication, maintaining the input convention.

    Notes:
        - Uses the skew-symmetric matrix formulation to perform quaternion multiplication.
        - Ensures the output follows the same convention as the input.
        - Quaternion multiplication represents **rotation composition**.
    """
    # Validate input quaternion vectors
    validate_vec4(q1)
    validate_vec4(q2)

    # Convert to NumPy arrays
    q1 = np.array(q1, dtype=float)
    q2 = np.array(q2, dtype=float)

    # Convert quaternion to skew-symmetric matrix form
    Q_mat = skew_symmetric(q1)

    # Perform quaternion multiplication using matrix-vector multiplication
    q_result = np.matmul(Q_mat, q2)

    return q_result

import numpy as np

def quat_inv(q):
    """
    Computes the inverse of a quaternion.

    Args:
        q (array-like): Quaternion [q0, q1, q2, q3] or [q1, q2, q3, q0].

    Returns:
        np.ndarray: The inverse quaternion.

    Notes:
        - Computes **q⁻¹ = normalize(q*)**, where `q*` is the conjugate.
        - Uses `normalize_quat` to ensure numerical stability.
        - The returned quaternion maintains the same convention as the input.
    """
    # Validate input quaternion vector
    validate_vec4(q)

    # Convert to NumPy array
    q = np.array(q, dtype=float)

    # Compute quaternion conjugate (negate vector part)
    q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])

    # Normalize the conjugate
    q_inv = normalize_quat(q_conjugate)

    return q_inv

def quat_diff(q1, q2):
    """
    Computes the relative quaternion (offset) between two quaternions.

    Args:
        q1 (array-like): First quaternion [q0, q1, q2, q3] or [q1, q2, q3, q0].
        q2 (array-like): Second quaternion [q0, q1, q2, q3] or [q1, q2, q3, q0].

    Returns:
        np.ndarray: The quaternion representing the relative orientation.

    Notes:
        - Computes the **rotation offset** needed to align `q1` to `q2`.
        - Uses the formula **q_diff = q2 ⊗ q1⁻¹**.
        - The output quaternion follows the **same convention** (scalar-first or scalar-last) as `q1`.
    """
    # Validate input quaternion vectors
    validate_vec4(q1)
    validate_vec4(q2)

    # Compute quaternion inverse
    q1_inv = quat_inv(q1)

    # Compute relative rotation (offset) quaternion using skew-symmetric multiplication
    q_diff = quat_mult(q2, q1_inv)

    return q_diff
