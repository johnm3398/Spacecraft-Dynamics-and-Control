import numpy as np

from .DCM_utils import *

def EP_to_DCM(q):
    """
    Converts the EP/Quaternion to a direction cosine matrix (C).

    Args:
        q (np.array): A numpy array of size 4 (a row vector) representing the quaternion,
                      where q[0] is the scalar part (beta_0), and q[1], q[2], q[3] are the 
                      vector parts (beta_1, beta_2, beta_3).

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
    
    # Extract components
    q0, q1, q2, q3 = q
    
    # Compute the elements of the C
    C = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1*q2 + q0*q3)          , 2 * (q1*q3 - q0*q2)          ],
        [2 * (q1*q2 - q0*q3)          , q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2*q3 + q0*q1)          ],
        [2 * (q1*q3 + q0*q2)          , 2 * (q2*q3 - q0*q1)          , q0**2 - q1**2 - q2**2 + q3**2]
    ])
    
    return C

def DCM_to_EP(C):
    """
    Converts a Direction Cosine Matrix (C) to a quaternion using Shepperd's method to ensure robustness against numerical issues.
    
    Args:
        C (np.array): A 3x3 rotation matrix (C).
    
    Returns:
        np.array: A quaternion represented as a numpy array of size 4, with the scalar component as the first element.
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
    
    return q

def Bmat_EP(q):
    """
    Computes the 4x3 B matrix that maps body angular velocity (omega) to the derivative of the quaternion (Euler parameters) vector.

        dQ/dt = 1/2 * [B(Q)] * omega

    Args:
        q (array-like): A 4-element quaternion (Euler parameter) vector [q0, q1, q2, q3].
    
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
    q0, q1, q2, q3 = q

    # Construct the B matrix using a structured array
    B = np.array([[-q1, -q2, -q3],
                  [ q0, -q3,  q2],
                  [ q3,  q0, -q1],
                  [-q2,  q1,  q0]])

    return B

def BInvmat_EP(q):
    """
    Computes the 3x4 B matrix that maps the derivative of the quaternion (Euler parameters) vector to the body angular velocity (omega).

        omega = 2 * [B(Q)]^(-1) * dQ/dt

    Args:
        q (array-like): A 4-element quaternion (Euler parameter) vector [q0, q1, q2, q3].

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
    q0, q1, q2, q3 = q

    # Construct the BInv matrix using a structured array
    B_inv = np.array([[-q1,  q0,  q3, -q2],
                      [-q2, -q3,  q0,  q1],
                      [-q3,  q2, -q1,  q0]])

    return B_inv