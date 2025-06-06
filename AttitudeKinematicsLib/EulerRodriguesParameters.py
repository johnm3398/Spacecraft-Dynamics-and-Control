import numpy as np
from scipy.integrate import solve_ivp


from .DCM_utils import *

def EP_to_DCM(q, convention="scalar_first", transformation_type="passive"):
    """
    Converts the EP/Quaternion to a direction cosine matrix (C).

     Args:
        q (np.array): A numpy array of size 4 (a row vector) representing the quaternion.
                      Depending on the convention:
                        - "scalar_first": [q0, q1, q2, q3], where q0 is the scalar part.
                        - "scalar_last": [q1, q2, q3, q0], where q0 is the scalar part.

        convention (str): Specifies the convention for quaternion representation.
                          Options: "scalar_first" (default) or "scalar_last".

        transformation_type (str): Specifies whether the DCM represents an 
                                   "active" or "passive" transformation.
                                   Options: "active" or "passive" (default).      

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
        q1, q2, q3, q0 = q 
    elif convention == "scalar_first":
        q0, q1, q2, q3 = q 
    else:
        raise ValueError(f"Invalid convention '{convention}'. Choose 'scalar_first' or 'scalar_last'.")
    
    # Compute the elements of the C
    C = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1*q2 + q0*q3)          , 2 * (q1*q3 - q0*q2)          ],
        [2 * (q1*q2 - q0*q3)          , q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2*q3 + q0*q1)          ],
        [2 * (q1*q3 + q0*q2)          , 2 * (q2*q3 - q0*q1)          , q0**2 - q1**2 - q2**2 + q3**2]
    ])

    # If an active transformation is requested, return C^T
    if transformation_type == "active":
        return C.T
    elif transformation_type == "passive":
        return C
    else:
        raise ValueError(f"Invalid transformation type '{transformation_type}'. Choose 'active' or 'passive'.")

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
        B = np.array([[-q1, -q2, -q3],
                      [ q0, -q3,  q2],
                      [ q3,  q0, -q1],
                      [-q2,  q1,  q0]])
        
    elif convention == "scalar_last":
        q1, q2, q3, q0 = q
        B = np.array([[ q0, -q3,  q2],
                      [ q3,  q0, -q1],
                      [-q2,  q1,  q0],
                      [-q1, -q2, -q3]])

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
        B_inv = np.array([[-q1,  q0,  q3, -q2],
                          [-q2, -q3,  q0,  q1],
                          [-q3,  q2, -q1,  q0]])
        
    elif convention == "scalar_last":
        q1, q2, q3, q0 = q
        B_inv = np.array([[ q0,  q3, -q2, -q1],
                          [-q3,  q0,  q1, -q2],
                          [ q2, -q1,  q0, -q3]])
    
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

def quat_mult(q1, q2, convention="scalar_first"):
    """
    Computes the Hamilton product (quaternion multiplication) with maximum precision 
    using direct computation rather than matrix multiplication. In this function, q1 is 
    treated as the initial quaternion and q2 as the final quaternion. The overall 
    composition is given by: q_result = q2 * q1, which means that the rotation corresponding 
    to q1 is applied first, then q2.
    
    Args:
        q1 (array-like): Initial quaternion [q0, q1, q2, q3] (scalar-first) or [q1, q2, q3, q0] (scalar-last).
        q2 (array-like): Final quaternion   [q0, q1, q2, q3] (scalar-first) or [q1, q2, q3, q0] (scalar-last).
        convention (str): Specifies the quaternion representation.
                          Options: "scalar_first" (default) or "scalar_last".
    
    Returns:
        np.ndarray: The resulting quaternion after multiplication (q2 * q1), maintaining 
                    the input convention.
    """
    # Validate input quaternion vectors
    validate_vec4(q1)
    validate_vec4(q2)

    # Convert to NumPy arrays
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)

    # q1 is already in scalar-first: [q0, q1, q2, q3]
    if convention == "scalar_first":
        pass
        
    # Convert from scalar-last [x, y, z, w] to scalar-first [w, x, y, z]
    elif convention == "scalar_last":
        q1 = np.array([q1[3], q1[0], q1[1], q1[2]])
        q2 = np.array([q2[3], q2[0], q2[1], q2[2]])
        
    else:
        raise ValueError("Convention must be 'scalar_first' or 'scalar_last'.")

    # Normalize quaternions
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)

    # Compute Hamilton product with q2 as the left operand and q1 as the right operand.
    # This corresponds to applying the initial rotation (q1) followed by the final rotation (q2).
    w = q2[0]*q1[0] - q2[1]*q1[1] - q2[2]*q1[2] - q2[3]*q1[3]
    x = q2[1]*q1[0] + q2[0]*q1[1] + q2[3]*q1[2] - q2[2]*q1[3]
    y = q2[2]*q1[0] - q2[3]*q1[1] + q2[0]*q1[2] + q2[1]*q1[3]
    z = q2[3]*q1[0] + q2[2]*q1[1] - q2[1]*q1[2] + q2[0]*q1[3]

    q_result = np.array([w, x, y, z], dtype=np.float64)

    # Convert back to scalar-last if needed
    if convention == "scalar_last":
        q_result = np.array([q_result[1], q_result[2], q_result[3], q_result[0]])

    return q_result

def quat_inv(q, convention="scalar_first"):
    """
    Computes the inverse (conjugate) of a quaternion.

    Args:
        q (array-like): Quaternion [q0, q1, q2, q3] (scalar-first) or [q1, q2, q3, q0] (scalar-last).
        convention (str): Specifies the quaternion representation.
                          Options -> "scalar_first" (default) or "scalar_last".

    Returns:
        np.ndarray: The inverse quaternion, maintaining the input convention.

    Notes:
        - If the quaternion is normalized, its inverse is simply the conjugate.
        - If the quaternion is not normalized, this function ensures proper inversion:
          q_inv = q_conjugate / (||q||^2).
        - The returned quaternion will follow the same convention as the input.
    """
    # Validate input quaternion vector
    validate_vec4(q)

    # Convert to NumPy array
    q = np.array(q, dtype=np.float64)

    # Compute the quaternion norm (do not use normalize_quat here)
    norm_q2 = np.dot(q, q)
    if norm_q2 < np.finfo(float).eps:
        raise ValueError("Cannot invert a near-zero quaternion due to precision limitations.")

    # Extract components based on convention and compute conjugate
    if convention == "scalar_first":
        q0, q1, q2, q3 = q
        q_conjugate = np.array([q0, -q1, -q2, -q3], dtype=np.float64)
    elif convention == "scalar_last":
        q1, q2, q3, q0 = q
        q_conjugate = np.array([-q1, -q2, -q3, q0], dtype=np.float64)
    else:
        raise ValueError("Convention must be 'scalar_first' or 'scalar_last'.")

    # If the quaternion is already normalized, avoid unnecessary division
    if np.abs(norm_q2 - 1.0) < np.finfo(float).eps:
        return q_conjugate  # Already a unit quaternion, no need to divide

    # Compute proper inverse: q_inv = q_conjugate / (norm_q^2)
    q_inv = q_conjugate / norm_q2

    return q_inv

def quat_diff(q1, q2, convention="scalar_first"):
    """
    Computes the relative quaternion that transforms q2 into q1.

    This is achieved by computing q2_inv * q1, which represents the rotation 
    from q2's frame to q1's frame. The inverse of q2 is computed implicitly 
    (assuming unit quaternions) by negating the vector part of q2.

    Args:
        q1 (array-like): Reference quaternion in either 
                         scalar-first [q0, q1, q2, q3] or 
                         scalar-last [q1, q2, q3, q0] format.
        q2 (array-like): Target quaternion in the same format as q1.
        convention (str): Defines input/output format.
                          Options: "scalar_first" (default) or "scalar_last".

    Returns:
        np.ndarray: The relative quaternion, preserving the input convention.
    """
    # Validate input quaternions
    validate_vec4(q1)
    validate_vec4(q2)

    # Convert to NumPy arrays
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)

    # Handle scalar-first vs scalar-last conventions
    if convention == "scalar_first":
        pass
    elif convention == "scalar_last":
        q1 = np.array([q1[3], q1[0], q1[1], q1[2]])
        q2 = np.array([q2[3], q2[0], q2[1], q2[2]])
    else:
        raise ValueError("Convention must be 'scalar_first' or 'scalar_last'.")

    # Normalize quaternions
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)

    # Compute the relative quaternion q_diff = q2_inv * q1 without explicit inversion.
    # For unit quaternions, the inverse of q2 is given by [q2[0], -q2[1], -q2[2], -q2[3]].
    # The Hamilton product is computed as follows:
    w =  q2[0] * q1[0] + q2[1] * q1[1] + q2[2] * q1[2] + q2[3] * q1[3]
    x =  q2[0] * q1[1] - q2[1] * q1[0] - q2[2] * q1[3] + q2[3] * q1[2]
    y =  q2[0] * q1[2] + q2[1] * q1[3] - q2[2] * q1[0] - q2[3] * q1[1]
    z =  q2[0] * q1[3] - q2[1] * q1[2] + q2[2] * q1[1] - q2[3] * q1[0]

    q_diff = np.array([w, x, y, z], dtype=np.float64)

    # Convert back to scalar-last if needed
    if convention == "scalar_last":
        q_diff = np.array([q_diff[1], q_diff[2], q_diff[3], q_diff[0]])

    return q_diff

def quat_derivative(q, omega_vec, convention="scalar_first"):
    """
    Computes the time derivative of a quaternion given body angular velocity.
        
        dQ/dt = 1/2 * [B(Q)] * omega

    Args:
        q (array-like): A 4-element quaternion [q0, q1, q2, q3].
        omega_vec (array-like): Angular velocity [omega_1, omega_2, omega_3] in body frame.
        convention (str): Specifies the quaternion representation.
                          Options ---> "scalar_first" (default) or "scalar_last".

    Returns:
        np.ndarray: The quaternion derivative dq/dt.

    Notes:
        - Uses the kinematic equation **dq/dt = 1/2 * B(q) * omega**.
        - Ensures quaternion remains normalized after integration.
    """
    # Validate quaternion and angular velocity vector
    validate_vec4(q)
    validate_vec3(omega_vec)

    # Convert to NumPy arrays
    q = np.array(q, dtype=float)
    omega_vec = np.array(omega_vec, dtype=float)

    # Compute B-matrix for quaternion kinematics
    B_mat = Bmat_EP(q, convention)

    # Compute quaternion derivative (kinematic equation)
    q_dot = 0.5 * np.matmul(B_mat, omega_vec)

    return q_dot

def integrate_quaternion(quat_init, omega_vec, delta_t, convention="scalar_first"):
    """
    Integrates the quaternion kinematics equation using solve_ivp.

    Args:
        quat_init (array-like): Initial quaternion [q0, q1, q2, q3].
        omega_vec (array-like): Angular velocity vector [ω1, ω2, ω3].
        delta_t (float): Time step for integration.
        convention (str): Specifies quaternion format ("scalar_first" or "scalar_last").

    Returns:
        np.ndarray: Updated quaternion after integration.
    """
    quat_init = np.array(quat_init, dtype=float)
    
    # Solve IVP to integrate quaternion kinematics
    sol = solve_ivp(fun=lambda t, q: quat_derivative(q, omega_vec, convention),
                    t_span=[0, delta_t], 
                    y0=quat_init, 
                    method='DOP853', 
                    rtol=1e-12, 
                    atol=1e-14)
    
    # Extract and normalize the final quaternion
    quat_final = sol.y[:, -1]
    quat_final = normalize_quat(quat_final)

    return quat_final
