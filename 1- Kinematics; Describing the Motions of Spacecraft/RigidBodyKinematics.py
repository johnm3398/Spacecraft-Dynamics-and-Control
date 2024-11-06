import numpy as np

'''===================================================================== Helper Functions ====================================================================='''
def skew_symmetric(vector):
    """
    Returns the skew-symmetric matrix of a 3-element vector.
    """
    x, y, z = vector
    return np.array([[ 0, -z,  y],
                     [ z,  0, -x],
                     [-y,  x,  0]])

'''===================================================================== Rotation Matrices for Euler Angles ====================================================================='''
def rotation_matrix_x(phi, transformation_type='passive'):
    """Generate rotation matrix for a roll (rotation about the x-axis).
    
    Args:
        phi (float): The angle of rotation in degrees.
        transformation_type (str): Specifies the type of transformation, 'passive' (default) or 'active'.
    
    Returns:
        numpy.ndarray: The rotation matrix for x-axis rotation.
    """
    phi = np.radians(phi)
    c, s = np.cos(phi), np.sin(phi)
    matrix = np.array([[1, 0, 0], 
                       [0, c, s], 
                       [0, -s, c]])
    if transformation_type == 'active':
        return matrix.T
    return matrix

def rotation_matrix_y(theta, transformation_type='passive'):
    """Generate rotation matrix for a pitch (rotation about the y-axis).
    
    Args:
        theta (float): The angle of rotation in degrees.
        transformation_type (str): Specifies the type of transformation, 'passive' (default) or 'active'.
    
    Returns:
        numpy.ndarray: The rotation matrix for y-axis rotation.
    """
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    matrix = np.array([[c, 0, -s], 
                       [0, 1, 0], 
                       [s, 0, c]])
    if transformation_type == 'active':
        return matrix.T
    return matrix

def rotation_matrix_z(psi, transformation_type='passive'):
    """Generate rotation matrix for a yaw (rotation about the z-axis).
    
    Args:
        psi (float): The angle of rotation in degrees.
        transformation_type (str): Specifies the type of transformation, 'passive' (default) or 'active'.
    
    Returns:
        numpy.ndarray: The rotation matrix for z-axis rotation.
    """
    psi = np.radians(psi)
    c, s = np.cos(psi), np.sin(psi)
    matrix = np.array([[c, s, 0], 
                       [-s, c, 0], 
                       [0, 0, 1]])
    if transformation_type == 'active':
        return matrix.T
    return matrix

'''===================================================================== PRV Codes ====================================================================='''
def prv_to_rotation_matrix(e, phi_deg):
    """
    Converts a Principal Rotation Vector (PRV) to a rotation matrix.

    Args:
        e (np.array)   : The unit vector of the PRV.
        phi_deg (float): The rotation angle of the PRV in degrees.

    Returns:
        np.array: A 3x3 rotation matrix.
    """
    # Convert the angle from degrees to radians
    phi_rad = np.radians(phi_deg)
    
    # Calculate the cosine and sine of the angle
    c_phi = np.cos(phi_rad)
    s_phi = np.sin(phi_rad)
    
    # Calculate the matrix Sigma
    Sigma = 1 - c_phi

    # Ensure e is a float array to avoid UFuncTypeError during in-place operations
    e = np.array(e, dtype=float)
    
    # Normalize e vector to ensure it's a valid unit vector
    e /= np.linalg.norm(e)
    
    # Decompose the unit vector into its components
    e1, e2, e3 = e
    
    # Construct the rotation matrix using the given formula
    C = np.array([[((e1**2)*Sigma + c_phi), (e1*e2*Sigma + e3*s_phi), (e1*e3*Sigma - e2*s_phi)],
                  [(e2*e1*Sigma - e3*s_phi), ((e2**2)*Sigma + c_phi), (e2*e3*Sigma + e1*s_phi)],
                  [(e3*e1*Sigma + e2*s_phi), (e3*e2*Sigma - e1*s_phi), ((e3**2)*Sigma + c_phi)]])

    return C

def rotation_matrix_to_prv(C):
    """
    Converts a rotation matrix to a Principal Rotation Vector (PRV).

    Args:
        C (np.array): A 3x3 rotation matrix.

    Returns:
        tuple: A PRV represented as (e_vector, phi_angle).
    """
    # Compute the angle phi from the trace of the rotation matrix
    trace_C = np.trace(C)
    phi = np.arccos((trace_C - 1) / 2)
    
    # Handle edge cases where phi is 0 or π
    if np.isclose(phi, 0) or np.isclose(phi, np.pi):
        
        # For phi=0, no rotation, the axis can be arbitrary, choose x-axis for simplicity
        # For phi=π, rotation by 180 degrees, find axis by identifying non-zero component
        e = np.array([1, 0, 0])  # Arbitrary axis, could also check for non-diagonal elements
    
    else:
        # Compute the unit vector e from the off-diagonal elements of the matrix C
        e = (1 / (2 * np.sin(phi))) * np.array([C[1, 2] - C[2, 1],
                                                C[2, 0] - C[0, 2],
                                                C[0, 1] - C[1, 0]])
        # Normalize the unit vector to ensure it's a valid unit vector
        e /= np.linalg.norm(e)

    # Ensure the angle phi is in the range [0, 2*pi)
    phi = np.mod(phi, 2 * np.pi)

    return e, phi

'''===================================================================== Euler-Rodrigues Parameters Codes ====================================================================='''
def quaternion_to_DCM(q):
    """
    Converts a quaternion to a direction cosine matrix (DCM).

    Args:
        q (np.array): A numpy array of size 4 (a row vector) representing the quaternion,
                      where q[0] is the scalar part (beta_0), and q[1], q[2], q[3] are the 
                      vector parts (beta_1, beta_2, beta_3).

    Returns:
        np.array: A 3x3 rotation matrix (DCM).

    Example:
        >>> q = np.array([1, 5, 6, 2])
        >>> quaternion_to_DCM(q)
        array([[ 0.38461538, -0.07692308,  0.91923077],
               [ 0.07692308,  0.99230769, -0.09615385],
               [-0.91923077,  0.09615385,  0.38461538]])
    """
    # Ensure q is a float array to maintain precision
    q = np.array(q, dtype=np.float64)
    
    # Check that the holonomic constraint of quaternion is satisfied, else normalize it
    q_norm = np.linalg.norm(q)
    if not np.isclose(q_norm, 1.0, atol=1e-8):
        q /= q_norm
    
    # Extract components
    q0, q1, q2, q3 = q
    
    # Compute the elements of the DCM
    C = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1*q2 + q0*q3),         2 * (q1*q3 - q0*q2)],
        [2 * (q1*q2 - q0*q3),           q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2*q3 + q0*q1)],
        [2 * (q1*q3 + q0*q2),           2 * (q2*q3 - q0*q1),           q0**2 - q1**2 - q2**2 + q3**2]
    ])
    
    return C

def DCM_to_quaternion(dcm):
    """
    Converts a Direction Cosine Matrix (DCM) to a quaternion using a method to ensure robustness against numerical issues.
    
    Args:
        dcm (np.array): A 3x3 rotation matrix (DCM).
    
    Returns:
        np.array: A quaternion represented as a numpy array of size 4, with the scalar component as the first element.
    
    Example Usage:
        >>> dcm = np.array([[-0.21212121, 0.96969697, 0.12121212],
                            [0.84848485, 0.12121212, 0.51515152],
                            [0.48484848, 0.21212121, -0.84848485]])
        >>> quaternion = DCM_to_quaternion(dcm)
        >>> print(quaternion)
    
    Notes:
    - If the scalar component (q0) is negative, it is flipped to positive. (Ensuring shortes path of rotation)
    - Corresponding vector component is also flipped when q0 is flipped. (Shepperd's Method)
    - Flipping maintains the quaternion's correct rotational encoding.
    - Ensures the quaternion represents a rotation of less than 180 degrees.
    - Adheres to quaternion algebra for accurate 3D rotation representation.
    """
    trace = np.trace(dcm)
    q_squared = np.zeros(4)
    q_squared[0] = (1.0 + trace) / 4.0
    q_squared[1] = (1.0 + 2 * dcm[0, 0] - trace) / 4.0
    q_squared[2] = (1.0 + 2 * dcm[1, 1] - trace) / 4.0
    q_squared[3] = (1.0 + 2 * dcm[2, 2] - trace) / 4.0

    q = np.zeros(4)
    max_index = np.argmax(q_squared)

    if max_index == 0:
        q[0] = np.sqrt(q_squared[0])
        q[1] = (dcm[1, 2] - dcm[2, 1]) / (4 * q[0])
        q[2] = (dcm[2, 0] - dcm[0, 2]) / (4 * q[0])
        q[3] = (dcm[0, 1] - dcm[1, 0]) / (4 * q[0])
    
    elif max_index == 1:
        q[1] = np.sqrt(q_squared[1])
        q[0] = (dcm[1, 2] - dcm[2, 1]) / (4 * q[1])
        if q[0] < 0:
            q[0] = -q[0]
            q[1] = -q[1]
        q[2] = (dcm[0, 1] + dcm[1, 0]) / (4 * q[1])
        q[3] = (dcm[2, 0] + dcm[0, 2]) / (4 * q[1])
        
    elif max_index == 2:
        q[2] = np.sqrt(q_squared[2])
        q[0] = (dcm[2, 0] - dcm[0, 2]) / (4 * q[2])
        if q[0] < 0:
            q[0] = -q[0]
            q[2] = -q[2]
        q[1] = (dcm[0, 1] + dcm[1, 0]) / (4 * q[2])
        q[3] = (dcm[1, 2] + dcm[2, 1]) / (4 * q[2])

    elif max_index == 3:
        q[3] = np.sqrt(q_squared[3])
        q[0] = (dcm[0, 1] - dcm[1, 0]) / (4 * q[3])
        if q[0] < 0:
            q[0] = -q[0]
            q[3] = -q[3]
        q[1] = (dcm[2, 0] + dcm[0, 2]) / (4 * q[3])
        q[2] = (dcm[1, 2] + dcm[2, 1]) / (4 * q[3])
    
    return q

'''===================================================================== CRP Codes ====================================================================='''
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
    # Convert DCM to quaternion
    b = DCM_to_quaternion(dcm)  # b should be a 4-element array [q0, q1, q2, q3]

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

'''===================================================================== MRP Codes ====================================================================='''
def MRP_to_DCM(sigma):
    """
    Converts a Modified Rodrigues Parameter (MRP) vector to a Direction Cosine Matrix (DCM).

    Args:
        sigma (numpy.ndarray): A 3-element array representing the MRP vector.

    Returns:
        numpy.ndarray: A 3x3 DCM matrix.
    """
    # Ensure sigma is a numpy array
    sigma = np.asarray(sigma).flatten()

    if sigma.shape[0] != 3:
        raise ValueError("Input sigma vector must be of length 3.")

    # Compute the Skew-Symmetric matrix of Sigma
    sigma_tilde = skew_symmetric(sigma)

    # 3x3 Identity Matrix
    I_3x3 = np.eye(3)

    # Comput the norm of MRP vector, sigma_squared
    sigma_squared = np.dot(sigma, sigma)

    # Vector Based Computation
    C = I_3x3 + ((8 * np.dot(sigma_tilde, sigma_tilde)) - (4 * (1 - sigma_squared) * sigma_tilde)) / ((1 + sigma_squared)**2)

    return C

def DCM_to_MRP(C):
    """
    Converts a Direction Cosine Matrix (DCM) to a Modified Rodrigues Parameter (MRP) vector.

    Args:
        C (numpy.ndarray): A 3x3 DCM matrix.

    Returns:
        numpy.ndarray: A 3-element array representing the MRP vector (sigma).
    """
    # Ensure C is a numpy array
    C = np.asarray(C)
    if C.shape != (3, 3):
        raise ValueError("Input DCM must be a 3x3 matrix.")

    # Convert DCM to quaternion, [q0, q1, q2, q3]
    q = DCM_to_quaternion(C)  

    # Ensure quaternion is normalized
    #q = q / np.linalg.norm(q)

    # Extract scalar and vector parts
    q0 = q[0]  # Scalar part
    qv = q[1:]  # Vector part (3-element array)

    # Handle the case when q0 is negative to ensure |sigma| <= 1
    if q0 < 0:
        q0 = -q0
        qv = -qv

    # Compute MRP vector
    sigma = qv / (1 + q0)

    return sigma