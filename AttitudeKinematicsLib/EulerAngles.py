import numpy as np

from .DCM_utils import *

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
    matrix = np.array([[1,  0,  0], 
                       [0,  c,  s], 
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
    matrix = np.array([[c,  0, -s], 
                       [0,  1,  0], 
                       [s,  0,  c]])
    
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
    matrix = np.array([[ c,  s,  0], 
                       [-s,  c,  0], 
                       [ 0,  0,  1]])
    
    if transformation_type == 'active':
        return matrix.T
    
    return matrix

def Euler_to_DCM(angles, sequence, transformation_type='passive'):
    """
    Converts a set of Euler angles into a Direction Cosine Matrix (DCM)
    based on the specified rotation sequence.

    Args:
        angles (array-like): A list or array of three rotation angles in degrees,
                             corresponding to the axes in the rotation sequence.
        sequence (str): The rotation sequence as a string (e.g., '321' or 'zyx').
                        The sequence defines the order and axes of rotations.
        transformation_type (str): 'active' or 'passive' rotation.

    Returns:
        numpy.ndarray: A 3x3 Direction Cosine Matrix (DCM).
    """
    if len(angles) != 3:
        raise ValueError("Angles must be a list or array of three elements.")

    if len(sequence) != 3:
        raise ValueError("Rotation sequence must be a string of three characters.")

    # Convert the sequence to lower case to handle both upper and lower case inputs
    sequence = sequence.lower()

    # Validate and map the rotation sequence to 'x', 'y', 'z'
    valid_axes = {'1': 'x', 
                  '2': 'y', 
                  '3': 'z', 
                  'x': 'x', 
                  'y': 'y', 
                  'z': 'z'}
    mapped_sequence = ''
    for axis_char in sequence:
        if axis_char not in valid_axes:
            raise ValueError(f"Invalid axis '{axis_char}' in rotation sequence. Use '1', '2', '3', 'x', 'y', or 'z'.")
        mapped_sequence += valid_axes[axis_char]

    # Now mapped_sequence contains only 'x', 'y', 'z'
    # Map axis characters to rotation functions
    axis_functions = {
        'x': rotation_matrix_x,
        'y': rotation_matrix_y,
        'z': rotation_matrix_z
    }

    # List to store the rotation matrices
    rotation_matrices = []

    # Map each angle to its corresponding axis in the mapped sequence
    for angle_deg, axis_char in zip(angles, mapped_sequence):
        # Get the corresponding rotation function
        rotation_function = axis_functions[axis_char]
        
        # Compute the rotation matrix
        R = rotation_function(angle_deg, transformation_type=transformation_type)
        
        # Append to the list
        rotation_matrices.append(R)

    # Initialize the DCM as an identity matrix
    DCM = np.eye(3)

    # Apply the rotations in reverse order for proper sequencing
    # Since rotations are applied right-to-left (R3 * R2 * R1 * v)
    for R in rotation_matrices:
        DCM = np.matmul(R, DCM)

    return DCM

def DCM_to_Euler(DCM, sequence, transformation_type='passive'):
    """
    Converts a Direction Cosine Matrix (DCM) into a set of Euler angles
    based on the specified rotation sequence.

    Args:
        DCM (numpy.ndarray): A 3x3 Direction Cosine Matrix.
        sequence (str): The rotation sequence as a string (e.g., '321' or 'zyx').
                        The sequence defines the order and axes of rotations.
        transformation_type (str): 'active' or 'passive' rotation.

    Returns:
        tuple: A tuple of three Euler angles in degrees, corresponding to the axes in the rotation sequence.

    Notes:
        - The function handles both 'active' and 'passive' transformations.
        - The output angles are in degrees.
        - The function accounts for possible singularities in the rotation representation.
    """
    # Validate the input DCM
    validate_DCM(DCM)

    # Validate and parse the rotation sequence
    if len(sequence) != 3:
        raise ValueError("Rotation sequence must be a string of three characters.")

    # Convert the sequence to lower case to handle both upper and lower case inputs
    sequence = sequence.lower()

    # Validate and map the rotation sequence to '1', '2', '3'
    valid_axes = {'1', '2', '3', 'x', 'y', 'z'}
    axis_map = {'x': '1', 
                'y': '2', 
                'z': '3'}
    seq_mapped = ''
    for axis_char in sequence:
        if axis_char not in valid_axes:
            raise ValueError(f"Invalid axis '{axis_char}' in rotation sequence.")
        seq_mapped += axis_map.get(axis_char, axis_char)

    # Handle the transformation type once
    if transformation_type == 'active':
        DCM = DCM.T  # Transpose for active transformation
    elif transformation_type != 'passive':
        raise ValueError("transformation_type must be 'active' or 'passive'.")

    # Extract angles based on the rotation sequence
    seq = seq_mapped

    # Initialize angles
    theta_1 = 0
    theta_2 = 0
    theta_3 = 0

    # Use if-elif chain to handle different sequences
    if seq == '121':
        theta_1 = np.arctan2(DCM[0,1], -DCM[0,2])
        theta_2 = np.arccos(DCM[0,0])
        theta_3 = np.arctan2(DCM[1,0], DCM[2,0])
    
    elif seq == '123':
        theta_1 = np.arctan2(-DCM[2,1], DCM[2,2])
        theta_2 = np.arcsin(DCM[2,0])
        theta_3 = np.arctan2(-DCM[1,0], DCM[0,0])
    
    elif seq == '131':
        theta_1 = np.arctan2(DCM[0,2], DCM[0,1])
        theta_2 = np.arccos(DCM[0,0])
        theta_3 = np.arctan2(DCM[2,0], -DCM[1,0])
    
    elif seq == '132':
        theta_1 = np.arctan2(DCM[1,2], DCM[1,1])
        theta_2 = np.arcsin(-DCM[1,0])
        theta_3 = np.arctan2(DCM[2,0], DCM[0,0])
    
    elif seq == '212':
        theta_1 = np.arctan2(DCM[1,0], DCM[1,2])
        theta_2 = np.arccos(DCM[1,1])
        theta_3 = np.arctan2(DCM[0,1], -DCM[2,1])
    
    elif seq == '213':
        theta_1 = np.arctan2(DCM[2,0], DCM[2,2])
        theta_2 = np.arcsin(-DCM[2,1])
        theta_3 = np.arctan2(DCM[0,1], DCM[1,1])
    
    elif seq == '231':
        theta_1 = np.arctan2(-DCM[0,2], DCM[0,0])
        theta_2 = np.arcsin(DCM[0,1])
        theta_3 = np.arctan2(-DCM[2,1], DCM[1,1])
    
    elif seq == '232':
        theta_1 = np.arctan2(DCM[1,2], -DCM[1,0])
        theta_2 = np.arccos(DCM[1,1])
        theta_3 = np.arctan2(DCM[2,1], DCM[0,1])
    
    elif seq == '312':
        theta_1 = np.arctan2(-DCM[1,0], DCM[1,1])
        theta_2 = np.arcsin(DCM[1,2])
        theta_3 = np.arctan2(-DCM[0,2], DCM[2,2])
    
    elif seq == '313':
        theta_1 = np.arctan2(DCM[2,0], -DCM[2,1])
        theta_2 = np.arccos(DCM[2,2])
        theta_3 = np.arctan2(DCM[0,2], DCM[1,2])
    
    elif seq == '321':
        theta_1 = np.arctan2(DCM[0,1], DCM[0,0])
        theta_2 = np.arcsin(-DCM[0,2])
        theta_3 = np.arctan2(DCM[1,2], DCM[2,2])
    
    elif seq == '323':
        theta_1 = np.arctan2(DCM[2,1], DCM[2,0])
        theta_2 = np.arccos(DCM[2,2])
        theta_3 = np.arctan2(DCM[1,2], -DCM[0,2])
    
    else:
        raise NotImplementedError(f"Rotation sequence '{sequence}' is not implemented.")

    # Convert angles from radians to degrees
    angles_deg = np.rad2deg([theta_1, theta_2, theta_3])

    return tuple(angles_deg)

def Bmat_Euler(angles, sequence):
    """
    Computes the B matrix for transforming body angular velocities to Euler angle rates
    based on the specified rotation sequence and Euler angles.

    Args:
        angles (list or tuple): Euler angles [theta_1, theta_2, theta_3] in degrees.
        sequence (str): The rotation sequence as a string (e.g., '321', 'ZYX').

    Returns:
        numpy.ndarray: The 3x3 B matrix that transforms body rates to Euler angle rates 
        for the specified rotation sequence and angles.

    Notes:
        - The function calculates the B matrix for the specified rotation sequence and
          angles, accounting for potential singularities at certain Euler angle values.
        - The B matrix is used to relate body angular velocity `ω` to Euler angle rates `dθ/dt`
          as follows:
          
            dθ/dt = B * ω

        - It supports both proper Euler angles and Tait-Bryan angles.
        - The `sequence` should specify the rotation axes in a 3-character string format 
          (e.g., '123', '321', '232').
        - Angles are input in degrees, but internal computations are performed in radians.
        - Returns a `ValueError` if the angles produce a singularity (e.g., cos(θ₂) = 0).
    """
    # Validate input lengths
    if len(angles) != 3:
        raise ValueError("The 'angles' parameter must have three elements.")
    if len(sequence) != 3:
        raise ValueError("The 'sequence' parameter must be a string of three characters.")

    # Convert input angles from degrees to radians
    angles_rad = np.deg2rad(angles)
    theta_1, theta_2, theta_3 = angles_rad  # Euler angles in radians

    # Map axes to indices (1-based indexing for clarity)
    axis_map = {'1': 1, '2': 2, '3': 3,
                'x': 1, 'y': 2, 'z': 3}

    # Convert sequence to indices
    sequence = sequence.lower()
    try:
        axes = [axis_map[axis] for axis in sequence]
    except KeyError as e:
        raise ValueError(f"Invalid axis '{e.args[0]}' in rotation sequence.")

    # Precompute trigonometric functions with subscripts matching theta_1, theta_2, theta_3
    s1 = np.sin(theta_1)
    c1 = np.cos(theta_1)
    s2 = np.sin(theta_2)
    c2 = np.cos(theta_2)
    s3 = np.sin(theta_3)
    c3 = np.cos(theta_3)

    # Initialize the B matrix
    B = np.zeros((3, 3))

    # Define a small threshold to avoid division by zero
    epsilon = np.finfo(float).eps

    # Compute the B matrix based on the rotation sequence
    # The indices in B correspond to the 1-based axis indices minus 1 (for zero-based indexing)
    if sequence == '121':
        if abs(s2) < epsilon:
            raise ValueError("Singularity encountered: sin(theta_2) is zero.")
        B[0, :] = [ 0,     s3,     c3]
        B[1, :] = [ 0,  s2*c3, -s2*s3]
        B[2, :] = [s2, -c2*s3, -c2*c3]
        B /= s2
        
    elif sequence == '123':
        if abs(c2) < epsilon:
            raise ValueError("Singularity encountered: cos(theta_2) is zero.")
        B[0, :] = [    c3,    -s3,  0]
        B[1, :] = [ c2*s3,  c2*c3,  0]
        B[2, :] = [-s2*c3,  s2*s3, c2]
        B /= c2
        
    elif sequence == '131':
        if abs(s2) < epsilon:
            raise ValueError("Singularity encountered: sin(theta_2) is zero.")
        B[0, :] = [ 0,   -c3,     s3]
        B[1, :] = [ 0, s2*s3,  s2*c3]
        B[2, :] = [s2, c2*c3, -c2*s3]
        B /= s2
        
    elif sequence == '132':
        if abs(c2) < epsilon:
            raise ValueError("Singularity encountered: cos(theta_2) is zero.")
        B[0, :] = [    c3,  0,    s3]
        B[1, :] = [-c2*s3,  0, c2*c3]
        B[2, :] = [ s2*c3, c2, s2*s3]
        B /= c2
        
    elif sequence == '212':
        if abs(s2) < epsilon:
            raise ValueError("Singularity encountered: sin(theta_2) is zero.")
        B[0, :] = [    s3,  0,   -c3]
        B[1, :] = [ s2*c3,  0, s2*s3]
        B[2, :] = [-c2*s3, s2, c2*c3]
        B /= s2
        
    elif sequence == '213':
        if abs(c2) < epsilon:
            raise ValueError("Singularity encountered: cos(theta_2) is zero.")
        B[0, :] = [   s3,     c3,  0]
        B[1, :] = [c2*c3, -c2*s3,  0]
        B[2, :] = [s2*s3,  s2*c3, c2]
        B /= c2
        
    elif sequence == '231':
        if abs(c2) < epsilon:
            raise ValueError("Singularity encountered: cos(theta_2) is zero.")
        B[0, :] = [ 0,     c3,   -s3]
        B[1, :] = [ 0,  c2*s3, c2*c3]
        B[2, :] = [c2, -s2*c3, s2*s3]
        B /= c2
    
    elif sequence == '232':
        if abs(s2) < epsilon:
            raise ValueError("Singularity encountered: sin(theta_2) is zero.")
        B[0, :] = [    c3,  0,     s3]
        B[1, :] = [-s2*s3,  0,  s2*c3]
        B[2, :] = [-c2*c3, s2, -c2*s3]
        B /= s2
        
    elif sequence == '312':
        if abs(c2) < epsilon:
            raise ValueError("Singularity encountered: cos(theta_2) is zero.")
        B[0, :] = [  -s3,  0,     c3]
        B[1, :] = [c2*c3,  0,  c2*s3]
        B[2, :] = [s2*s3, c2, -s2*c3]
        B /= c2
        
    elif sequence == '313':
        if abs(s2) < epsilon:
            raise ValueError("Singularity encountered: sin(theta_2) is zero.")
        B[0, :] = [    s3,     c3,  0]
        B[1, :] = [ c3*s2, -s3*s2,  0]
        B[2, :] = [-s3*c2, -c3*c2, s2]
        B /= s2
        
    elif sequence == '321':
        if abs(c2) < epsilon:
            raise ValueError("Singularity encountered: cos(theta_2) is zero.")
        B[0, :] = [ 0,    s3,     c3]
        B[1, :] = [ 0, c2*c3, -c2*s3]
        B[2, :] = [c2, s2*s3,  s2*c3]
        B /= c2
        
    elif sequence == '323':
        if abs(s2) < epsilon:
            raise ValueError("Singularity encountered: sin(theta_2) is zero.")
        B[0, :] = [  -c3,     s3,  0]
        B[1, :] = [s2*s3,  s2*c3,  0]
        B[2, :] = [c2*c3, -c2*s3, s2]
        B /= s2
        
    else:
        raise NotImplementedError(f"Rotation sequence '{sequence}' is not implemented.")

    return B

def BInvmat_Euler(angles, sequence):
    """
    Computes the inverse B matrix (B_inv) for transforming Euler angle rates 
    to body angular velocities, based on the specified rotation sequence and 
    Euler angles.

    Args:
        angles (list or tuple): Euler angles [theta_1, theta_2, theta_3] in degrees.
        sequence (str): The rotation sequence as a string (e.g., '321', 'ZYX').

    Returns:
        numpy.ndarray: The 3x3 B_inv matrix that transforms Euler angle rates to 
        body angular velocities for the specified rotation sequence and angles.

    Notes:
        - This function calculates the inverse B matrix, B_inv, which relates Euler angle 
          rates `dθ/dt` to body angular velocity `ω` as follows:
          
            ω = B_inv * dθ/dt

        - The function supports both proper Euler angles and Tait-Bryan angles.
        - The `sequence` parameter should specify the rotation axes in a 3-character string 
          format (e.g., '123', '321', '232').
        - Angles are input in degrees; internal computations are performed in radians.
        - This function is not prone to singularities, as it does not involve divisions 
          by trigonometric terms that vanish at certain Euler angle values.
        - A `NotImplementedError` is raised if an unsupported rotation sequence is specified.
    """
    # Validate input lengths
    if len(angles) != 3:
        raise ValueError("The 'angles' parameter must have three elements.")
    if len(sequence) != 3:
        raise ValueError("The 'sequence' parameter must be a string of three characters.")

    # Convert input angles from degrees to radians
    angles_rad = np.deg2rad(angles)
    theta_1, theta_2, theta_3 = angles_rad  # Euler angles in radians

    # Map axes to indices (1-based indexing for clarity)
    axis_map = {'1': 1, '2': 2, '3': 3,
                'x': 1, 'y': 2, 'z': 3}

    # Convert sequence to indices
    sequence = sequence.lower()
    try:
        axes = [axis_map[axis] for axis in sequence]
    except KeyError as e:
        raise ValueError(f"Invalid axis '{e.args[0]}' in rotation sequence.")

    # Precompute trigonometric functions with subscripts matching theta_1, theta_2, theta_3
    s1 = np.sin(theta_1)
    c1 = np.cos(theta_1)
    s2 = np.sin(theta_2)
    c2 = np.cos(theta_2)
    s3 = np.sin(theta_3)
    c3 = np.cos(theta_3)

    # Initialize the B matrix
    B_inv = np.zeros((3, 3))

    # Define a small threshold to avoid division by zero
    epsilon = np.finfo(float).eps

    # Compute the B Inverse matrix based on the rotation sequence
    # The indices in B correspond to the 1-based axis indices minus 1 (for zero-based indexing)
    if sequence == '121':
        B_inv[0, :] = [   c2,   0,   1]
        B_inv[1, :] = [s2*s3,  c3,   0]
        B_inv[2, :] = [s2*c3, -s3,   0]
        
    elif sequence == '123':
        B_inv[0, :] = [ c2*c3,  s3,   0]
        B_inv[1, :] = [-c2*s3,  c3,   0]
        B_inv[2, :] = [    s2,   0,   1]
        
    elif sequence == '131':
        B_inv[0, :] = [    c2,   0,   1]
        B_inv[1, :] = [-s2*c3,  s3,   0]
        B_inv[2, :] = [ s2*s3,  c3,   0]
        
    elif sequence == '132':
        B_inv[0, :] = [ c2*c3, -s3,   0]
        B_inv[1, :] = [   -s2,   0,   1]
        B_inv[2, :] = [ c2*s3,  c3,   0]
        
    elif sequence == '212':
        B_inv[0, :] = [ s2*s3,  c3,   0]
        B_inv[1, :] = [    c2,   0,   1]
        B_inv[2, :] = [-s2*c3,  s3,   0]
        
    elif sequence == '213':
        B_inv[0, :] = [ c2*s3,  c3,   0]
        B_inv[1, :] = [ c2*c3, -s3,   0]
        B_inv[2, :] = [   -s2,   0,   1]
        
    elif sequence == '231':
        B_inv[0, :] = [    s2,   0,   1]
        B_inv[1, :] = [ c2*c3,  s3,   0]
        B_inv[2, :] = [-c2*s3,  c3,   0]
    
    elif sequence == '232':
        B_inv[0, :] = [ s2*c3, -s3,   0]
        B_inv[1, :] = [    c2,   0,   1]
        B_inv[2, :] = [ s2*s3,  c3,   0]
        
    elif sequence == '312':
        B_inv[0, :] = [-c2*s3,  c3,   0]
        B_inv[1, :] = [    s2,   0,   1]
        B_inv[2, :] = [ c2*c3,  s3,   0]
        
    elif sequence == '313':
        B_inv[0, :] = [ s3*s2,  c3,   0]
        B_inv[1, :] = [ s2*c3, -s3,   0]
        B_inv[2, :] = [    c2,   0,   1]
        
    elif sequence == '321':
        B_inv[0, :] = [   -s2,   0,   1]
        B_inv[1, :] = [ c2*s3,  c3,   0]
        B_inv[2, :] = [ c2*c3, -s3,   0]
        
    elif sequence == '323':
        B_inv[0, :] = [-s2*c3,  s3,   0]
        B_inv[1, :] = [ s2*s3,  c3,   0]
        B_inv[2, :] = [    c2,   0,   1]
        
    else:
        raise NotImplementedError(f"Rotation sequence '{sequence}' is not implemented.")

    return B_inv