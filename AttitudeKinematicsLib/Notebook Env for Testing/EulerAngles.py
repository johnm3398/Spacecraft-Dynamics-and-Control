import numpy as np

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

def EulerAngles_to_DCM(angles, sequence, transformation_type='passive'):
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

def DCM_to_EulerAngles(DCM, sequence, transformation_type='passive'):
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
    if DCM.shape != (3, 3):
        raise ValueError("DCM must be a 3x3 matrix.")

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