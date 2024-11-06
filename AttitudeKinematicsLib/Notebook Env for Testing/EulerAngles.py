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