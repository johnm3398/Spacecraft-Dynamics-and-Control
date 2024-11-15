import numpy as np

from .DCM_utils import *

def PRV_to_DCM(e, phi_deg=None):
    """
    Converts a Principal Rotation Vector (PRV) or Axis-Angle representation to a rotation matrix.

    Args:
        e (array-like): The rotation vector.
            - If `phi_deg` is None, `e` is interpreted as the PRV vector,
              where the direction is the rotation axis and the magnitude is the rotation angle in radians.
            - If `phi_deg` is provided, `e` is interpreted as the rotation axis (unit vector).
        
        phi_deg (float, optional): The rotation angle in degrees. Default is None.

    Returns:
        np.ndarray: A 3x3 rotation matrix.

    Notes:
        - If `phi_deg` is not specified, the function assumes `e` is the PRV vector.
        - If `phi_deg` is specified, the function assumes `e` is a unit vector representing the rotation axis.
        - If the rotation angle is zero, the rotation matrix is the identity matrix.
    """
    # Validate input vector
    validate_vec3(e)

    # Define the threshold for numerical errors
    threshold = 1e-10    

    # Convert e to a NumPy array with float data type
    e = np.array(e, dtype=float)

    # Initialize the variable phi (the angle in radians)
    phi = 0.0

    if phi_deg is None:
        # Interpret e as the PRV vector
        # Norm of the vector is the rotation angle phi in radians
        phi = np.linalg.norm(e)

        if phi == 0:
            e1, e2, e3 = e
            
        else:
            # Normalize e to get the rotation axis
            e = e / phi
            
    else:
        # Interpret e as the rotation axis (unit vector)
        # Validate phi_deg
        if not isinstance(phi_deg, (int, float)):
            raise TypeError("Rotation angle phi_deg must be a numeric value (int or float).")
        
        # Convert phi from degrees to radians
        phi = np.deg2rad(phi_deg)

        if phi == 0:
            e1, e2, e3 = e
        
        # Normalize e to get the rotation axis
        e = e / np.linalg.norm(e)

    e1, e2, e3 = e

    # Calculate the cosine and sine of the angle
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    Sigma = 1 - c_phi

    # Construct the DCM
    C = np.array([[ ((e1**2)*Sigma + c_phi), (e1*e2*Sigma + e3*s_phi), (e1*e3*Sigma - e2*s_phi)],
                  [(e2*e1*Sigma - e3*s_phi),  ((e2**2)*Sigma + c_phi), (e2*e3*Sigma + e1*s_phi)],
                  [(e3*e1*Sigma + e2*s_phi), (e3*e2*Sigma - e1*s_phi),  ((e3**2)*Sigma + c_phi)]])

    C[np.abs(C) < threshold] = 0.0

    return C

def DCM_to_PRV(C):
    """
    Converts a rotation matrix to a Principal Rotation Vector (PRV).

    Args:
        C (np.array): A 3x3 rotation matrix.

    Returns:
        tuple: A PRV represented as (e_vector, phi_angle), where e_vector is the
               rotation axis (unit vector), and phi_angle is the rotation angle in degrees.

    Notes:
        - **Zero Rotation Case:**
             - If the rotation angle φ is close to zero, there is no rotation, and the rotation axis is
               arbitrary. The function defaults the rotation axis to [1, 1, 1] in this case, which corresponds to the diagonal of C
        
        - **Handling 180-Degree Rotations (Edge Case):**
             - When the rotation angle φ is close to 180 degrees (π radians), the standard computation
                of the rotation axis becomes numerically unstable because sin(φ) approaches zero,
                leading to potential division by zero.
              - To handle this, the function derives the rotation axis from Rodrigues' rotation formula.
                For φ = π, the rotation matrix simplifies to:
                
                    C = -I + 2 * e * eᵗ
                
                Rearranged:
                
                    C + I = 2 * e * eᵗ
                
                This means that (C + I) is proportional to the outer product of the rotation axis with itself.
                All columns of (C + I) are scalar multiples of the rotation axis e.
              - The function computes (C + I), calculates the norms of its columns, and selects the column
                with the largest norm as the rotation axis e.
        
        - **Handling Numerical Errors:**
              - A threshold is defined to handle numerical precision issues. Any component of the rotation axis
                with an absolute value below the threshold is set to zero to clean up the output.
              - The threshold also helps in determining if the rotation angle is effectively zero or 180 degrees.
    """
    # Validate Input DCM
    validate_DCM(C)
    
    # Define the threshold for numerical errors
    threshold = 1e-10
    
    # Compute the angle phi from the trace of the rotation matrix
    trace_C = np.trace(C)
    argument = (trace_C - 1) / 2
    # Ensure the argument is within the valid range of arccos
    argument = np.clip(argument, -1.0, 1.0)                       
    phi = np.arccos(argument)

    # Ensure phi is in the range [0, pi]
    phi = np.clip(phi, 0, np.pi)

    # Check if there's no rotation (phi is close to 0)
    if np.isclose(phi, 0, atol=threshold):
        e = np.diag(C).copy()
        
    # Handle the special case where phi is close to pi (180 degrees)
    elif np.isclose(phi, np.pi, atol=threshold):
        # Compute (C + I)
        CpI = C + np.identity(3)

        # Compute the norms of each column in CpI
        norms = np.linalg.norm(CpI, axis=0)

        # Find the index of the column with the largest norm
        index = np.argmax(norms)

        # Extract the column corresponding to the largest norm
        e = CpI[:, index]
            
    else:
        # Compute the rotation axis using the standard formula
        e = (1 / (2 * np.sin(phi))) * np.array([C[1, 2] - C[2, 1],
                                                C[2, 0] - C[0, 2],
                                                C[0, 1] - C[1, 0]])

    # Normalize the rotation axis to ensure it's a unit vector
    e /= np.linalg.norm(e)
    
    # Convert phi to degrees
    phi = np.rad2deg(phi)

    # Set small components of e to zero
    e[np.abs(e) < threshold] = 0.0

    # If phi is very close to zero, set it to zero
    if np.abs(phi) < threshold:
        phi = 0.0

    return e, phi

def Bmat_PRV(e):
    """
    Computes the B matrix which relates the body angular velocity vector (ω) to the derivative of the Principal Rotation Vector (PRV).
    
    The B matrix is used in the kinematic differential equation:

       (d(e)/dt) = [B] * ω

    where:
        - ω is the body angular velocity vector.
        - e is the Principal Rotation Vector (PRV), whose norm is the rotation angle φ (in radians).
        - B is the matrix mapping body rates to PRV rates


    Args:
        e (array-like): Principal Rotation Vector (3-element array or list). 
                        Its norm is the rotation angle (phi) in radians.

    Returns:
        np.ndarray: A 3x3 matrix mapping body rates to PRV rates.
    """
    # Ensure input is a valid 3-element vector
    validate_vec3(e)

    # Convert to a NumPy array
    e = np.array(e, dtype=float)

    # Compute the rotation angle phi from the norm of e
    phi = np.linalg.norm(e)

    # Handle the zero rotation case
    if phi < 1e-10:  # Threshold for numerical stability
        return np.eye(3)

    # Compute the skew-symmetric matrix of e
    e_tilde = skew_symmetric(e)

    # Compute the B matrix using the updated formula
    B = np.eye(3) + 0.5 * e_tilde + (1 / phi**2) * (1 - (phi / 2) / np.tan(phi / 2)) * np.dot(e_tilde, e_tilde)

    return B

def BInvmat_PRV(e):
    """
    Computes the inverse B matrix (B_inv) which relates the derivative of the Principal Rotation Vector (PRV) to the body angular velocity vector (ω).

    The inverse B matrix is used in the kinematic differential equation:

        ω = [B_inv] * (d(e)/dt)

    where:
        - ω is the body angular velocity vector.
        - e is the Principal Rotation Vector (PRV), whose norm is the rotation angle φ (in radians).
        - B_inv is the inverse B matrix.

    Args:
        e (array-like): Principal Rotation Vector (3-element array or list). 
                        Its norm is the rotation angle φ in radians.

    Returns:
        np.ndarray: A 3x3 matrix mapping PRV rates to body rates.
    """
    # Ensure input is a valid 3-element vector
    validate_vec3(e)

    # Convert to a NumPy array
    e = np.array(e, dtype=float)

    # Compute the rotation angle phi from the norm of e
    phi = np.linalg.norm(e)

    # Handle the zero rotation case
    if phi < 1e-10:  # Threshold for numerical stability
        return np.eye(3)

    # Compute the skew-symmetric matrix of e
    e_tilde = skew_symmetric(e)

    # Compute the B^-1 matrix
    B_inv = np.eye(3) - (((1 - np.cos(phi)) / phi**2) * e_tilde) + (((phi - np.sin(phi)) / phi**3) * np.dot(e_tilde, e_tilde))

    return B_inv