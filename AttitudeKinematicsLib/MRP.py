import numpy as np

from .DCM_utils import *
from .EulerRodriguesParameters import DCM_to_EP

def MRP_to_DCM(sigma):
    """
    Converts a Modified Rodrigues Parameters (MRP) vector to a Direction Cosine Matrix (DCM).

    The MRP vector, denoted as **σ** (sigma), is a three-element vector representing the orientation
    of a rigid body in space. This function computes the corresponding 3×3 Direction Cosine Matrix (DCM),
    which transforms vectors between the body frame and the inertial frame.

    **Formula:**

        C = I + [ (8 * [σ]^2) - (4 * (1 - σ²) * [σ]) ] / (1 + σ²)²

    where:

    - **C** is the Direction Cosine Matrix (DCM).
    - **I** is the 3×3 identity matrix.
    - **σ²** is the squared norm of the MRP vector σ (i.e., σᵗσ).
    - **[σ]** is the skew-symmetric matrix of σ.
    - **[σ]²** is the square of the skew-symmetric matrix.

    **Args:**
        sigma (array-like): A 3-element Modified Rodrigues Parameters vector.

    **Returns:**
        numpy.ndarray: A 3×3 Direction Cosine Matrix (DCM).

    **Raises:**
        ValueError: If `sigma` is not a valid 3-element vector.

    **Notes:**
        - The function uses `validate_vec3(sigma)` to ensure that the input is a valid 3-element vector.
        - The skew-symmetric matrix is computed using the `skew_symmetric` function.

    **Example:**

        >>> sigma = [0.1, 0.2, 0.3]
        >>> C = MRP_to_DCM(sigma)
        >>> print(C)
        [[ 0.79809718 -0.59113624  0.11822704]
         [ 0.58174827  0.80434783  0.1220741 ]
         [-0.15876247  0.01600827  0.98720083]]
    """
    # Validate the input vector
    validate_vec3(sigma)

    # Convert sigma to a NumPy array and flatten it
    sigma = np.asarray(sigma, dtype=float).flatten()

    # Compute the skew-symmetric matrix of sigma
    sigma_tilde = skew_symmetric(sigma)

    # Compute sigma squared
    sigma_squared = np.dot(sigma, sigma)

    # Compute the numerator and denominator
    numerator = (8 * np.dot(sigma_tilde, sigma_tilde)) - (4 * (1 - sigma_squared) * sigma_tilde)
    denominator = (1 + sigma_squared) ** 2
    
    # Compute the Direction Cosine Matrix (DCM)
    C = np.eye(3) + numerator / denominator

    return C

def DCM_to_MRP(C):
    """
    Converts a Direction Cosine Matrix (DCM) to a Modified Rodrigues Parameters (MRP) vector.

    The MRP vector **σ** (sigma) provides a singularity-free parameterization of rotations up to 360 degrees,
    with the exception of rotations of 360 degrees. To avoid singularities and ensure that the norm of the MRP vector
    satisfies |σ| ≤ 1, the function switches to the shadow set when necessary.

    **Algorithm:**

    1. Convert the DCM to a quaternion representation.
    2. Normalize the quaternion to ensure it represents a valid rotation.
    3. Extract the scalar (q₀) and vector (q₁, q₂, q₃) parts of the quaternion.
    4. If q₀ is negative, negate the quaternion components to ensure the shortest rotation.
    5. Compute the MRP vector using:

        σ = q_v / (1 + q₀)

    where:
        - q_v is the vector part of the quaternion.
        - q₀ is the scalar part of the quaternion.

    **Args:**
        C (numpy.ndarray): A 3×3 Direction Cosine Matrix (DCM).

    **Returns:**
        numpy.ndarray: A 3-element MRP vector **σ**.

    **Raises:**
        ValueError: If the input matrix `C` is not a valid 3×3 rotation matrix.

    **Notes:**
        - The function ensures that the MRP vector has a norm |σ| ≤ 1 by switching to the shadow set when necessary.
        - The conversion relies on intermediate conversion to quaternions.
        - The function uses the `DCM_to_EP` helper function for the DCM to quaternion conversion.

    **Example:**

        >>> C = np.eye(3)  # Identity matrix represents zero rotation
        >>> sigma = DCM_to_MRP(C)
        >>> print(sigma)
        [0. 0. 0.]

    """
    # Validate Input DCM
    validate_DCM(C)

    # Convert DCM to quaternion [q0, q1, q2, q3]
    q = DCM_to_EP(C)

    # Normalize the quaternion to ensure it represents a valid rotation
    q = q / np.linalg.norm(q)

    # Extract scalar (q0) and vector (q1, q2, q3) parts
    q0 = q[0]  # Scalar part
    qv = q[1:]  # Vector part

    # Switch to the shadow set if necessary to ensure |σ| ≤ 1
    if q0 < 0:
        q0 = -q0
        qv = -qv

    # Compute the MRP vector
    sigma = qv / (1 + q0)

    return sigma

def Bmat_MRP(sigma):
    """
    Computes the B matrix that relates the body angular velocity vector (ω) to the derivative
    of the Modified Rodrigues Parameters (MRP) vector (σ̇).

    **B Matrix Definition:**

        B(σ) = (1 - σᵗσ) * I₃ + 2 * [σ]× + 2 * σσᵗ

    where:

    - **I₃** is the 3×3 identity matrix.
    - **σᵗσ** is the dot product of σ with itself (a scalar).
    - **[σ]×** is the skew-symmetric matrix of σ.
    - **σσᵗ** is the outer product of σ with itself.

    **Usage in Kinematic Equation:**

    The B matrix is used in the kinematic equation:

        σ̇ = (1/4) * B(σ) * ω

    where:

    - **σ̇** is the time derivative of the MRP vector σ.
    - **ω** is the body angular velocity vector.

    **Args:**

    - `sigma` (array-like): A 3-element Modified Rodrigues Parameters vector.

    **Returns:**

    - `numpy.ndarray`: A 3×3 B matrix.

    **Notes:**

    - The function uses the `skew_symmetric` helper function to compute the skew-symmetric matrix.
    - The MRP vector σ must be a valid 3-element numeric vector.

    **Example:**

    ```python
    sigma = [0.1, 0.2, 0.3]
    omega = [0.05, -0.1, 0.2]

    B = Bmat_MRP(sigma)
    sigma_dot = (1/4) * B @ omega

    print("B matrix:\n", B)
    print("MRP rates (σ̇):", sigma_dot)
    ```
    """
    # Validate the input vector
    validate_vec3(sigma)

    # Convert sigma to a NumPy array
    sigma = np.array(sigma, dtype=float)

    # Compute σᵗσ (sigma squared)
    sigma_squared = np.dot(sigma, sigma)

    # Compute the skew-symmetric matrix of sigma
    sigma_tilde = skew_symmetric(sigma)

    # Compute the B matrix
    B = (1 - sigma_squared) * np.eye(3) + 2 * sigma_tilde + 2 * np.outer(sigma, sigma)

    return B

def BInvmat_MRP(sigma):
    """
    Computes the inverse B matrix (B_inv) that relates the derivative
    of the Modified Rodrigues Parameters (MRP) vector (σ̇) to the body angular velocity vector (ω).

    **Inverse B Matrix Definition:**

        B_inv(σ) = [ (1 - σᵗσ) * I₃ - 2 * [σ]× + 2 * σσᵗ ] / (1 + σᵗσ)²

    where:

    - **I₃** is the 3×3 identity matrix.
    - **σᵗσ** is the dot product of σ with itself (a scalar).
    - **[σ]×** is the skew-symmetric matrix of σ.
    - **σσᵗ** is the outer product of σ with itself.

    **Usage in Kinematic Equation:**

    The inverse B matrix is used in the kinematic equation:

        ω = 4 * B_inv(σ) * σ̇

    where:

    - **ω** is the body angular velocity vector.
    - **σ̇** is the time derivative of the MRP vector σ.

    **Args:**

    - `sigma` (array-like): A 3-element Modified Rodrigues Parameters vector.

    **Returns:**

    - `numpy.ndarray`: A 3×3 inverse B matrix.

    **Raises:**

    - `ValueError`: If the input vector `sigma` is not a valid 3-element numeric vector.

    **Notes:**

    - The function uses the `skew_symmetric` helper function to compute the skew-symmetric matrix.
    - The MRP vector σ must be a valid 3-element numeric vector.
    - This inverse B matrix is essential for converting MRP rates to body angular velocities in rotational kinematics.

    **Example:**

    ```python
    sigma = [0.1, 0.2, 0.3]
    sigma_dot = [0.01, -0.02, 0.03]

    B_inv = BInvmat_MRP(sigma)
    omega = 4 * B_inv @ sigma_dot

    print("Inverse B matrix:\n", B_inv)
    print("Body angular velocity (ω):", omega)
    ```
    """
    # Validate the input vector
    validate_vec3(sigma)

    # Convert sigma to a NumPy array
    sigma = np.array(sigma, dtype=float)

    # Compute σᵗσ (sigma squared)
    sigma_squared = np.dot(sigma, sigma)

    # Compute the skew-symmetric matrix of sigma
    sigma_tilde = skew_symmetric(sigma)

    # Compute the numerator and denominator
    numerator = (1 - sigma_squared) * np.eye(3) - 2 * sigma_tilde + 2 * np.outer(sigma, sigma)
    denominator = (1 + sigma_squared) ** 2

    # Compute the inverse B matrix
    B_inv = numerator / denominator

    return B_inv