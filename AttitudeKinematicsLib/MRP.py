import numpy as np

from .DCM_utils import *
from .EulerRodriguesParameters import DCM_to_EP

def MRP_shadow(sigma: np.ndarray) -> np.ndarray:
    """
    Enforce the principal MRP set by applying the shadow transformation.

    Parameters
    ----------
    sigma : (3,) array_like
        Modified Rodrigues Parameters vector.

    Returns
    -------
    sigma_out : (3,) ndarray
        Equivalent MRP vector. If ||sigma|| > 1, the shadow set
        sigma_shadow = -sigma / ||sigma||^2 is returned to maintain
        numerical conditioning. Otherwise sigma is returned unchanged.

    Notes
    -----
    The shadow set represents the same physical attitude but keeps
    the parameter norm bounded (||sigma|| <= 1), preventing numerical
    growth during integration.
    """
    validate_vec3(sigma)
    sigma_norm = np.linalg.norm(sigma)
    sigma_sq = np.dot(sigma, sigma)
    if sigma_sq > 1.0:
        return -sigma / sigma_sq
    return sigma

def MRP_to_DCM(sigma):
    """
    Convert a Modified Rodrigues Parameters (MRP) vector to a passive
    Direction Cosine Matrix (DCM).

    This function implements the standard MRP-to-DCM mapping used in
    spacecraft attitude dynamics.

    If sigma represents the attitude sigma_BN (body relative to inertial),
    the returned matrix C corresponds to the passive DCM [BN], meaning:

        v_B = C * v_N

    where v_N are vector components expressed in the inertial frame and
    v_B are the same vector components expressed in the body frame.

    Parameters
    ----------
    sigma : array_like, shape (3,)
        Modified Rodrigues Parameters vector.

    Returns
    -------
    C : ndarray, shape (3,3)
        Passive Direction Cosine Matrix corresponding to sigma.

    Notes
    -----
    - The returned matrix is orthogonal and has determinant +1.
    - The mapping follows the standard formulation:
        C = I + (8 * S^2 - 4 * (1 - sigma_sq) * S) / (1 + sigma_sq)^2
      where S is the skew-symmetric matrix of sigma and
      sigma_sq = sigma dot sigma.
    - The function assumes a passive rotation convention.
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

def DCM_to_MRP(C: np.ndarray) -> np.ndarray:
    """
    Convert a Direction Cosine Matrix (DCM) to Modified Rodrigues Parameters (MRPs).

    This implementation uses the closed-form DCM to MRP relation (no quaternion
    intermediate). For a proper orthogonal DCM C, define:

        z = sqrt(trace(C) + 1)

    Then the MRP vector is computed as:

        sigma = 1 / ( z * (z + 2) ) * [ C12 - C21,
                                        C20 - C02,
                                        C01 - C10 ]
    Note that the indices of the elements are reflective of programming conventions (0, 1, 2...)

    This produces an MRP representation consistent with the standard Schaub
    formulation for passive DCMs.

    Parameters
    ----------
    C : ndarray, shape (3,3)
        Direction Cosine Matrix (must be a valid rotation matrix).

    Returns
    -------
    sigma : ndarray, shape (3,)
        Modified Rodrigues Parameters vector corresponding to C.

    Notes
    -----
    - This function does not apply the MRP shadow set. If you require the
      principal set (norm <= 1), apply MRP_shadow(sigma) at the call site.
    - The expression involves z = sqrt(trace(C) + 1). For rotations near 180 deg,
      trace(C) approaches -1, which can amplify numerical sensitivity. This is
      inherent to this closed-form extraction.

    """
    validate_DCM(C)

    z = np.sqrt(np.trace(C) + 1.0)
    sigma = (1.0 / (z * (z + 2.0))) * np.array(
        [
            C[1, 2] - C[2, 1],
            C[2, 0] - C[0, 2],
            C[0, 1] - C[1, 0],
        ],
        dtype=float,
    )

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