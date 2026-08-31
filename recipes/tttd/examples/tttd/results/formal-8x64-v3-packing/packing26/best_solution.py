import numpy as np
import math
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def run_packing():
    n_circles = 26
    # Improved initial guess with optimized hexagonal packing
    r_initial = 0.1  # Radius for the first four rows
    r_row5 = 1 / 12  # Radius for the fifth row

    # Calculate vertical spacing for rows
    y_prev = r_initial + 3 * math.sqrt(3) * r_initial
    # Calculate y-coordinate for the fifth row to utilize vertical space
    y_row5 = y_prev - math.sqrt(3) * r_row5 + 0.001  # Slight adjustment to utilize vertical space

    # Generate centers with varying horizontal spacing
    centers = []
    for row in range(5):
        if row < 4:
            y = r_initial + row * math.sqrt(3) * r_initial
            for col in range(5):
                x = r_initial + col * 2 * r_initial
                centers.append([x, y])
        else:
            for col in range(6):
                x = r_row5 + col * 2 * r_row5
                centers.append([x, y_row5])
    radii = [r_initial] * (4 * 5) + [r_row5] * 6
    centers = np.array(centers)
    radii = np.array(radii)
    x0 = np.hstack((centers.reshape(-1), radii))

    def objective(x_flat):
        centers_flat = x_flat[:2 * n_circles].reshape(n_circles, 2)
        radii_flat = x_flat[2 * n_circles:]
        return -np.sum(radii_flat)  # Minimize negative sum

    constraints = []

    # Add constraints for bounding box
    for i in range(n_circles):
        # x_i - r_i >= 0
        A = np.zeros(2 * n_circles + n_circles)
        A[2 * i] = 1
        A[2 * n_circles + i] = -1
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A: np.dot(x, A), 'jac': lambda x, A=A: A})
        # x_i + r_i <= 1
        A = np.zeros(2 * n_circles + n_circles)
        A[2 * i] = 1
        A[2 * n_circles + i] = 1
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A: 1 - np.dot(x, A), 'jac': lambda x, A=A: -A})
        # y_i - r_i >= 0
        A = np.zeros(2 * n_circles + n_circles)
        A[2 * i + 1] = 1
        A[2 * n_circles + i] = -1
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A: np.dot(x, A), 'jac': lambda x, A=A: A})
        # y_i + r_i <= 1
        A = np.zeros(2 * n_circles + n_circles)
        A[2 * i + 1] = 1
        A[2 * n_circles + i] = 1
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A: 1 - np.dot(x, A), 'jac': lambda x, A=A: -A})

    # Add circle-to-circle distance constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            def constraint_ij(x, i=i, j=j):
                x_i = x[2 * i]
                y_i = x[2 * i + 1]
                r_i = x[2 * n_circles + i]
                x_j = x[2 * j]
                y_j = x[2 * j + 1]
                r_j = x[2 * n_circles + j]
                dist = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                return dist - r_i - r_j
            constraints.append({'type': 'ineq', 'fun': constraint_ij})

    # Optimize with increased tolerance and iterations
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=constraints,
        tol=1e-12,
        options={'maxiter': 100000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
    )

    if not result.success:
        print("Optimization failed")
        centers_final = centers
        radii_final = radii
        sum_radii = np.sum(radii)
    else:
        x_opt = result.x
        centers_final = x_opt[:2 * n_circles].reshape(n_circles, 2)
        radii_final = x_opt[2 * n_circles:]
        sum_radii = np.sum(radii_final)

    if not validate_packing(centers_final, radii_final):
        print("Validation failed")
        centers_final = centers
        radii_final = radii
        sum_radii = np.sum(radii)

    return centers_final, radii_final, sum_radii
