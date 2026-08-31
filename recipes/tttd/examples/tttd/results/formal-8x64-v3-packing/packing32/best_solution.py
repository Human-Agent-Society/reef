import numpy as np
from scipy.optimize import minimize, differential_evolution

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
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

def generate_initial_guess(num_circles=32):
    """
    Generates multiple initial guesses for the centers and radii of circles.
    """
    initial_guesses = []

    # Hexagonal packing with 6 rows (corrected radius)
    rows_hex_6 = []
    r_hex_6 = 1 / ((6 - 1) * np.sqrt(3) + 2)
    for row_idx in range(6):  # 6 rows
        if row_idx % 2 == 0:
            num_circles_row = 6
        else:
            num_circles_row = 5
        x_positions = np.linspace(r_hex_6, 1 - r_hex_6, num_circles_row)
        y_positions = r_hex_6 + row_idx * np.sqrt(3) * r_hex_6
        for x in x_positions:
            rows_hex_6.append((x, y_positions))
    # Trim to 32 circles
    centers_hex_6 = np.array(rows_hex_6[:32])
    radii_hex_6 = np.full(32, r_hex_6 * 1.05)  # Slightly increased radius
    np.random.seed(42)
    perturbation = 0.01 * np.random.randn(32, 2)
    centers_hex_6 += perturbation
    centers_hex_6 = np.clip(centers_hex_6, [0, 0], [1, 1])
    initial_guesses.append((centers_hex_6, radii_hex_6))

    # Hexagonal packing with 7 rows (corrected radius)
    rows_hex_7 = []
    r_hex_7 = 1 / ((7 - 1) * np.sqrt(3) + 2)
    for row_idx in range(7):  # 7 rows
        if row_idx % 2 == 0:
            num_circles_row = 6
        else:
            num_circles_row = 5
        x_positions = np.linspace(r_hex_7, 1 - r_hex_7, num_circles_row)
        y_positions = r_hex_7 + row_idx * np.sqrt(3) * r_hex_7
        for x in x_positions:
            rows_hex_7.append((x, y_positions))
    # Trim to 32 circles
    centers_hex_7 = np.array(rows_hex_7[:32])
    radii_hex_7 = np.full(32, r_hex_7 * 1.05)
    np.random.seed(43)
    perturbation = 0.01 * np.random.randn(32, 2)
    centers_hex_7 += perturbation
    centers_hex_7 = np.clip(centers_hex_7, [0, 0], [1, 1])
    initial_guesses.append((centers_hex_7, radii_hex_7))

    # Hexagonal packing with 8 rows (corrected radius)
    rows_hex_8 = []
    r_hex_8 = 1 / ((8 - 1) * np.sqrt(3) + 2)
    for row_idx in range(8):  # 8 rows
        if row_idx % 2 == 0:
            num_circles_row = 6
        else:
            num_circles_row = 5
        x_positions = np.linspace(r_hex_8, 1 - r_hex_8, num_circles_row)
        y_positions = r_hex_8 + row_idx * np.sqrt(3) * r_hex_8
        for x in x_positions:
            rows_hex_8.append((x, y_positions))
    # Trim to 32 circles
    centers_hex_8 = np.array(rows_hex_8[:32])
    radii_hex_8 = np.full(32, r_hex_8 * 1.05)
    np.random.seed(44)
    perturbation = 0.01 * np.random.randn(32, 2)
    centers_hex_8 += perturbation
    centers_hex_8 = np.clip(centers_hex_8, [0, 0], [1, 1])
    initial_guesses.append((centers_hex_8, radii_hex_8))

    # Square grid with 4x8 arrangement
    num_rows = 4
    num_cols = 8
    spacing_x = 1.0 / num_cols
    spacing_y = 1.0 / num_rows
    initial_radius = min(spacing_x / 2, spacing_y / 2)
    centers_grid_4x8 = np.zeros((num_rows * num_cols, 2))
    for i in range(num_rows):
        for j in range(num_cols):
            centers_grid_4x8[i*num_cols + j] = [j * spacing_x, i * spacing_y]
    # Trim to 32 circles
    centers_grid_4x8 = centers_grid_4x8[:32]
    radii_grid_4x8 = np.full(32, initial_radius * 1.05)
    np.random.seed(45)
    perturbation = 0.01 * np.random.randn(32, 2)
    centers_grid_4x8 += perturbation
    centers_grid_4x8 = np.clip(centers_grid_4x8, [0, 0], [1, 1])
    initial_guesses.append((centers_grid_4x8, radii_grid_4x8))

    # Square grid with 5x7 arrangement
    num_rows = 5
    num_cols = 7
    spacing_x = 1.0 / num_cols
    spacing_y = 1.0 / num_rows
    initial_radius = min(spacing_x / 2, spacing_y / 2)
    centers_grid_5x7 = np.zeros((num_rows * num_cols, 2))
    for i in range(num_rows):
        for j in range(num_cols):
            centers_grid_5x7[i*num_cols + j] = [j * spacing_x, i * spacing_y]
    # Trim to 32 circles
    centers_grid_5x7 = centers_grid_5x7[:32]
    radii_grid_5x7 = np.full(32, initial_radius * 1.05)
    np.random.seed(46)
    perturbation = 0.01 * np.random.randn(32, 2)
    centers_grid_5x7 += perturbation
    centers_grid_5x7 = np.clip(centers_grid_5x7, [0, 0], [1, 1])
    initial_guesses.append((centers_grid_5x7, radii_grid_5x7))

    # Square grid with 6x6 arrangement
    num_rows = 6
    num_cols = 6
    spacing_x = 1.0 / num_cols
    spacing_y = 1.0 / num_rows
    initial_radius = min(spacing_x / 2, spacing_y / 2)
    centers_grid_6x6 = np.zeros((num_rows * num_cols, 2))
    for i in range(num_rows):
        for j in range(num_cols):
            centers_grid_6x6[i*num_cols + j] = [j * spacing_x, i * spacing_y]
    # Trim to 32 circles
    centers_grid_6x6 = centers_grid_6x6[:32]
    radii_grid_6x6 = np.full(32, initial_radius * 1.05)
    np.random.seed(47)
    perturbation = 0.01 * np.random.randn(32, 2)
    centers_grid_6x6 += perturbation
    centers_grid_6x6 = np.clip(centers_grid_6x6, [0, 0], [1, 1])
    initial_guesses.append((centers_grid_6x6, radii_grid_6x6))

    # Spiral initial guess
    def spiral_coords(num_points, radius_factor=0.1):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radii = np.linspace(radius_factor, radius_factor * 2, num_points)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    spiral_centers = spiral_coords(32)
    spiral_radii = np.full(32, 0.15 * 1.05)
    np.random.seed(48)
    perturbation = 0.01 * np.random.randn(32, 2)
    spiral_centers += perturbation
    spiral_centers = np.clip(spiral_centers, [0, 0], [1, 1])
    initial_guesses.append((spiral_centers, spiral_radii))

    # Random initial guess with larger radii
    np.random.seed(49)
    centers_rand = np.random.rand(32, 2)
    radii_rand = np.full(32, 0.1 * 1.05)  # Larger initial radius
    initial_guesses.append((centers_rand, radii_rand))

    # Corner-based grid
    corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    radii = np.full(4, 0.08 * 1.05)  # initial radius for corners
    remaining = num_circles - 4
    centers_center = np.random.rand(remaining, 2)
    radii_center = np.full(remaining, 0.08 * 1.05)
    centers = np.vstack([corners, centers_center])
    radii = np.hstack([radii, radii_center])
    np.random.seed(50)
    perturbation = 0.01 * np.random.randn(32, 2)
    centers += perturbation
    centers = np.clip(centers, [0, 0], [1, 1])
    initial_guesses.append((centers, radii))

    # Hybrid Grid
    rows_hybrid = []
    r_hybrid = 1 / ((7 - 1) * np.sqrt(3) + 2)
    for row_idx in range(7):  # 7 rows
        if row_idx % 2 == 0:
            num_circles_row = 6
        else:
            num_circles_row = 5
        x_positions = np.linspace(r_hybrid, 1 - r_hybrid, num_circles_row)
        y_positions = r_hybrid + row_idx * np.sqrt(3) * r_hybrid
        for x in x_positions:
            rows_hybrid.append((x, y_positions))
    # Trim to 32 circles
    centers_hybrid = np.array(rows_hybrid[:32])
    radii_hybrid = np.full(32, r_hybrid * 1.05)
    np.random.seed(55)
    perturbation = 0.01 * np.random.randn(32, 2)
    centers_hybrid += perturbation
    centers_hybrid = np.clip(centers_hybrid, [0, 0], [1, 1])
    initial_guesses.append((centers_hybrid, radii_hybrid))

    # Dense grid near the center
    num_rows = 6
    num_cols = 6
    spacing_x = 0.5 / num_cols  # spacing from 0.5 to 0.5
    spacing_y = 0.5 / num_rows
    initial_radius = min(spacing_x / 2, spacing_y / 2)
    centers_dense_grid = np.zeros((num_rows * num_cols, 2))
    for i in range(num_rows):
        for j in range(num_cols):
            centers_dense_grid[i*num_cols + j] = [0.5 + j * spacing_x, 0.5 + i * spacing_y]
    # Trim to 32 circles
    centers_dense_grid = centers_dense_grid[:32]
    radii_dense_grid = np.full(32, initial_radius * 1.05)
    np.random.seed(56)
    perturbation = 0.01 * np.random.randn(32, 2)
    centers_dense_grid += perturbation
    centers_dense_grid = np.clip(centers_dense_grid, [0, 0], [1, 1])
    initial_guesses.append((centers_dense_grid, radii_dense_grid))

    # New corner-based initial guess
    def generate_corner_based_guess(num_circles=32):
        corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        radii = np.full(4, 0.08 * 1.05)  # initial radius for corners
        remaining = num_circles - 4
        centers_center = np.random.rand(remaining, 2)
        radii_center = np.full(remaining, 0.08 * 1.05)
        centers = np.vstack([corners, centers_center])
        radii = np.hstack([radii, radii_center])
        np.random.seed(60)
        perturbation = 0.01 * np.random.randn(32, 2)
        centers += perturbation
        centers = np.clip(centers, [0, 0], [1, 1])
        return (centers, radii)

    initial_guesses.append(generate_corner_based_guess())

    return initial_guesses

def run_packing():
    initial_guesses = generate_initial_guess()
    best_centers = None
    best_radii = None
    best_sum = 0.0

    for centers_initial, radii_initial in initial_guesses:
        variables = np.concatenate([centers_initial.flatten(), radii_initial])

        def objective(x):
            n = 32
            centers = x[:2*n].reshape(n, 2)
            radii = x[2*n:]
            return -np.sum(radii)

        constraints = []
        for i in range(32):
            def constraint_x_min(x, i=i):
                x_i = x[2*i]
                r_i = x[2*32 + i]
                return x_i - r_i

            def constraint_x_max(x, i=i):
                x_i = x[2*i]
                r_i = x[2*32 + i]
                return 1 - x_i - r_i

            def constraint_y_min(x, i=i):
                y_i = x[2*i + 1]
                r_i = x[2*32 + i]
                return y_i - r_i

            def constraint_y_max(x, i=i):
                y_i = x[2*i + 1]
                r_i = x[2*32 + i]
                return 1 - y_i - r_i

            constraints.append({'type': 'ineq', 'fun': constraint_x_min})
            constraints.append({'type': 'ineq', 'fun': constraint_x_max})
            constraints.append({'type': 'ineq', 'fun': constraint_y_min})
            constraints.append({'type': 'ineq', 'fun': constraint_y_max})

        for i in range(32):
            for j in range(i+1, 32):
                def constraint_overlap(x, i=i, j=j):
                    x_i = x[2*i]
                    y_i = x[2*i + 1]
                    r_i = x[2*32 + i]
                    x_j = x[2*j]
                    y_j = x[2*j + 1]
                    r_j = x[2*32 + j]
                    dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                    return dist - r_i - r_j
                constraints.append({'type': 'ineq', 'fun': constraint_overlap})

        bounds = [(0, 1) for _ in range(32*3)]

        result = minimize(
            fun=objective,
            x0=variables,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 50000, 'ftol': 1e-9, 'eps': 1e-7}
        )

        if not result.success:
            continue

        optimal_vars = result.x
        centers_opt = optimal_vars[:2*32].reshape(32, 2)
        radii_opt = optimal_vars[2*32:]
        valid = validate_packing(centers_opt, radii_opt)

        if not valid:
            continue

        sum_radii = np.sum(radii_opt)
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_centers = centers_opt
            best_radii = radii_opt

    if best_centers is None:
        return np.array([]), np.array([]), 0.0

    return best_centers, best_radii, best_sum
