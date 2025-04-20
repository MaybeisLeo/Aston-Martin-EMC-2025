import subprocess
import os
from time import perf_counter
import time
import matplotlib.pyplot as plt
import numpy as np
from pyswarms.single.local_best import LocalBestPSO
from pyswarms.single.global_best import GlobalBestPSO
from pyswarm import pso
from scipy.stats import qmc

base_dir = "C:/Users/kgain/Desktop/Aston Martin EMC"
xfoil_path = os.path.join(base_dir, "xfoil.exe")
airfoil_file = os.path.join(base_dir, "airfoil_temp.dat")
results_file = os.path.join(base_dir, "results.csv")
cache = {}
bad_result = [-1e6, 1e6, -1e6]

lower_bounds = np.array([0.003, 0.25, 0.04, -0.5, 0.4, -0.05, 0.6, 0.002, 0.000, 0.02, -0.1])
upper_bounds = np.array([0.02, 0.55, 0.12, -0.05, 0.75, -0.005, 2.0, 0.015, 0.01,0.08,-0.02])

# Normalize bounds for optimization
norm_lower = np.zeros_like(lower_bounds)
norm_upper = np.ones_like(upper_bounds)
bounds = np.array([norm_lower, norm_upper])

def denormalize_params(params):
    return lower_bounds + params * (upper_bounds - lower_bounds)

with open("constants.txt", "r") as f:
    lines = f.readlines()
    focus, velocity, AoA, drag_lb, drag_ub, lift_lb, lift_ub = lines
    focus, velocity, AoA = int(focus), float(velocity), float(AoA)
    drag_lb, drag_ub, lift_lb, lift_ub = float(drag_lb), float(drag_ub), float(lift_lb), float(lift_ub)
def calc_re(v):
    v = 1000 * float(v) / 3600
    c = 1
    p = 1.225
    u = 1.81e-5
    return round(p * v * c / u)
def calc_mach(v):
    v = 1000 * float(v) / 3600
    a = 343
    return round(v / a, 4)

RE = 1e5
MACH = 0

def read_output(output_file):
    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
            line = lines[-1].split()
            lift, drag = float(line[1]), float(line[2])
            if drag == 0:
                return bad_result
            ratio = round(lift / drag, 7)
        return [lift, drag, ratio]
    except:
        return bad_result

def run_xfoil(load_file=None, input_file=None, output_file=None, RE=None, MACH=None, AoA=None):
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(input_file):
        os.remove(input_file)
    with open(input_file, "w") as f:
        f.write(f"LOAD {load_file}\nPANE\n")
        f.write("OPER\n")
        f.write(f"ITER {200}\n")
        f.write(f"VISC {RE}\n")
        f.write(f"MACH {MACH}\n")
        f.write(f"PACC\n{output_file}\n\n")
        f.write(f"A {AoA}\n")
        f.write(f"PACC\n")
        f.write("QUIT\n")
    try:
        result = subprocess.run(
            [xfoil_path], stdin=open(input_file, "r"), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=base_dir, timeout=2, startupinfo=startupinfo
        )
        if "VISCAL: Convergence failed" in result.stdout:
            return bad_result
    except subprocess.TimeoutExpired:
        return bad_result
    result = read_output(output_file)
    os.remove(output_file)
    os.remove(input_file)
    return result

# PARSEC Optimization
def generate_airfoil(params, filename='airfoil.dat', plot_airfoil=False):
    """Generate airfoil using PARSEC parameterization with thickness verification.

    Args:
        params: List of PARSEC parameters [r_le, X_up, Z_up, Z_xx_up, X_lo, Z_lo, Z_xx_lo, Z_te, Delta_Z_te, alpha_te, beta_te]
        filename: Output filename for airfoil coordinates
        plot_airfoil: Whether to plot the airfoil

    Returns:
        True if the airfoil is valid (no self-intersections), False otherwise
    """
    r_le, X_up, Z_up, Z_xx_up, X_lo, Z_lo, Z_xx_lo, Z_te, Delta_Z_te, alpha_te, beta_te = params

    a1 = np.sqrt(2 * r_le)
    b1 = -np.sqrt(2 * r_le)
    exponents = [1.5, 2.5, 3.5, 4.5, 5.5]

    te_z_upper = Delta_Z_te + Z_te / 2
    te_z_lower = Delta_Z_te - Z_te / 2

    def solve_coefficients(X, Z, Z_xx, edge_coeff, te_z, te_angle):
        A = np.zeros((5, 5))
        A[0, :] = [X ** exp for exp in exponents]
        A[1, :] = [exp * X ** (exp - 1) for exp in exponents]
        A[2, :] = [exp * (exp - 1) * X ** (exp - 2) for exp in exponents]
        A[3, :] = [1.0 ** exp for exp in exponents]
        A[4, :] = [exp * 1.0 ** (exp - 1) for exp in exponents]

        b = np.array([
            Z - edge_coeff * X ** 0.5,
            -0.5 * edge_coeff / np.sqrt(X),
            Z_xx + 0.25 * edge_coeff / (X ** 1.5),
            te_z - edge_coeff * 1.0 ** 0.5,
            np.tan(te_angle) - 0.5 * edge_coeff / np.sqrt(1.0)
        ])

        return [edge_coeff] + list(np.linalg.solve(A, b))

    upper_coeffs = solve_coefficients(X_up, Z_up, Z_xx_up, a1, te_z_upper, alpha_te)
    lower_coeffs = solve_coefficients(X_lo, Z_lo, Z_xx_lo, b1, te_z_lower, beta_te)

    def compute_z(x, coeffs):
        return sum(c * x ** e for c, e in zip(coeffs, [0.5] + exponents))

    N = 150
    beta = np.linspace(0, np.pi, N)
    x_dist = 0.5 * (1 - np.cos(beta))

    x_upper, x_lower = np.flip(x_dist), x_dist
    z_upper = np.array([compute_z(xi, upper_coeffs) for xi in x_upper])
    z_lower = np.array([compute_z(xi, lower_coeffs) for xi in x_lower])

    x_upper[0], x_lower[-1] = 1.0, 1.0
    z_upper[0], z_lower[-1] = te_z_upper, te_z_lower

    x_upper[-1], x_lower[0] = 0.0, 0.0
    le_z = (z_upper[-1] + z_lower[0]) / 2
    z_upper[-1], z_lower[0] = le_z, le_z

    # Thickness verification
    # Create common x points for comparison
    common_x = np.linspace(0.001, 0.999, 100)  # Avoid exact 0 and 1 to prevent interpolation issues

    # Interpolate upper and lower surfaces to common x points
    from scipy.interpolate import interp1d

    upper_interp = interp1d(x_upper, z_upper)
    lower_interp = interp1d(x_lower, z_lower)

    # Calculate upper and lower z values at common x points
    z_upper_interp = upper_interp(common_x)
    z_lower_interp = lower_interp(common_x)

    # Check thickness at each x coordinate
    min_thickness = 0.0001  # Minimum allowed thickness
    thickness = z_upper_interp - z_lower_interp

    # If any point has negative or too small thickness, airfoil is invalid
    if np.any(thickness < min_thickness):
        return False

    x_coords = np.concatenate([x_upper, x_lower[1:]])
    z_coords = np.concatenate([z_upper, z_lower[1:]])

    with open(filename, 'w') as f:
        f.write("PARSEC Optimized Airfoil\n")
        for x, z in zip(x_coords, z_coords):
            f.write(f"{x:.6f} {z:.6f}\n")

    if plot_airfoil:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_upper, z_upper, 'b-', label='Upper Surface')
        ax.plot(x_lower, z_lower, 'r-', label='Lower Surface')
        ax.scatter([1.0], [te_z_upper], color='g', marker='o', label="Trailing Edge")
        ax.scatter([0.0], [le_z], color='m', marker='o', label="Leading Edge")
        ax.set_xlabel('x/c')
        ax.set_ylabel('z/c')
        ax.set_title('Airfoil Shape')
        ax.legend()
        ax.axis('equal')
        ax.grid(True)

        inset_ax = fig.add_axes((0.65, 0.15, 0.2, 0.2))
        inset_ax.plot(x_upper[:10], z_upper[:10], 'b-')
        inset_ax.plot(x_lower[-10:], z_lower[-10:], 'r-')
        inset_ax.scatter([1.0], [te_z_upper], color='g', marker='o')
        inset_ax.set_xlim(0.95, 1.01)
        inset_ax.set_ylim(min(z_lower[-10:]) - 0.001, max(z_upper[:10]) + 0.001)
        inset_ax.grid(True)
        inset_ax.set_title('TE Closeup')

        plt.show()

    return True

def objective_PARSEC(parameters):
    n_particles = parameters.shape[0]
    results = np.zeros(n_particles)
    for i in range(n_particles):
        # Reflect particles that go out of bounds
        for j in range(parameters.shape[1]):
            if parameters[i, j] < 0:
                parameters[i, j] = -parameters[i, j]  # Reflect back into [0, 1]
            elif parameters[i, j] > 1:
                parameters[i, j] = 2 - parameters[i, j]  # Reflect from upper bound
            # If still out of bounds (e.g., param < -1 or > 2), clamp
            parameters[i, j] = np.clip(parameters[i, j], 0, 1)
        denorm_params = denormalize_params(parameters[i])
        pid = os.getpid()
        timestamp = int(time.time() * 1000) % 10000
        filename = f'temp_airfoil_{pid}_{timestamp}.dat'
        output_file = f'parsec_output_{pid}_{timestamp}.txt'
        input_file = f'parsec_input_{pid}_{timestamp}.txt'
        try:
            _ = generate_airfoil(denorm_params, filename, plot_airfoil=False)
            if _ == True:
                result = run_xfoil(load_file=filename, input_file=input_file, output_file=output_file, RE=RE, MACH=MACH, AoA=AoA)
                lift, drag, ratio = result
                if (drag_lb != -1 and drag < drag_lb) or (drag_ub != -1 and drag > drag_ub):
                    drag = 1e6
                if (lift_lb != -1 and lift < lift_lb) or (lift_ub != -1 and lift > lift_ub):
                    lift = -1e6
                if focus == 0:
                    results[i] = -lift #Maximize lift
                elif focus == 1:
                    results[i] = drag #Minimize drag
                elif focus == 2:
                    results[i] = -ratio #Maximize lift/drag ratio
                else:
                    print(f"The value of target:{focus}")
                    print("Error")
                    results[i] = 1e6
            else:
                results[i] = 1e6

        except Exception as e:
            print(f"Error with params {parameters[i]}: {e}")
            results[i] = 1e6
        finally:
            for f in [filename, input_file, output_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
    return results

def pso_PARSEC(iters, n_particles):
    dimensions = 11
    sampler = qmc.LatinHypercube(d=dimensions)
    sample = sampler.random(n=n_particles)
    scaled = qmc.scale(sample, norm_lower, norm_upper)
    options = {
        'c1': 0.15, 'c2': 0.05, 'w': 0.75,
        'k': n_particles//10, 'p': 2,
        'velocity_clamp': (-0.5, 0.5)  # Velocity clamping added
    }


    optimizer = LocalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds, init_pos=scaled)
    best_cost, best_position = optimizer.optimize(objective_PARSEC, iters=iters, verbose=True, n_processes=4)
    best_position = denormalize_params(best_position)
    print(f"Optimal Solution: {best_position}")
    print(f"Minimum Cost: {best_cost}")
    generate_airfoil(best_position, 'optimized_airfoil.dat', plot_airfoil=True)
    print("Optimized airfoil saved as 'optimized_airfoil.dat' and plotted.")
    history = np.negative(optimizer.cost_history)
    plt.plot(history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Convergence Plot')
    plt.show()


def main():
    start = perf_counter()
    pso_PARSEC(100, 50)
    end = perf_counter()
    period = end - start
    print(f"Execution finished in {round(period / 60, 2)} minutes")

if __name__ == "__main__":
    main()
