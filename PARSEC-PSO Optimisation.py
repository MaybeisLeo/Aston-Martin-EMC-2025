import subprocess
import os
from time import perf_counter
import time
import matplotlib.pyplot as plt
import numpy as np
from pyswarms.single.local_best import LocalBestPSO
from scipy.stats import qmc
import random

base_dir = "C:/Users/kgain/Desktop/Aston Martin EMC" #To change
xfoil_path = os.path.join(base_dir, "xfoil.exe")
airfoil_file = os.path.join(base_dir, "airfoil_temp.dat")
results_file = os.path.join(base_dir, "results.csv")
bad_result = [-1e6, 1e6, -1e6]

lower_bounds = np.array([0.002, 0.10, 0.02, -1.00, 0.10, -0.10, 0.30, 0.000, -0.020, 0.000, -0.30])
upper_bounds = np.array([0.040, 0.70, 0.15,  0.10, 0.95, -0.001, 3.00, 0.020,  0.020, 0.120, -0.001])

#Wider bounds
lower_bounds = np.array([
    0.002,  # r_le (leading edge radius)
    0.10,   # X_up (position of max upper camber)
    0.02,   # Z_up (max upper camber)
   -1.00,   # Zxx_up (curvature at max upper camber)
    0.10,   # X_low (position of max lower camber)
   -0.10,   # Z_low (max lower camber)
    0.30,   # Zxx_low (curvature at max lower camber)
    0.000,  # alpha_te (trailing edge angle)
   -0.020,  # beta_te (trailing edge offset)
    0.000,  # h_te (trailing edge thickness)
   -0.30    # dz_te (trailing edge camber)
])

upper_bounds = np.array([
    0.040,  # r_le
    0.70,   # X_up
    0.15,   # Z_up
    0.10,   # Zxx_up
    0.95,   # X_low
   -0.001,  # Z_low
    3.00,   # Zxx_low
    0.020,  # alpha_te
    0.020,  # beta_te
    0.120,  # h_te
   -0.001   # dz_te
])

#Super wide bounds
lower_bounds = np.array([0.001, 0.00, -0.05, -2.00, 0.00, -0.20, -2.00, -0.020, -0.050, 0.000, -0.50])
upper_bounds = np.array([0.060, 1.00,  0.20,  2.00, 1.00,  0.05,  5.00,  0.050,  0.050, 0.200,  0.050])


norm_lower = np.zeros_like(lower_bounds)
norm_upper = np.ones_like(upper_bounds)
bounds = np.array([norm_lower, norm_upper])
v1, v2 = 70, 300
AoA = 5
w1 = 0.5
w2 = 1 - w1

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
re_1 = calc_re(v1)
mach_1 = calc_mach(v1)
re_2 = calc_re(v2)
mach_2 = calc_mach(v2)
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)
def denormalize_params(params):
    return lower_bounds + params * (upper_bounds - lower_bounds)
def read_output(output_file):
    with open(output_file, "r") as f:
        lines = f.readlines()
        line = lines[-1].split()
        lift, drag = float(line[1]), float(line[2])
        if drag == 0:
            return bad_result
        ratio = round(lift / drag, 7)
    return [lift, drag, ratio]
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
    try:
        result = read_output(output_file)
    except:
        result = bad_result
    try:
        os.remove(input_file)
        os.remove(output_file)
    except Exception as e:
        pass
        #print(f"file cleanup issue, {e}")
    return result
def generate_airfoil(params, filename='airfoil.dat', plot_airfoil=False):

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

    common_x = np.linspace(0.001, 0.999, 100)

    from scipy.interpolate import interp1d

    upper_interp = interp1d(x_upper, z_upper)
    lower_interp = interp1d(x_lower, z_lower)

    z_upper_interp = upper_interp(common_x)
    z_lower_interp = lower_interp(common_x)


    min_thickness = 0.001
    thickness = z_upper_interp - z_lower_interp

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
        denorm_params = denormalize_params(parameters[i])
        pid = os.getpid()
        timestamp = int(time.time() * 1000) % 10000
        filename = f'temp_airfoil_{pid}_{timestamp}.dat'
        ran_1 = random.randint(0, 9999)
        output_file_1 = f'parsec_output_{pid}_{timestamp}_{ran_1}.txt'
        input_file_1 = f'parsec_input_{pid}_{timestamp}_{ran_1}.txt'
        pid = os.getpid()
        timestamp = int(time.time() * 1000) % 10000
        ran_2 = random.randint(0, 9999)
        output_file_2 = f'parsec_output_{pid}_{timestamp}_{ran_2}.txt'
        input_file_2 = f'parsec_input_{pid}_{timestamp}_{ran_2}.txt'

        try:
            _ = generate_airfoil(denorm_params, filename, plot_airfoil=False)
            if _ == True:
                result_1 = run_xfoil(load_file=filename, input_file=input_file_1, output_file=output_file_1, RE=re_1, MACH=mach_1, AoA=AoA)
                result_2 = run_xfoil(load_file=filename, input_file=input_file_2, output_file=output_file_2, RE=re_2, MACH=mach_2, AoA=AoA)
                l_1, d_1, ratio_1 = result_1
                l_2, d_2, ratio_2 = result_2
                l_1 = normalize(l_1, 0.5, 3)
                d_2 = normalize(d_2, 0, 0.2)
                result = - (w1 * l_1 - w2 * d_2)
                results[i] = result
            else:
                results[i] = 1e6
        except Exception as e:
            print(f"Error with params {parameters[i]}: {e}")
            results[i] = 1e6
        finally:
            for f in [filename, input_file_1, input_file_2, output_file_1, output_file_2]:
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
        'c1': 2, 'c2': 1.5, 'w': 0.8,
        'k': n_particles//4, 'p': 2,
        'velocity_clamp': (-0.1, 0.1)
    }


    optimizer = LocalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds, init_pos=scaled)
    best_cost, best_position = optimizer.optimize(objective_PARSEC, iters=iters, verbose=True, n_processes=4)
    best_position = denormalize_params(best_position)
    print(f"Optimal Solution: {best_position}")
    print(f"Minimum Cost: {best_cost}")
    generate_airfoil(best_position, 'optimized_airfoil.dat', plot_airfoil=True)
    print("Optimized airfoil saved as 'optimized_airfoil.dat' and plotted.")

    filename = "optimized_airfoil.dat"
    pid = os.getpid()
    timestamp = int(time.time() * 1000) % 10000
    ran_1 = random.randint(0, 9999)
    output_file_1 = f'parsec_output_{pid}_{timestamp}_{ran_1}.txt'
    input_file_1 = f'parsec_input_{pid}_{timestamp}_{ran_1}.txt'
    pid = os.getpid()
    timestamp = int(time.time() * 1000) % 10000
    ran_2 = random.randint(0, 9999)
    output_file_2 = f'parsec_output_{pid}_{timestamp}_{ran_2}.txt'
    input_file_2 = f'parsec_input_{pid}_{timestamp}_{ran_2}.txt'

    result_1 = run_xfoil(load_file=filename, input_file=input_file_1, output_file=output_file_1, RE=re_1, MACH=mach_1,
                         AoA=AoA)
    result_2 = run_xfoil(load_file=filename, input_file=input_file_2, output_file=output_file_2, RE=re_2, MACH=mach_2,
                         AoA=AoA)
    print(re_1, mach_1)
    print(re_2, mach_2)
    print(result_1)
    print(result_2)
    history = np.negative(optimizer.cost_history)
    plt.plot(history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Convergence Plot')
    plt.show()

def main():
    start = perf_counter()
    pso_PARSEC(100, 100)
    end = perf_counter()
    period = end - start
    print(f"Execution finished in {round(period / 60, 2)} minutes")

if __name__ == "__main__":
    main()

