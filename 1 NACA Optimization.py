import csv
from multiprocessing import Pool, cpu_count
import subprocess
import os
from time import perf_counter
import time
from scipy.optimize import differential_evolution
import pyswarms as ps
import matplotlib.pyplot as plt
import numpy as np

#Variables
base_dir = "C:/Users/kgain/Desktop/Aston Martin EMC"
xfoil_path = os.path.join(base_dir, "xfoil.exe")
airfoil_file = os.path.join(base_dir, "airfoil_temp.dat")
results_file = os.path.join(base_dir, "results.csv")
cache = {}
focus = 2  # Ratio 0 is Lift, 1 is drag
velocity = 70  # km/h
AoA = 5

#Level 1 functions
def calc_re(velocity):
    v = 1000 * float(velocity) / 3600  # km/h to m/s
    c = 1  # Chord length (m)
    p = 1.225  # Air density (kg/m^3)
    u = 1.81e-5  # Dynamic viscosity (kg/m·s)
    return round(p * v * c / u)
def calc_mach(velocity):
    v = 1000 * float(velocity) / 3600  # km/h to m/s
    a = 343  # Speed of sound (m/s)
    return round(v / a, 4)
RE = calc_re(velocity)
MACH = calc_mach(velocity)
def generate_commands(input_file, NACA, ITER, RE, MACH, output_file):
    with open(input_file, "w") as f:
        f.write(f"NACA {NACA}\n")
        f.write("OPER\n")
        f.write(f"ITER {ITER}\n")
        f.write(f"VISC {RE}\n")
        f.write(f"MACH {MACH}\n")
        f.write(f"PACC\n{output_file}\n\n")
        f.write(f"A {AoA}\n")
        f.write("QUIT\n")
def generate_naca_codes(lbA=0, ubA=10, lbB=0, ubB=10, lbC=0, ubC=10, lbD=0, ubD=10):
    return [f"{a}{b}{c}{d}" for a in range(lbA,ubA) for b in range(lbB,ubB) for c in range(lbC,ubC) for d in range(lbD,ubD) if f"{c}{d}" != "00"]
def read_output(output_file):
    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
            line = lines[-1].split()
            lift, drag = float(line[1]), float(line[2])
            if drag == 0:  # Prevent divide by zero
                return [-1e6, 1e6, -1e6]
            ratio = round(lift / drag, 3)
        return [lift, drag, ratio]
    except:
        return [-1e6, 1e6, -1e6]

#Level 2 functions
def store_all_results(NACA_LIST, LIFT_COFFS, DRAG_COFFS, RATIOS):
    valid_lifts = [x for x in LIFT_COFFS if x is not None]
    valid_drags = [x for x in DRAG_COFFS if x is not None]
    valid_ratios = [x for x in RATIOS if x is not None]
    if not valid_lifts:
        print("No valid data to store.")
        return

    max_lift, min_lift = max(valid_lifts), min(valid_lifts)
    max_drag, min_drag = max(valid_drags), min(valid_drags)
    max_lift_drag, min_lift_drag = max(valid_ratios), min(valid_ratios)

    NACA_MAX_LIFT = NACA_LIST[LIFT_COFFS.index(max_lift)]
    NACA_MIN_LIFT = NACA_LIST[LIFT_COFFS.index(min_lift)]
    NACA_MAX_DRAG = NACA_LIST[DRAG_COFFS.index(max_drag)]
    NACA_MIN_DRAG = NACA_LIST[DRAG_COFFS.index(min_drag)]
    NACA_LIFT_DRAG = NACA_LIST[RATIOS.index(max_lift_drag)]

    with open(results_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Airfoil", "Lift", "Drag", "Drag(P)", "Lift/Drag"])
        for i, naca in enumerate(NACA_LIST):
            csv_writer.writerow([naca, LIFT_COFFS[i], DRAG_COFFS[i], RATIOS[i]])
        csv_writer.writerow([])
        csv_writer.writerow(["Summary Statistics"])
        csv_writer.writerow(["Max Lift", max_lift, "Airfoil", NACA_MAX_LIFT])
        csv_writer.writerow(["Min Lift", min_lift, "Airfoil", NACA_MIN_LIFT])
        csv_writer.writerow(["Max Drag", max_drag, "Airfoil", NACA_MAX_DRAG])
        csv_writer.writerow(["Min Drag", min_drag, "Airfoil", NACA_MIN_DRAG])
        csv_writer.writerow(["Max Lift/Drag", max_lift_drag, "Airfoil", NACA_LIFT_DRAG]) #
def run_xfoil(NACA=None, load_file=None, input_file=None, output_file=None, RE=None, MACH=None, AoA=None):
    bad_result = [-1e6, 1e6, -1e6]
    if load_file:
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
            f.write(f"PACC")
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
    else:
        global cache
        a, b, c, d = map(int, NACA)
        NACA = f"{a}{b}{c}{d}"
        if NACA[-2:] == "00":
            return [NACA, -1e6, 1e6, -1e6]
        if NACA in cache:
            return cache[NACA]
        output_file = os.path.join(base_dir, f"xfoil_output_{NACA}.txt")
        input_file = os.path.join(base_dir, f"xfoil_input_{NACA}.txt")
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        converged = True
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(input_file):
            os.remove(input_file)

        for ITER in range(100, 501, 400):
            generate_commands(input_file, NACA, ITER, RE, MACH, output_file)
            try:
                result = subprocess.run(
                    [xfoil_path], stdin=open(input_file, "r"), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, cwd=base_dir, timeout=2, startupinfo=startupinfo
                )
                if "VISCAL: Convergence failed" in result.stdout:
                    converged = False
                    continue
                else:
                    converged = True
                    break
            except subprocess.TimeoutExpired:
                converged = False
                break
        if not converged:
            return [NACA, -1e6, 1e6, -1e6]

        result = read_output(output_file)
        cache[NACA] = result
        result = [NACA] + result
        os.remove(input_file)
        os.remove(output_file)
        return result
def brute_NACA():
    start = perf_counter()
    naca_codes = generate_naca_codes(ubA=1, ubB=1,ubC=10, ubD=10)  #Limit
    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.map(run_xfoil, naca_codes)
    valid_results = [r for r in results if r[1] != -1e6]
    if not valid_results:
        print("No valid results obtained. Check your parameters.")
        return

    NACA_LIST, LIFT_COFFS, DRAG_COFFS, RATIOS = [], [], [], []
    for group in valid_results:
        NACA_LIST.append(group[0])
        LIFT_COFFS.append(group[1])
        DRAG_COFFS.append(group[2])
        RATIOS.append(group[3])

    store_all_results(NACA_LIST, LIFT_COFFS, DRAG_COFFS, RATIOS)
    end = perf_counter()
    period = end - start
    print(f"Execution finished in {round(period, 2)} seconds ({round(period / 60, 2)} minutes)")
def run_xfoil_pso(NACA):
    # If NACA is a 2D array (from PSO), process each row
    if isinstance(NACA, np.ndarray) and NACA.ndim == 2:
        results = []
        for naca in NACA:
            a, b, c, d = round(naca[0]), round(naca[1]), round(naca[2]), round(naca[3])
            naca_code = f"{a}{b}{c}{d}"
            result = run_xfoil(naca_code)
            results.append(result)
        return np.array(results)
    else:
        # If NACA is a 1D array (from PSO), process it directly
        a, b, c, d = int(NACA[0]), int(NACA[1]), int(NACA[2]), int(NACA[3])
        naca_code = f"{a}{b}{c}{d}"
        return run_xfoil(naca_code)
def objective_NACA(NACA):
    return -run_xfoil_pso(NACA)[3]
def evo_NACA():
    start = perf_counter()
    bounds = [(0,9),(0,9),(0,9),(0,9)]
    result = differential_evolution(objective_NACA, bounds, integrality=[True, True, True, True], maxiter=50, popsize=20, disp=True, strategy="best2bin", workers=-1)
    end = perf_counter()
    print(result)
    period = end - start
    print(f"Execution finished in {round(period, 2)} seconds ({round(period / 60, 2)} minutes)")
def pso_NACA():
    start = perf_counter()
    bounds = (np.array([0, 0, 0, 0]), np.array([9, 9, 9, 9]))
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # Cognitive, social, and inertia weights
    dimensions = 4
    optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=dimensions, options=options, bounds=bounds)
    best_cost, best_position = optimizer.optimize(objective_NACA, iters=20, verbose=True)
    print(f"Optimal Solution: {best_position}")
    print(f"Minimum Cost: {best_cost}")
    end = perf_counter()
    period = end - start
    print(f"Execution finished in {round(period, 2)} seconds ({round(period / 60, 2)} minutes)")
    plt.plot(optimizer.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Convergence Plot')
    plt.show()

def main():
    start = perf_counter()
    brute_NACA()
    end = perf_counter()
    period = end - start
    print(f"Execution finished in {round(period / 60, 2)} minutes)")

if __name__ == "__main__":
    main()