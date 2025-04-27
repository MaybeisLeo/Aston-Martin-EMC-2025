import os
import subprocess
import numpy as np
import time
from scipy.optimize import differential_evolution
from multiprocessing import Pool,cpu_count
start=time.time()
alpha=5
directory=os.getcwd()
w1, w2 = 0.5, 0.5
Niter=500
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
re1=calc_re(70)
re2=calc_re(300)
mach1=calc_mach(70)
mach2=calc_mach(300)

def Run_parsec(name,re,mach,verbose=False):
    input_file = (f"{name}input.txt")
    output_file = (f"{name}output.txt")
    
    with open(input_file, "w") as file:
        file.write(f"LOAD {name}.dat\nPANE\nOPER\nITER 500\nVISC {re}\nMACH {mach}\n")
        file.write(f"PACC\n{output_file}\n\nALFA {alpha}\nquit\n")

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        result = subprocess.run(
            ["C:/Users/liuwi/OneDrive/Documents/XFOIL6.99/no ai/xfoil.exe"],
            input=open(input_file, "r").read(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=directory,
            timeout=2,
            startupinfo=startupinfo
        )
        if "VISCAL:  Convergence failed" in result.stdout:
            if verbose:
                print(f"{name}: Convergence failed")
            return 0, 1
        with open(output_file, "r") as file:
            lines = file.readlines()
            if len(lines) < 13:
                if verbose:
                    print(f"{name}: Incomplete output")
                return 0, 1
            last = lines[12].split()  # Polar data line
            cl, cd = float(last[1]), float(last[2])
        return cl, cd
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"{name}: TIMEOUT")
        return 0, 1
    except FileNotFoundError:
        if verbose:
            print(f"{name}: XFOIL or file not found")
        return 0, 1
    except Exception as e:
        if verbose:
            print(f"{name}: Error - {e}")
        return 0, 1

def generate_airfoil(params, filename='best.dat', verbose=True, visualize=True):
    r_le, X_up, Z_up, Z_xx_up, X_lo, Z_lo, Z_xx_lo, Z_te, Delta_Z_te, alpha_te, beta_te = params
    
    a1 = np.sqrt(2 * r_le)
    b1 = -np.sqrt(2 * r_le)
    exponents = [1.5, 2.5, 3.5, 4.5, 5.5]
    te_z_upper = Delta_Z_te + Z_te / 2
    te_z_lower = Delta_Z_te - Z_te / 2

    A_up = np.zeros((5, 5))
    A_up[0, :] = [X_up ** exp for exp in exponents]
    A_up[1, :] = [exp * X_up ** (exp - 1) for exp in exponents]
    A_up[2, :] = [exp * (exp - 1) * X_up ** (exp - 2) for exp in exponents]
    A_up[3, :] = [1.0 ** exp for exp in exponents]
    A_up[4, :] = [exp * 1.0 ** (exp - 1) for exp in exponents]
    b_up = np.array([
        Z_up - a1 * X_up ** 0.5,
        -0.5 * a1 / np.sqrt(X_up),
        Z_xx_up + 0.25 * a1 / (X_up ** 1.5),
        te_z_upper - a1,
        np.tan(alpha_te) - 0.5 * a1
    ])
    a_coeffs = np.linalg.solve(A_up, b_up)
    upper_coeffs = [a1] + list(a_coeffs)

    A_lo = np.zeros((5, 5))
    A_lo[0, :] = [X_lo ** exp for exp in exponents]
    A_lo[1, :] = [exp * X_lo ** (exp - 1) for exp in exponents]
    A_lo[2, :] = [exp * (exp - 1) * X_lo ** (exp - 2) for exp in exponents]
    A_lo[3, :] = [1.0 ** exp for exp in exponents]
    A_lo[4, :] = [exp * 1.0 ** (exp - 1) for exp in exponents]
    b_lo = np.array([
        Z_lo - b1 * X_lo ** 0.5,
        -0.5 * b1 / np.sqrt(X_lo),
        Z_xx_lo + 0.25 * b1 / (X_lo ** 1.5),
        te_z_lower - b1,
        np.tan(beta_te) - 0.5 * b1
    ])
    b_coeffs = np.linalg.solve(A_lo, b_lo)
    lower_coeffs = [b1] + list(b_coeffs)

    def compute_z(x, coeffs, exps=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]):
        return sum(c * x ** e for c, e in zip(coeffs, exps))

    N = 100
    beta = np.linspace(0, np.pi, N)
    x_dist = 0.5 * (1 - np.cos(beta))
    x_upper = np.flip(x_dist)
    x_lower = x_dist
    z_upper = np.array([compute_z(xi, upper_coeffs) for xi in x_upper])
    z_lower = np.array([compute_z(xi, lower_coeffs) for xi in x_lower])

    x_upper[0], z_upper[0] = 1.0, te_z_upper
    x_lower[-1], z_lower[-1] = 1.0, te_z_lower
    x_upper[-1], x_lower[0] = 0.0, 0.0
    le_z = (z_upper[-1] + z_lower[0]) / 2
    z_upper[-1] = z_lower[0] = le_z

    x_coords = np.concatenate([x_upper, x_lower[1:]])
    z_coords = np.concatenate([z_upper, z_lower[1:]])

    with open(filename, 'w') as f:
        f.write("PARSEC Optimized Airfoil\n")
        for x, z in zip(x_coords, z_coords):
            f.write(f"{x:.6f} {z:.6f}\n")

    return x_coords, z_coords

def objective_PARSEC(params, verbose=True):
    pid = os.getpid()
    timestamp = int(time.time() * 1000) % 10000
    base_name = f"parsec_{pid}_{timestamp}"
    dat_file = os.path.join(directory, f"{base_name}.dat")
    try:
        coords = generate_airfoil(params, dat_file, verbose=verbose)
        if coords[0] is None: 
            return float('inf')
        if True:
            cl1, cd1 = Run_parsec(base_name,re1,mach1,verbose=verbose)
            cl2, cd2 = Run_parsec(base_name,re2,mach2,verbose=verbose)
            score = (w1 * cl1 - w2 * cd2) 
        if score == 0:
            return float('inf')
        if verbose:
            print(f"{base_name}: Score = {score:.4f}")
        return -score
    except Exception as e:
        if verbose:
            print(f"Error with params {params}: {e}")
        return float('inf')
    finally:
        for f in [dat_file, os.path.join(directory, f"{base_name}input.txt"), os.path.join(directory, f"{base_name}output.txt")]:
            if os.path.exists(f):
                os.remove(f)

def evo_PARSEC(pop, maxITER, verbose=True):
    # lower_bounds = np.array([0.003, 0.25, 0.04, -0.5, 0.4, -0.05, 0.6, 0.002, 0.000, 0.02, -0.1])
    # upper_bounds = np.array([0.02, 0.55, 0.12, -0.05, 0.75, -0.005, 2.0, 0.015, 0.01, 0.08, -0.02])
    lower_bounds = np.array([0.001,0.1,0.02,-1.0, 0.1,-0.1,0.3,0.000,-0.01, 0.00,-0.3])
    upper_bounds = np.array([0.04,0.7,0.15,0.1,0.95,-0.001,3.0,0.02,0.02,0.12,-0.001])
    bounds_PARSEC = [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
    
    with Pool(processes=cpu_count(), maxtasksperchild=30) as pool:  
        result = differential_evolution(
            objective_PARSEC,
            bounds_PARSEC,
            strategy='randtobest1exp',
            maxiter=maxITER,
            popsize=pop,
            disp=verbose,
            tol=0.0001,
            workers=-1
        ) 
    if verbose:
        print("\nOptimization Results:")
        print(f"Optimal parameters: {result.x}")
        print(f"Maximum Score: {-result.fun:.4f}")    
 
    final_dat =f"DE_multi_object{(-result.fun)}.dat"
    generate_airfoil(result.x, final_dat, verbose=verbose)
    if verbose:
        print(f"Optimized airfoil saved as '{final_dat}'")
        print(f"Total time: {time.time() - start:.2f}s")

    return result.x, -result.fun


if __name__=="__main__":
    print(f"Renold at 70km/h : {re1}")
    print(f"Renold at 300km/h : {re2}")
    print(f"Mach at 70km/h : {mach1}")
    print(f"Mach at 300km/h : {mach2}")
    pop=int(input("The population size"))
    maxITER=int(input("The iteration size"))
    best_par,best_if=evo_PARSEC(pop,maxITER,verbose=True)
    print(time.time()-start)
    
