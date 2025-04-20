from multiprocessing import Pool
import multiprocessing
import os
import subprocess
import numpy as np
import csv
import time
from scipy.optimize import differential_evolution
from pyswarm import pso
from scipy.interpolate import interp1d
from multiprocessing import Pool,cpu_count
import matplotlib.pyplot as plt
start=time.time()
maxiter=30
population=55
m_range= range(4, 6)  # Maximum camber
p_range= range(0, 6)  # Location of the camber
t_range= range(0, 3) 
alpha=5
naca=[]
L=[]
D=[]
R=[]
Dp=[]
directory="C:/Users/liuwi/OneDrive/Documents/XFOIL6.99/no ai"
store=[]
Niter=500
re=1e5
bond=[(0.2,0.9),(0.2,0.9),(0.01,0.70)]#For DE in NACA

# Performs PSO optimization for PARSEC airfoil parameters with fixed bounds and visualizes results
def swarm_PARSEC(pop,maxITER,verbose=True):
    print("Starting PSO optimization for PARSEC airfoil parameters...")
    start_time = time.time()


    lower_bounds = np.array([0.005, 0.3, 0.05, -0.4, 0.5, -0.04, 1.0, 0.005, 0.0, 0.02, -0.08])
    upper_bounds = np.array([0.02, 0.5, 0.12, -0.1, 0.8, -0.01, 2.0, 0.015, 0.01, 0.08, -0.02])

    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    
    # Configure and run PSO
    best_x, best_f = pso(
        objective_PARSEC,
        lower_bounds,
        upper_bounds,
        swarmsize=pop,  # Number of particles
        maxiter=maxITER,    # Maximum iterations
        debug=verbose,
        omega=0.5,     # Inertia weight
        phip=1.5,      # Personal best weight
        phig=1.5       # Global best weight
    )
    
    if verbose:
        print("\nPSO Optimization Results:")
        print(f"Optimal parameters: {best_x}")
        print(f"Maximum L/D: {-best_f:.4f}")
    
    final_dat = 'pso_optimized_airfoil.dat'
    generate_airfoil(best_x, final_dat, verbose=verbose)
    
    # Rename the file to include the L/D ratio
    os.rename(final_dat,f"swarm_{round(-best_f)}.dat")
    
    # Visualize the optimized airfoil
    plot_airfoil_from_dat(os.path.join(directory, f"swarm_{round(-best_f)}.dat"))
    
    if verbose:
        print(f"Optimized airfoil saved as 'pso_{round(-best_f)}.dat'")
        print(f"Total optimization time: {time.time() - start_time:.2f}s")
        

    return best_x, -best_f

# Visualizes an airfoil profile from a .dat file with detailed insets for leading and trailing edges
def plot_airfoil_from_dat(dat_file, save_path=None, show=True):
    """
    Plot an airfoil from a .dat file and optionally save it as a PNG.
    
    Parameters:
    dat_file (str): Path to the .dat file
    save_path (str): Path to save the PNG file (if None, won't save)
    show (bool): Whether to display the plot
    """
    # Read the .dat file
    with open(dat_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    data = [line.strip().split() for line in lines[1:]]
    x_coords = [float(point[0]) for point in data]
    z_coords = [float(point[1]) for point in data]
    
    # Find the trailing edge index (closest to x=1)
    te_index = x_coords.index(max(x_coords))
    
    # Split into upper and lower surfaces
    x_upper = x_coords[:te_index+1]
    z_upper = z_coords[:te_index+1]
    x_lower = x_coords[te_index:]
    z_lower = z_coords[te_index:]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_upper, z_upper, 'b-', label='Upper Surface')
    ax.plot(x_lower, z_lower, 'r-', label='Lower Surface')
    
    # Add leading and trailing edge markers
    ax.scatter([x_coords[0]], [z_coords[0]], color='m', marker='o', label="Leading Edge")
    ax.scatter([x_coords[te_index]], [z_coords[te_index]], color='g', marker='o', label="Trailing Edge")
    
    # Add title and labels
    airfoil_name = os.path.basename(dat_file).replace('.dat', '')
    ax.set_title(f'Airfoil: {airfoil_name}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('z/c')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    # Add inset for trailing edge closeup
    inset_ax = fig.add_axes([0.65, 0.15, 0.2, 0.2])
    inset_ax.plot(x_upper[-10:], z_upper[-10:], 'b-')
    inset_ax.plot(x_lower[:10], z_lower[:10], 'r-')
    inset_ax.set_xlim(0.95, 1.01)
    inset_ax.set_ylim(min(z_lower[:10] + z_upper[-10:]) - 0.001, 
                      max(z_lower[:10] + z_upper[-10:]) + 0.001)
    inset_ax.grid(True)
    inset_ax.set_title('TE Closeup')
    
    # Add inset for leading edge closeup
    le_inset_ax = fig.add_axes([0.15, 0.15, 0.2, 0.2])
    le_inset_ax.plot(x_upper[:10], z_upper[:10], 'b-')
    le_inset_ax.plot(x_lower[-10:], z_lower[-10:], 'r-')
    le_inset_ax.set_xlim(-0.01, 0.05)
    le_inset_ax.set_ylim(min(z_lower[-10:] + z_upper[:10]) - 0.001,
                         max(z_lower[-10:] + z_upper[:10]) + 0.001)
    le_inset_ax.grid(True)
    le_inset_ax.set_title('LE Closeup')
    
    plt.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

# Runs XFOIL simulation on a PARSEC airfoil with the given name and returns lift and drag coefficients
def Run_parsec(name,verbose=False):
    input_file = (f"{name}input.txt")
    output_file = (f"{name}output.txt")
    
    with open(input_file, "w") as file:
        file.write(f"LOAD {name}.dat\nPANE\nOPER\nITER 500\nVISC {re}\n")
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
        if verbose:
            print(f"{name}: CL={cl:.4f}, CD={cd:.4f}, L/D={cl/cd:.4f}")
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

# Creates an airfoil using PARSEC parameters and saves to a .dat file, checking for validity
def generate_airfoil(params, filename='best.dat', verbose=True, visualize=True):
    r_le, X_up, Z_up, Z_xx_up, X_lo, Z_lo, Z_xx_lo, Z_te, Delta_Z_te, alpha_te, beta_te = params
    
    a1 = np.sqrt(2 * r_le)
    b1 = -np.sqrt(2 * r_le)
    exponents = [1.5, 2.5, 3.5, 4.5, 5.5]
    te_z_upper = Delta_Z_te + Z_te / 2
    te_z_lower = Delta_Z_te - Z_te / 2

    # Upper surface
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

    # Lower surface
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

    # Check for self-intersection
    has_intersection = check_self_intersection(x_upper, z_upper, x_lower, z_lower)
    
    if has_intersection:
        # if verbose:
        #     print(f"Airfoil has self-intersection - not writing file {filename}")
        return None, None  # Return None to indicate failure
    
    # Only write file if no intersection is found
    with open(filename, 'w') as f:
        f.write("PARSEC Optimized Airfoil\n")
        for x, z in zip(x_coords, z_coords):
            f.write(f"{x:.6f} {z:.6f}\n")

    return x_coords, z_coords

# Checks if an airfoil design has self-intersections or unrealistic properties
def check_self_intersection(x_upper, z_upper, x_lower, z_lower):
    """
    Check if the airfoil has any self-intersections by comparing
    the upper and lower surfaces.
    
    Returns True if intersection is found, False otherwise.
    """
    # Create interpolation functions for the upper and lower surfaces
    f_upper = interp1d(x_upper, z_upper, kind='cubic', bounds_error=False, fill_value="extrapolate")
    f_lower = interp1d(x_lower, z_lower, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    # Define a set of x coordinates for checking
    x_check = np.linspace(0.001, 0.999, 200)  # Avoid endpoints due to potential numerical issues
    
    # Check for intersection
    for x in x_check:
        if f_upper(x) <= f_lower(x):  # Upper surface should always be above lower surface
            return True
    
    # Check for unrealistic thickness
    max_thickness = np.max(f_upper(x_check) - f_lower(x_check))
    if max_thickness > 0.5:  # 50% thickness is generally unrealistic
        return True
    
    # Check for sharp corners or unrealistic curvature
    upper_diffs = np.diff(z_upper) / np.diff(x_upper)
    lower_diffs = np.diff(z_lower) / np.diff(x_lower)
    
    # Check for extreme gradients
    if np.max(np.abs(upper_diffs)) > 10 or np.max(np.abs(lower_diffs)) > 10:
        return True
    
    return False

# Objective function for PARSEC optimization - evaluates aerodynamic performance through XFOIL
def objective_PARSEC(params, verbose=True):
    pid = os.getpid()
    timestamp = int(time.time() * 1000) % 10000
    base_name = f"parsec_{pid}_{timestamp}"
    dat_file = os.path.join(directory, f"{base_name}.dat")
    try:
        coords = generate_airfoil(params, dat_file, verbose=verbose)
        if coords[0] is None:  # Check if airfoil generation failed due to intersection
            return float('inf')
        
        cl, cd = Run_parsec(base_name, verbose=verbose)
        if cl == 0:
            return float('inf')
        ratio = cl / cd
        if verbose:
            print(f"{base_name}: L/D = {ratio:.4f}")
        
        return -ratio
    except Exception as e:
        if verbose:
            print(f"Error with params {params}: {e}")
        return float('inf')
    finally:
        for f in [dat_file, os.path.join(directory, f"{base_name}input.txt"), os.path.join(directory, f"{base_name}output.txt")]:
            if os.path.exists(f):
                os.remove(f)

# Performs differential evolution optimization for PARSEC airfoil parameters with convergence tracking
def evo_PARSEC(pop, maxITER, verbose=True):
    # Create lists to store convergence history
    convergence_history = []
    best_ld_ratios = []
    iteration_count = []
    
    # Function to record iteration data for plotting
    def callback_DE(xk, convergence=None):
        # Calculate current L/D ratio (negative because we're minimizing)
        current_ld = -objective_PARSEC(xk, verbose=False)
        best_ld_ratios.append(current_ld)
        iteration_count.append(len(best_ld_ratios))
        if verbose and len(best_ld_ratios) % 5 == 0:  # Print every 5 iterations to avoid cluttering
            print(f"Iteration {len(best_ld_ratios)}: Best L/D = {current_ld:.4f}")
        return False  # Continue optimization
    
    lower_bounds = np.array([0.003, 0.25, 0.04, -0.5, 0.4, -0.05, 0.6, 0.002, 0.000, 0.02, -0.1])
    upper_bounds = np.array([0.02, 0.55, 0.12, -0.05, 0.75, -0.005, 2.0, 0.015, 0.01, 0.08, -0.02])
    bounds_PARSEC = [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
    
    with Pool(processes=cpu_count(), maxtasksperchild=30) as pool:  # Explicit Pool control
        result = differential_evolution(
            objective_PARSEC,
            bounds_PARSEC,
            strategy='randtobest1exp',
            maxiter=maxITER,
            popsize=pop,
            disp=verbose,
            tol=0.0001,
            workers=-1,  # Custom worker mapping
            callback=callback_DE  # Add callback function to track convergence
        )
    
    if verbose:
        print("\nOptimization Results:")
        print(f"Optimal parameters: {result.x}")
        print(f"Maximum L/D: {-result.fun:.4f}")
    
    # Create and save the convergence plot with the desired style
    plt.figure(figsize=(8, 6))
    
    # Use a step plot with a distinct blue line
    plt.step(iteration_count, best_ld_ratios, where='post', color='#1f77b4', linewidth=2)
    
    # Configure the plot appearance
    plt.title('Convergence Plot', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost', fontsize=12)  # Changed from 'L/D Ratio' to 'Cost' as shown in example
    
    # Set clean grid lines

    # Configure axis limits - allow some padding
    plt.xlim(0, max(iteration_count) + 5)
    y_min = min(best_ld_ratios) * 0.9
    y_max = max(best_ld_ratios) * 1.1
    plt.ylim(y_min, y_max)
    
    # Style the plot
    plt.tight_layout()
    
    # Save the convergence plot
    plt.savefig('parsec_convergence.png', dpi=300, bbox_inches='tight')
    if verbose:
        print("Convergence plot saved as 'parsec_convergence.png'")
        plt.show()
    
    final_dat = os.path.join(directory, 'optimized_airfoil.dat')
    generate_airfoil(result.x, final_dat, verbose=verbose)
    
    if verbose:
        print(f"Optimized airfoil saved as '{final_dat}'")
        print(f"Total time: {time.time() - start:.2f}s")
    
    plot_airfoil_from_dat("optimized_airfoil.dat")
    os.rename("optimized_airfoil.dat", f"{round(-result.fun)}_aerofoil.dat")
    plot_airfoil_from_dat(f"{round(-result.fun)}_aerofoil.dat")
    
    return result.x, -result.fun

# Runs XFOIL simulation on a NACA airfoil with the given code and returns lift and drag coefficients
def Run_Xfoil(naca):
    smallNiter=300
    input=f"diff{naca}in.txt"
    out=f"diff{naca}out.txt"
    with open(out,"w"):
        pass
    with open(input, "w") as file:
        file.write(f"NACA {naca}\n")
        file.write(f"OPER\n")
        file.write(f"ITER {smallNiter}\n")
        file.write(f"visc {re}\n")
        file.write(f"PACC\n{out}\nn\n \n")
        file.write(f"ALFA {alpha}")
        file.write(f"\n")
        file.write(f"PWRT\n{out}\nY\n\n")
        file.write(f"quit\n")
        file.close()
    
    # Run XFOIL using 
    # Run XFOIL properly
    starupinfo=subprocess.STARTUPINFO()
    starupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result=subprocess.run(
            ["C:/Users/liuwi/OneDrive/Documents/XFOIL6.99/no ai/xfoil.exe"],
            stdin=open(input,"r"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=directory,
            timeout=10,
            startupinfo=starupinfo
            )
        if "VISCAL:  Convergence failed" in result.stdout:
            print(f"{naca} FAILED to converge")
            return 0,1
        with open(out, "r") as result:
            lines=result.readlines()
            last=lines[-1].split()
            cl,cd=float(last[1]),float(last[2])
        return  cl,cd      
    except subprocess.TimeoutExpired:
        print(naca,":TIME OUT ")
        return 0,1
    except FileNotFoundError:
        print("no file")
        return 0,1
    except Exception as error:
        print(f"Error:{error}")
        return 0,1

# Objective function for NACA differential evolution - evaluates airfoil performance for DE algorithm
def Diff_Object(x):
    m,p,t=x
    m,p,t=int(m*10),int(p*10),int(t*100)
    naca=f"{m}{p}{t:02d}"
    cl,cd=Run_Xfoil(naca)
    if cl==0:
        print(f"{naca} failed")
        return float("inf")
    print(f"{naca} passed")
    return -cl/cd

# Objective function for NACA PSO optimization - evaluates airfoil performance for PSO algorithm
def pso_Object(x):
    m,p,t=x
    m,p,t=int(m*10),int(p*10),int(t*100)
    naca=f"{m}{p}{t:02d}"
    cl,cd=Run_Xfoil(naca)
    if cl==0:
        print(f"{naca} failed")
        return 1e6
    print(f"{naca} passed")
    return -cl/cd

# Performs differential evolution optimization for NACA airfoil parameters
def diff():
        num_workers = multiprocessing.cpu_count() 
        if num_workers<1:
            num_workers=1
        result=differential_evolution(Diff_Object,bond,strategy="best1bin",popsize=25,mutation=(0.5,1),recombination=0.8,maxiter=15,disp=True,tol=0.001,workers=num_workers,updating="deferred")
        best_m,best_p,best_t=result.x
        naca=f"{int(best_m*10)}{int(best_p*10)}{int(best_t*100):02d}"
        print(naca)
        cl,cd=(Run_Xfoil(f"{int(best_m*10)}{int(best_p*10)}{int(best_t*100):02d}"))
        print(cl/cd)
        print(f"Final:{naca}")
        print(Run_Xfoil(naca))
        print(f"time:{time.time()-start}")

# Performs PSO optimization for NACA airfoil parameters until target ratio is achieved
def Swarm():
    Target_ratio=70
    while True:
        Lower_bond=[0.1,0.1,0.01]#for pso
        Upper_bond=[0.9,0.9,0.35]#for pso
        Lower_bond=np.array(Lower_bond)
        Upper_bond=np.array(Upper_bond)
        best_x,best_f=pso(pso_Object,Lower_bond,Upper_bond,swarmsize=20,maxiter=20,debug=True)
        if -best_f>Target_ratio:
            break
    naca=str(best_x).replace("[","").replace("]","").split(" ")
    m=naca[0][2]
    p=naca[1][2]
    t=naca[2][2:4]
    naca=m+p+t
    print(f"Best:{naca},best ratio:{-best_f}")
    print(f"data:{Run_Xfoil(naca)}")
    print(f"Executed {time.time()-start}s")
        
# Calculates Reynolds number based on airspeed, air density, chord length and viscosity
def find_Re(velocity=70): #velocity in km/h
    p=1.225
    V=velocity*1000/3600 #m/s
    L=1 #1 meter long chord
    u=1.83e-5 #dynamic viscosity of the air in 20 degrees and 200kPa
    return round((p*V*L)/u)


# Generates a list of NACA airfoils based on parameter ranges for camber, position, and thickness
def genNACA(m_range,p_range,t_range):
    naca=[]
    for m in m_range:
        for p in p_range:
            for t in t_range:
                naca.append(f"{m:01d}{p:01d}{t:02d}")
        try:
            naca.remove("0000")
        except:
            pass
    for output in naca:
        if not os.path.exists(f"{output}Output.txt"):
            with open(f"{output}Output.txt","w") as file:
                file.write(f"{output}")
    return naca

# Detailed analysis of an airfoil using XFOIL with handling for timeout and convergence failures
def anaylise(aerofoil):
    Niter=500
    with open(aerofoil+".txt","w") as file:
        file.write(f"NACA {aerofoil}\n")
        file.write("OPER\n")
        file.write("ITER {0}\n".format(Niter))
        file.write("visc {0}\n".format(re))
        file.write("PACC\n{0}\nn\n \n".format(f"{aerofoil}Output.txt"))
        file.write("ALFA {0}".format(alpha))
        file.write("\n")
        file.write("PWRT\n{0}\nY\n\n".format(f"{aerofoil}Output.txt"))
        file.write("quit\n")
        file.close()
    starupinfo=subprocess.STARTUPINFO()
    starupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result=subprocess.run(
            ["C:/Users/liuwi/OneDrive/Documents/XFOIL6.99/no ai/xfoil.exe"],
            stdin=open(aerofoil+".txt","r"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=directory,
            timeout=5,
            startupinfo=starupinfo
            )
        if "VISCAL:  Convergence failed" in result.stdout:
            return aerofoil, False
        print(aerofoil,":PASS",Niter)
        return aerofoil,True           
    except subprocess.TimeoutExpired:
        print(aerofoil,":TIME OUT ")
        return aerofoil,False
    except FileNotFoundError:
        print("no file")
        return aerofoil,False
    except Exception as erorr:
        print("Error:{error}")
        return aerofoil,False

#main program
if __name__=="__main__":
    choose=str(input("1.brute,2.diff"))
    if choose=="1":
        naca=genNACA(m_range,p_range,t_range)
        num_cores = multiprocessing.cpu_count()-1
        pool = multiprocessing.Pool(processes=num_cores)
        resultOfNaca=(pool.map(anaylise,naca))
        for this in list(resultOfNaca):
            if this != None :
                if this[1]==True:
                    store.append(this[0])
        store=sorted(store)
        print(store)
        for final in store:
            with open(f"{final}Output.txt","r") as read:
                data=read.readlines()
                row=data[12]
                column=row.split()
                L.append(float(column[1]))
                D.append(float(column[2]))
                R.append(float(float(column[1])/float(column[2])))
                Dp.append(float(column[3]))
        print(f"Max Lift is {max(L)} from {store[L.index(max(L))]}")
        print(f"Min Lift is {min(L)} from {store[L.index(min(L))]}")
        print(f"Max Drag is {max(D)} from {store[D.index(max(D))]}")
        print(f"Min Drag is {min(D)} from {store[D.index(min(D))]}")
        print(f"Max ratio is {max(R)} from {store[R.index(max(R))]}")
        print(f"Max Drag(p) is {max(Dp)} from {store[Dp.index(max(Dp))]}")
        print(f"Min Drag(p) is {min(Dp)} from {store[Dp.index(min(Dp))]}")
        print(f"executed {time.time()-start} seconds")
            


        #indata in csv
        CSV=input("CSV y/n")
        if CSV=="Y" or CSV=="y":
            with open('70kmh.csv','w',newline='') as Files:
                write=csv.writer(Files)
                field=['Naca','Lift','Drag','Drag(p)','Ratio']
                write.writerow(field)
                for row in range(len(store)):
                    write.writerow([store[row],L[row],D[row],Dp[row],R[row]])
            print("csv finish")
    elif choose=="2":
        diff()
    elif choose=="3":
        Swarm()
    elif choose=="4":
        best_par,best_if=evo_PARSEC(population,maxiter,verbose=True)
        print(best_par,best_if)
    elif choose=="5":
        plot_airfoil_from_dat("56_aerofoil.dat")
    elif choose=="6":
        best_parsec_pso_params, best_parsec_pso_ld = swarm_PARSEC(population,maxiter,verbose=True)
        print(f"PSO optimized PARSEC airfoil with L/D ratio: {best_parsec_pso_ld:.4f}")
        print(f"Parameters: {best_parsec_pso_params}")
        print(time.time()-start)