import os
import glob

def four():
    m_range= range(0, 10)  # Maximum camber
    p_range= range(0, 10)  # Location of the camber
    t_range= range(0, 100) 
    naca=[]
    for m in m_range:
        for p in p_range:
            for t in t_range:
                naca.append(f"{m:01d}{p:01d}{t:02d}")
    for i in naca:
        if os.path.exists(f"{i}.txt"):
            os.remove(f"{i}.txt")
        if os.path.exists(f"{i}Output.txt"):
            os.remove(f"{i}Output.txt")
        if os.path.exists(f"diff{i}in.txt"):
            os.remove(f"diff{i}in.txt")
        if os.path.exists(f"diff{i}out.txt"):
            os.remove(f"diff{i}out.txt")

def five():
    l_range=range(2,7)
    m_range= range(0, 10)  # Maximum camber
    p_range= range(0, 10)  # Location of the camber
    t_range= range(0, 100) 
    naca=[]
    for l in l_range:
        for m in m_range:
            for p in p_range:
                for t in t_range:
                    naca.append(f"{l:01d}{m:01d}{p:01d}{t:02d}")
    for i in naca:
        if os.path.exists(f"{i}.txt"):
            os.remove(f"{i}.txt")
        if os.path.exists(f"{i}Output.txt"):
            os.remove(f"{i}Output.txt")

def coord():
    for i in range(100):
        if os.path.exists(f"aero{i}input.dat"):
            os.remove(f"aero{i}input.dat")
        if os.path.exists(f"aero{i}commands.txt"):
            os.remove(f"aero{i}commands.txt")

def par():
    for i in range(100000,1000000):
        if os.path.exists(f"{i}.dat"):
            os.remove(f"{i}.dat")
        if os.path.exists(f"{i}input.txt"):
            os.remove(f"{i}input.txt")
        if os.path.exists(f"{i}Output.txt"):
            os.remove(f"{i}Output.txt")

def clean_parsec_de():
    """
    Clean up files related to PARSEC differential evolution optimization.
    These files typically have 'parsec' in their filenames with various number patterns.
    """
    
    # Get a list of all parsec-related files with various patterns
    parsec_patterns = [
        "parsec_*.*",           # General pattern for parsec files
        "parsec_*_*.*",         # Pattern like parsec_132_458
        "parsec_*_*_*.*",       # Pattern with more underscores
        "*parsec*input.txt",    # Input files
        "*parsec*output.txt",   # Output files
        "*parsec*.dat",         # Data files
        "*parsec*.png",         # PNG image files
        "optimized_airfoil.*"   # Optimized airfoil files
    ]
    
    parsec_files = []
    for pattern in parsec_patterns:
        parsec_files.extend(glob.glob(pattern))
    
    # Count the files found
    file_count = len(parsec_files)
    
    # Remove each file
    for file_path in parsec_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    print(f"Removed {file_count} PARSEC-related files")
def clean_parsec_pso():
    """
    Clean up files related to PARSEC PSO optimization.
    These files typically have 'pso' in their filenames with various number patterns.
    """
    # Get a list of all PSO-related files with various patterns
    pso_patterns = [
        "pso_*.*",              # Files starting with 'pso_'
        "*pso*input.txt",       # Input files
        "*pso*output.txt",      # Output files
        "pso_optimized_airfoil.*", # PSO optimized airfoil
        "pso_*.dat",            # PSO data files
    ]
    
    pso_files = []
    for pattern in pso_patterns:
        pso_files.extend(glob.glob(pattern))
    
    # Count the files found
    file_count = len(pso_files)
    
    # Remove each file
    for file_path in pso_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    print(f"Removed {file_count} PSO-related files")
four()
five()
coord()
par()
clean_parsec_de()
clean_parsec_pso()