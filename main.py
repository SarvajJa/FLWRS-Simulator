import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime

# GPU detection (automatic)
try:
    import cupy as cp
    use_gpu = True
    print("GPU acceleration ENABLED")
except ImportError:
    use_gpu = False
    print("GPU acceleration DISABLED")

# ========== FLWRS PARAMETERS ==========
N = 1024                  # Spatial points
alpha = 1.995             # Fractal dimension parameter
t_span = (0, 50)          # Simulation time range
dx = 0.1                  # Spatial step
rho_crit = 1.2            # Critical density for CDMO formation
beta = 0.7                # Dark matter transition sharpness
gR = 0.12                 # Rectification strength
sigma_R = 1.5             # Rectification kernel width

# ========== INITIAL CONDITION ==========
def initial_wavefunction(x):
    """Create initial state: 3 Gaussian wave packets with phase modulation"""
    psi1 = 0.5 * np.exp(-(x+5)**2) * np.exp(1j * 2*x)
    psi2 = 0.7 * np.exp(-x**2) * np.exp(-1j * 3*x)
    psi3 = 0.6 * np.exp(-(x-5)**2)
    psi = psi1 + psi2 + psi3
    return psi / np.linalg.norm(psi)

# ========== FUNDAMENTAL OPERATORS ==========
def H_L(Psi):
    """Fractal Laplacian operator (Ĥ_L)"""
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    k_alpha = np.abs(k)**alpha
    return np.fft.ifft(k_alpha * np.fft.fft(Psi))

def H_P(Psi, gP=0.3):
    """Nonlinear self-interaction operator (Ĥ_P)"""
    return gP * np.abs(Psi)**2 * Psi

def H_D(Psi, Gamma=0.1):
    """Decoherence operator (Ĥ_D)"""
    return 1j * Gamma * (1 - np.abs(Psi)**2) * Psi

def H_DM(Psi):
    """Dark matter operator (Ĥ_DM)"""
    rho = np.abs(Psi)**2
    eta = 0.5 * (1 + np.tanh(10 * beta * (rho - rho_crit)))
    return -1j * eta * Psi

def H_R(Psi):
    """Rectification operator (Ĥ_R)"""
    rho = np.abs(Psi)**2 + 1e-10
    dS_drho = -np.log(rho) - 1
    dS_dPsi = dS_drho * Psi / (2 * rho)
    
    # Rectification kernel
    x = np.linspace(-10, 10, N)
    K_R = np.exp(-x**2 / (2 * sigma_R**2))
    K_R /= np.trapz(K_R)
    
    return -1j * gR * np.convolve(K_R, dS_dPsi, mode='same')

# ========== SOLVE FUNDAMENTAL EQUATION ==========
def total_H(Psi):
    """Sum all Hamiltonian components"""
    return (
        H_L(Psi) + 
        H_P(Psi) + 
        H_D(Psi) + 
        H_DM(Psi) + 
        H_R(Psi)
    )

def FED(t, Psi_flat):
    """Fundamental Equation of Dynamics solver"""
    Psi = Psi_flat[:N] + 1j * Psi_flat[N:]
    dPsi_dt = -1j * total_H(Psi)
    return np.concatenate((dPsi_dt.real, dPsi_dt.imag))

# ========== SIMULATION EXECUTION ==========
if __name__ == "__main__":
    print(f"Starting FLWRS Simulation v2.0 | {datetime.now()}")
    
    # Initialize spatial grid and wavefunction
    x = np.linspace(-20, 20, N)
    Psi0 = initial_wavefunction(x)
    
    # Run simulation
    solution = solve_ivp(
        FED, 
        t_span, 
        np.concatenate((Psi0.real, Psi0.imag)),
        t_eval=np.linspace(t_span[0], t_span[1], 50),
        method='BDF'
    )
    
    # Save results
    np.savez(
        'flwrs_results.npz',
        t=solution.t,
        Psi=solution.y,
        x=x,
        params={
            'N': N, 'alpha': alpha, 
            'rho_crit': rho_crit, 'gR': gR
        }
    )
    print("Simulation completed successfully! Results saved.")
